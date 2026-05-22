"""Session - A session with an execution environment."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field

from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Set

import uuid

if TYPE_CHECKING:
    from peteos.channels.channel import Channel

from peteos.activeclass import ActiveClass
from peteos.chatbot import ChatBotManager, ChatHistory, Message, ContentPart, SystemPromptMessage, ToolDefinitionsMessage
from peteos.logger import get_logger
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager

from peteos.replexecutionenvironment import REPLExecutionEnvironment
from peteos.executionenvironment import ExecStatus

_logger = get_logger(__name__)


class ToolApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"


class ToolExecutionStatus(str, Enum):
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    DENIED = "denied"
    EXECUTING = "executing"
    EXECUTED = "executed"
    ABORTED = "aborted"


@dataclass
class ToolCallRecord:
    """Tracks the lifecycle of a tool call through approval and execution."""
    tool_call_id: str
    tool_call: dict
    approval_status: ToolApprovalStatus = ToolApprovalStatus.PENDING
    execution_status: ToolExecutionStatus = ToolExecutionStatus.WAITING_FOR_APPROVAL
    nextcloud_message_id: Optional[str] = None
    execution_result: Optional[str] = None
    execution_success: Optional[bool] = None
    denied_reason: Optional[str] = None


@dataclass
class ApprovalEvent:
    """Event pushed to Session.event_queue to signal approval of a tool call."""
    tool_call_id: str = ""
    tool_call: dict = field(default_factory=dict)
    approved: bool = True


class ToolApprovalPending(Exception):
    """Raised when a tool call requires user approval."""
    def __init__(self, tool_call: dict) -> None:
        self.tool_call = tool_call
        super().__init__("Tool approval pending")


class Session(ActiveClass):
    """A session with an execution environment.

    Session extends ActiveClass so it has its own event loop for:
    - Processing incoming user messages
    - Driving the execution environment
    - Waiting for tool call approval from channels

    The event loop runs continuously once started. Messages are queued
    via queue_message() which is non-blocking.
    """

    @staticmethod
    def _initialize_chat_history(role: Role, tool_manager: ToolManager) -> ChatHistory:
        chat_history = ChatHistory()

        # System prompt anchored at front (persists, appears first to LLM)
        if role.system_prompt or role.system_prompt_hooks:
            chat_history.append_message(
                SystemPromptMessage(hooks=role._all_hooks),
                anchor="front"
            )

        # Tool definitions anchored at back (persist, appear after conversation)
        chat_history.append_message(
            ToolDefinitionsMessage(tool_manager=tool_manager),
            anchor="front"
        )

        return chat_history

    def __init__(
        self,
        role: Role,
        tool_manager: ToolManager,
        chatbot_manager: ChatBotManager,
        chat_history: Optional[ChatHistory] = None,
        session_uuid: Optional[uuid.UUID] = None,
        execution_environment: Optional[REPLExecutionEnvironment] = None
    ):
        super().__init__()
        self.uuid = session_uuid if session_uuid is not None else uuid.uuid4()
        self.role = role
        self.auto_approve_tools: list[str] = list(role.auto_approve_tools)
        self.chat_history = chat_history if chat_history is not None else self._initialize_chat_history(role, tool_manager)
        self.chatbot_manager = chatbot_manager
        self._channels: Set[Channel] = set()

        self.execution_environment = execution_environment if execution_environment is not None else REPLExecutionEnvironment(
            chatbot_manager=chatbot_manager,
            chat_history=self.chat_history,
            tool_manager=tool_manager,
            role=role,
        )

        self._pending_tool_calls: list[ToolCallRecord] = []
        self.tool_failure_policy: str = "continue"

    def subscribe(self, channel: "Channel") -> bool:
        """Subscribe a channel to notifications for this session.

        Returns False if the channel is already subscribed.
        """
        if channel in self._channels:
            return False  # already subscribed
        self._channels.add(channel)
        return True

    def unsubscribe(self, channel: "Channel") -> bool:
        """Unsubscribe a channel from notifications for this session.

        Returns False if the channel is not subscribed.
        """
        if channel not in self._channels:
            return False  # not subscribed
        self._channels.remove(channel)
        return True

    async def queue_message(self, message: Message) -> None:
        """Queue a message for processing.

        Non-blocking. Pushes to event_queue and starts the event loop if not running.
        The Session.run() loop processes the message and runs the
        execution environment.

        Args:
            message: The message to queue.
        """
        if not self.is_running():
            await self.start()
        self.push_event(message)

    async def run(self) -> None:
        """Main event loop for the session.

        In each iteration: waits for an event, drains all accumulated events,
        calls step(), then evaluates the status. Events are drained after
        every step() so messages arriving during execution are picked up.
        """
        while self.is_running():
            # Block until at least one event arrives
            if not await self._wait_for_event():
                continue

            # Drain all accumulated events
            events_processed = 0
            while self.has_event():
                event = self.event_queue.get_nowait()
                if event is None:
                    continue

                events_processed += 1

                if isinstance(event, Message):
                    self.chat_history.append_message(event)
                elif isinstance(event, ApprovalEvent):
                    self._handle_approval(event)
                else:
                    events_processed -= 1
                    continue  # Skip other event types

            if events_processed == 0:
                continue

            status, _ = await self.execution_environment.step(self)

            if status in (ExecStatus.FINISHED, ExecStatus.INTERRUPTED, ExecStatus.ERROR):
                break
            elif status == ExecStatus.PENDING:
                continue  # Re-enter: check for approvals in drained events
            elif status in (ExecStatus.TOOL_NOT_FOUND, ExecStatus.TOOL_DENIED):
                continue  # Re-enter: let chatbot handle the error/denial message
            elif status == ExecStatus.TOOL_FAILED:
                # Abort all remaining pending tool calls
                for record in self._pending_tool_calls:
                    if record.approval_status == ToolApprovalStatus.PENDING:
                        record.approval_status = ToolApprovalStatus.DENIED
                        record.execution_status = ToolExecutionStatus.ABORTED
                        record.tool_call["denied_reason"] = "Tool call was aborted due to a previous tool failure."
                _logger.debug("[session] Aborted all pending tool calls due to tool failure")
                continue  # tool errors can be handled by chatbot
            # CONTINUE -> loop back to step()

    def add_tool_call(self, tool_call: dict) -> None:
        """Add a tool call to the pending list, auto-approving if applicable."""
        tc_id = tool_call.get("id")
        if tc_id is None:
            raise ValueError("Tool call missing required 'id' field")
        for record in self._pending_tool_calls:
            if record.tool_call_id == tc_id:
                raise ValueError(f"Duplicate tool call id: {tc_id}")
        tool_name = tool_call.get("name", "")
        if tool_name in self.auto_approve_tools:
            self._pending_tool_calls.append(
                ToolCallRecord(
                    tool_call_id=tc_id,
                    tool_call=tool_call,
                    approval_status=ToolApprovalStatus.APPROVED,
                    execution_status=ToolExecutionStatus.EXECUTING,
                )
            )
        else:
            self._pending_tool_calls.append(
                ToolCallRecord(
                    tool_call_id=tc_id,
                    tool_call=tool_call,
                    approval_status=ToolApprovalStatus.PENDING,
                    execution_status=ToolExecutionStatus.WAITING_FOR_APPROVAL,
                )
            )

    def has_pending_tool_call(self) -> bool:
        """Check if the first pending tool call status is no longer PENDING."""
        for record in self._pending_tool_calls:
            if record.approval_status == ToolApprovalStatus.PENDING:
                return True
        return False

    def has_unfinished_tool_call(self) -> bool:
        """Check if the first pending tool call status is no longer PENDING."""
        for record in self._pending_tool_calls:
            if record.execution_status != ToolExecutionStatus.EXECUTED:
                return True
        return False

    def has_reviewed_tool_call(self) -> bool:
        """Check if the first pending tool call status is no longer PENDING."""
        for record in self._pending_tool_calls:
            if record.approval_status != ToolApprovalStatus.PENDING:
                return True
            return False
        return False

    def pop_pending_tool_call(self) -> None:
        """Remove the first tool call from the pending list."""
        if not self._pending_tool_calls:
            raise RuntimeError("No pending tool calls")
        tool_call = self._pending_tool_calls[0]
        self._pending_tool_calls.pop(0)
        return tool_call

    def _find_pending_record(self, tool_call_id: str) -> Optional[ToolCallRecord]:
        """Find a pending tool call record by its tool_call_id."""
        for record in self._pending_tool_calls:
            if record.tool_call_id == tool_call_id:
                return record
        return None

    def is_tool_call_pending(self, tool_call_id: str) -> bool:
        """Check if a tool call needs user approval (approval status is PENDING)."""
        record = self._find_pending_record(tool_call_id)
        return record is not None and record.approval_status == ToolApprovalStatus.PENDING

    def get_pending_tool_calls(self) -> List[ToolCallRecord]:
        """Return all currently pending tool calls."""
        return list(self._pending_tool_calls)

    def _handle_approval(self, event: ApprovalEvent) -> bool:
        """Handle an approval/denial event for a pending tool call.

        Returns True if the tool was approved (keep draining queue),
        False if the tool was denied (stop draining, session loop will run below).
        """
        record = self._find_pending_record(event.tool_call_id)
        if record is None:
            _logger.error("No pending tool call found for tool_call_id=%s", event.tool_call_id)
            # TODO: raise an error

        if event.approved:
            record.approval_status = ToolApprovalStatus.APPROVED
            return True
        else:
            record.approval_status = ToolApprovalStatus.DENIED
            return False

    def publish_notification(
            self,
            message: Message
    ) -> None:
        """Publish a notification to all subscribed channels."""
        from peteos.channels.channel import NotificationEvent
        self.execution_environment._call_hooks("before_notification_publish", self, message)
        for channel in self._channels:
            channel.push_event(NotificationEvent(self.uuid, message))

    @staticmethod
    def load_from_json(
        json_data: dict,
        chatbot_manager: ChatBotManager,
        role_manager: RoleManager,
        tool_manager: ToolManager
    ) -> "Session":
        uuid_str = json_data.get("uuid")
        role_name = json_data["role"]
        chat_history_data = json_data.get("chat_history", {})

        role = role_manager.get_role(role_name)
        if role is None:
            raise ValueError(f"Role '{role_name}' not found in RoleManager")

        for tool_name in role.required_tools:
            if tool_manager.get_tool(tool_name) is None:
                raise ValueError(
                    f"Role '{role_name}' requires tool '{tool_name}', "
                    f"but it's not registered in tool_manager"
                )

        chat_history = ChatHistory.from_dict(chat_history_data)
        session_uuid = uuid.UUID(uuid_str) if uuid_str else None

        return Session(
            role=role,
            tool_manager=tool_manager,
            chatbot_manager=chatbot_manager,
            chat_history=chat_history,
            session_uuid=session_uuid,
        )

    @staticmethod
    def load_from_file(
        file_path: str,
        chatbot_manager: ChatBotManager,
        role_manager: RoleManager,
        tool_manager: ToolManager
    ) -> "Session":
        with open(file_path, "r") as f:
            json_data = json.load(f)
        return Session.load_from_json(
            json_data,
            chatbot_manager,
            role_manager,
            tool_manager,
        )
