"""Session - A session with an execution environment."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

import uuid

from peteos.activeclass import ActiveClass
from peteos.chatbot import ChatBotManager, ChatHistory, Message, ContentPart
from peteos.logger import get_logger
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager

from peteos.replexecutionenvironment import REPLExecutionEnvironment

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

        if role.system_prompt:
            chat_history.append_message(Message(
                role="system",
                content=[ContentPart(part_type="text", text=role.system_prompt)]
            ))

        tool_list = tool_manager.get_tool_list()
        for tool in tool_list:
            chat_history.append_message(Message(
                role="tool",
                content=[ContentPart(
                    part_type="tool",
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters
                )]
            ))

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
        self.chat_history = chat_history if chat_history is not None else self._initialize_chat_history(role, tool_manager)
        self.chatbot_manager = chatbot_manager

        self.execution_environment = execution_environment if execution_environment is not None else REPLExecutionEnvironment(
            chatbot_manager=chatbot_manager,
            chat_history=self.chat_history,
            tool_manager=tool_manager,
            role=role,
            session=self,
        )

        self._pending_tool_calls: list[ToolCallRecord] = []

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

        Waits for the first event, then drains all accumulated events.
        Processes messages into chat history, handles tool approvals/denials.
        Runs the session loop only if at least one event was processed.
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
                    approved = await self._handle_approval(event)
                    if not approved:
                        break  # Tool denied — stop draining
                elif not isinstance(event, Message):
                    continue  # Skip other event types

            # Process accumulated history only if we processed events
            if self.is_running() and events_processed > 0:
                await self._run_session_loop()

    async def _run_session_loop(self) -> None:
        """Run step loop until done, tool_pending, or interrupted."""
        while self.is_running():
            status, data = await self.execution_environment.step()
            if status == "done":
                break
            elif status == "tool_pending":
                tool_call = data["tool_call"]
                record = ToolCallRecord(
                    tool_call_id=tool_call.get("id", ""),
                    tool_call=tool_call,
                    approval_status=ToolApprovalStatus.PENDING,
                    execution_status=ToolExecutionStatus.WAITING_FOR_APPROVAL,
                )
                self._pending_tool_calls.append(record)
                return  # Exit step loop, wait for ApprovalEvent
            # "continue" -> loop back to step()

    def _find_pending_record(self, tool_call_id: str) -> Optional[ToolCallRecord]:
        """Find a pending tool call record by its tool_call_id."""
        for record in self._pending_tool_calls:
            if record.tool_call_id == tool_call_id:
                return record
        return None

    def get_pending_tool_calls(self) -> List[ToolCallRecord]:
        """Return all currently pending tool calls."""
        return list(self._pending_tool_calls)

    async def _handle_approval(self, event: ApprovalEvent) -> bool:
        """Handle an approval/denial event for a pending tool call.

        Returns True if the tool was approved (keep draining queue),
        False if the tool was denied (stop draining, session loop will run below).
        """
        record = self._find_pending_record(event.tool_call_id)
        if record is None:
            _logger.warning("No pending tool call found for tool_call_id=%s", event.tool_call_id)
            return True

        if event.approved:
            record.approval_status = ToolApprovalStatus.APPROVED
            record.execution_status = ToolExecutionStatus.EXECUTING
            await self.execution_environment.execute_pending_tool(event.tool_call)
            record.execution_status = ToolExecutionStatus.EXECUTED
            self._pending_tool_calls.remove(record)
            return True
        else:
            record.approval_status = ToolApprovalStatus.DENIED
            record.execution_status = ToolExecutionStatus.DENIED
            tool_name = event.tool_call.get("name", "unknown")
            denial_msg = event.tool_call.get("denied_reason", "Tool call was denied by user.")
            msg = Message(
                role="tool_result",
                content=[ContentPart(
                    part_type="tool_result",
                    name=tool_name,
                    content=denial_msg,
                )],
            )
            self.chat_history.append_message(msg)
            await self.execution_environment._call_hooks("after_tool_execution", event.tool_call, denial_msg, False)
            self._pending_tool_calls.remove(record)
            return False

    @staticmethod
    def load_from_json(
        json_data: dict,
        chatbot_manager: ChatBotManager,
        role_manager: RoleManager,
        tool_manager: ToolManager
    ) -> "Session":
        uuid_str = json_data.get("uuid")
        role_name = json_data["role"]
        chat_history_data = json_data.get("chat_history", [])

        role = role_manager.get_role(role_name)
        if role is None:
            raise ValueError(f"Role '{role_name}' not found in RoleManager")

        for tool_name in role.required_tools:
            if tool_manager.get_tool(tool_name) is None:
                raise ValueError(
                    f"Role '{role_name}' requires tool '{tool_name}', "
                    f"but it's not registered in tool_manager"
                )

        messages = [
            Message.from_dict(
                {"role": msg.get("role", "user"), "content": msg.get("content", [])},
                creation_timestamp=datetime.fromisoformat(msg["creation_timestamp"]) if "creation_timestamp" in msg else None,
                message_id=msg.get("id")
            )
            for msg in chat_history_data
        ]
        chat_history = ChatHistory()
        for msg in messages:
            chat_history.append_message(msg)

        session_uuid = uuid.UUID(uuid_str) if uuid_str else None

        env = REPLExecutionEnvironment(
            chatbot_manager=chatbot_manager,
            chat_history=chat_history,
            tool_manager=tool_manager,
            role=role,
        )

        return Session(
            role=role,
            tool_manager=tool_manager,
            chatbot_manager=chatbot_manager,
            chat_history=chat_history,
            session_uuid=session_uuid,
            execution_environment=env,
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
