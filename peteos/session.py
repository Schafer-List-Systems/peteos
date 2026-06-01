"""Session - A session with an execution environment."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import uuid

if TYPE_CHECKING:
    from peteos.channels.channel import Channel

from peteos.activeclass import ActiveClass
from peteos.chatbot import ChatHistory, Message, ContentPart, SystemPromptMessage, ToolDefinitionsMessage
from peteos.logger import get_logger
from peteos.role import Role
from peteos.toolmanager import ToolManager

from peteos.replexecutionenvironment import REPLExecutionEnvironment
from peteos.executionenvironment import ExecStatus

_logger = get_logger(__name__)


def _yield_back() -> None:
    """Signal that the agent has finished its task and wants to yield control."""
    pass


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


class AgenticState:
    """Mutable key-value store for agents to leave intermediate state.

    Variables store strings. Five distinct methods with clear semantics:

    - get(name): returns the value or None if not found
    - create(name, value): creates a new variable (raises if exists)
    - update(name, old_value, new_value): compare-and-swap (both non-None)
    - delete(name): removes a variable (raises if not found)
    - list(): returns all variable names
    """

    def __init__(self) -> None:
        self._data: Dict[str, str] = {}

    def get(self, name: str) -> Optional[str]:
        """Get a variable's value.

        Args:
            name: Variable name.

        Returns:
            The variable's value, or None if it does not exist.
        """
        return self._data.get(name)

    def create(self, name: str, value: str) -> None:
        """Create a new variable.

        Args:
            name: Variable name.
            value: Non-empty string value.

        Raises:
            ValueError: If variable already exists or value is empty.
        """
        if name in self._data:
            raise ValueError(f"Variable '{name}' already exists in AgenticState")
        if value is None or value == "":
            raise ValueError("Value must be a non-empty string")
        self._data[name] = value

    def update(self, name: str, old_value: str, new_value: str) -> None:
        """Update a variable with compare-and-swap semantics.

        Both old_value and new_value must be non-None strings.

        Args:
            name: Variable name.
            old_value: Expected current value.
            new_value: New value to set.

        Raises:
            ValueError: If old_value or new_value is None, or
                if the variable doesn't exist or the current value
                doesn't match old_value.
        """
        if old_value is None:
            raise ValueError("old_value must not be None")
        if new_value is None:
            raise ValueError("new_value must not be None")

        if name not in self._data:
            raise KeyError(f"Variable '{name}' does not exist in AgenticState")

        current = self._data[name]
        if current != old_value:
            raise ValueError(
                f"Variable '{name}' has value {current!r}, expected {old_value!r}"
            )
        self._data[name] = new_value

    def delete(self, name: str) -> None:
        """Delete a variable.

        Args:
            name: Variable name.

        Raises:
            KeyError: If variable does not exist.
        """
        if name not in self._data:
            raise KeyError(f"Variable '{name}' does not exist in AgenticState")
        del self._data[name]

    def list(self) -> List[str]:
        """Return a list of all variable names.

        Returns:
            List of variable names.
        """
        return list(self._data.keys())


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

        # Tool definitions anchored at front (persist, appear after conversation)
        chat_history.append_message(
            ToolDefinitionsMessage(tool_manager=tool_manager, tool_filter=role.tool_filter),
            anchor="front"
        )

        # Continuous roles get a persistent yield_back reminder from the assistant's perspective
        #if role.behavior_policy == "continuous":
        #    chat_history.append_message(
        #        Message(
        #            role="assistant",
        #            content=[ContentPart(
        #                part_type="thinking",
        #                text="I must call the `yield_back()` tool when I want to yield back control to the user.",
        #            )],
        #        ),
        #        anchor="back"
        #    )

        return chat_history

    def __init__(
        self,
        role: Role,
        tool_manager: ToolManager,
        chat_history: Optional[ChatHistory] = None,
        session_uuid: Optional[uuid.UUID] = None,
        execution_environment: Optional[REPLExecutionEnvironment] = None
    ):
        super().__init__()
        self.uuid = session_uuid if session_uuid is not None else uuid.uuid4()
        self.role = role
        self.tool_manager = tool_manager
        self.auto_approve_tools: list[str] = list(role.auto_approve_tools)
        if role.behavior_policy == "continuous":
            tool_manager.register_tool(func=_yield_back, name="yield_back", description="Hand back control to the user to wait for further input. Not calling yield_back forces you to create more output!")
            self.auto_approve_tools.append("yield_back")
            if role.tool_filter is None:
                role.tool_filter = []
            if "yield_back" not in role.tool_filter:
                role.tool_filter.append("yield_back")
        self.chat_history = chat_history if chat_history is not None else self._initialize_chat_history(role, tool_manager)
        self._channels: Set[Channel] = set()
        self._state = AgenticState()

        self.execution_environment = execution_environment if execution_environment is not None else REPLExecutionEnvironment(
            chat_history=self.chat_history,
            tool_manager=tool_manager,
            role=role,
        )

        self._pending_tool_calls: list[ToolCallRecord] = []
        self.tool_failure_policy: str = "continue"

    @property
    def state(self) -> AgenticState:
        """Access the session's mutable state store."""
        return self._state

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
        When step() returns a reentry status (PENDING, tool errors), the loop
        re-enters step() immediately without waiting for new events.
        """
        need_reentry = False

        while self.is_running():
            if not need_reentry:
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
                    await self.append_and_notify(event)
                elif isinstance(event, ApprovalEvent):
                    self._handle_approval(event)
                else:
                    events_processed -= 1
                    continue  # Skip other event types

            if not need_reentry and events_processed == 0:
                continue
            need_reentry = False

            status, _ = await self.execution_environment.step(self)
            await self.execution_environment._call_hooks("after_step", status)

            if status == ExecStatus.INTERRUPTED:
                _logger.debug("[session] run(): Execution interrupted, exiting loop.")
                break
            if status == ExecStatus.ERROR:
                _logger.debug("[session] run(): Execution error, exiting loop.")
                break
            if status == ExecStatus.FINISHED:
                _logger.debug("[session] run(): Step finished, waiting for next events.")
                continue
            if status == ExecStatus.PENDING:
                _logger.debug("[session] run(): Pending tool approval, re-entering step.")
                continue
            if status in (ExecStatus.TOOL_NOT_FOUND, ExecStatus.TOOL_DENIED):
                _logger.debug("[session] run(): Tool call failed/denied, re-entering step loop.")
                need_reentry = True
                continue
            if status == ExecStatus.TOOL_FAILED:
                # Abort all remaining pending tool calls
                for record in self._pending_tool_calls:
                    if record.approval_status == ToolApprovalStatus.PENDING:
                        record.approval_status = ToolApprovalStatus.DENIED
                        record.execution_status = ToolExecutionStatus.ABORTED
                        record.tool_call["denied_reason"] = (
                            "Tool call was aborted due to a previous tool failure."
                        )
                _logger.debug("[session] Aborted all pending tool calls due to tool failure")
                need_reentry = True
                continue
            # CONTINUE -> step returned success, re-enter to process tool results
            need_reentry = True
            continue

    def add_tool_call(self, tool_call: dict) -> None:
        """Add a tool call to the pending list, auto-approving if applicable."""
        tc_id = tool_call.get("id")
        if tc_id is None:
            raise ValueError("Tool call missing required 'id' field")
        for record in self._pending_tool_calls:
            if record.tool_call_id == tc_id:
                raise ValueError(f"Duplicate tool call id: {tc_id}")
        tool_name = tool_call.get("name", "")

        # Validate tool exists
        if not self.tool_manager.get_tool(tool_name):
            tool_call["denied_reason"] = f"Tool '{tool_name}' is not available for this agent"
            self._pending_tool_calls.append(
                ToolCallRecord(
                    tool_call_id=tc_id,
                    tool_call=tool_call,
                    approval_status=ToolApprovalStatus.DENIED,
                    execution_status=ToolExecutionStatus.DENIED,
                )
            )
            _logger.warning("[session] Tool call '%s' denied: tool not found", tool_name)
            return

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

    async def append_and_notify(self, message: Message) -> None:
        """Append a message to chat history and publish a notification.

        Replaces the inline pattern:
            self.chat_history.append_message(msg)
            self.publish_notification(msg)

        Calls the ``after_message_append`` hook between append and publish.

        Args:
            message: The message to append and notify on.
        """
        self.chat_history.append_message(message)
        await self.execution_environment._call_hooks("after_message_append", self, message)
        await self.publish_notification(message)

    async def publish_notification(
            self,
            message: Message
    ) -> None:
        """Publish a notification to all subscribed channels."""
        from peteos.channels.channel import NotificationEvent
        await self.execution_environment._call_hooks("before_notification_publish", self, message)
        for channel in self._channels:
            channel.push_event(NotificationEvent(self.uuid, message))

def _extract_last_assistant_text(chat_history: "ChatHistory") -> str:
    """Extract the text of the last assistant message from chat history."""
    for msg in reversed(chat_history.messages):
        if msg.get_role() == "assistant":
            texts = [part.text for part in msg.content if part.type == "text" and part.text]
            return " ".join(texts).replace("  ", " ")
    return ""


async def invoke_agent(
    prompt: str,
    agent: "Agent | None" = None,
    *,
    existing_session: Optional["Session"] = None,
    keep_session: bool = False,
    timeout: float,
) -> Dict[str, Any]:
    """Invoke an agent role and return its final answer.

    Creates a Session (or uses an existing one), queues a user message,
    waits for processing to complete, and returns the assistant's answer.

    Args:
        prompt: The message / prompt to send to the agent.
        agent: Optional Agent with an associated role and tool_manager.
               If provided, creates a new session via
               agent.create_session() with full hook support.
        existing_session: Optional pre-existing Session to reuse for
            continuation. Must already be started.
        keep_session: If False (default), stop the session before returning.
            If True, the returned dict includes the running session so
            the caller can queue more messages.
        timeout: Maximum seconds to wait for a response (user-configured, no default).

    Returns:
        dict with keys:
            - answer (str): The last assistant message text.
            - session (Session|None): The session (None if keep_session=False).
            - history (list[Message]): Full chat history for inspection.

    Raises:
        ValueError: If neither agent nor existing_session provided.
        asyncio.TimeoutError: If timeout expires before response.
        RuntimeError: If existing_session is not running.
    """
    from peteos.agent import Agent as AgentType

    # --- Session setup ---
    if existing_session is not None:
        session = existing_session
        if not session.is_running():
            raise RuntimeError(
                "existing_session is not running; pass keep_session=True "
                "or call session.start()"
            )
    elif agent is not None:
        session = await agent.create_session()
    else:
        raise ValueError(
            "Either agent or existing_session must be provided."
        )

    env = session.execution_environment

    # --- Register temporary hook to signal completion ---
    done: asyncio.Event = asyncio.Event()

    def on_finished(session: "Session", status: ExecStatus) -> None:
        if status == ExecStatus.FINISHED:
            done.set()

    env.register_hook("after_step", on_finished, session)
    session_created = agent is not None and existing_session is None

    try:
        # --- Queue message ---
        user_message = Message(
            role="user",
            content=[ContentPart(part_type="text", text=prompt)],
        )
        await session.queue_message(user_message)

        # --- Wait for processing to complete ---
        await asyncio.wait_for(done.wait(), timeout=timeout)

        # --- Extract answer ---
        answer = _extract_last_assistant_text(session.chat_history)
        success = True

    except Exception:
        raise
    finally:
        env.deregister_hook("after_step", on_finished)

    return_state = dict(session.state._data)

    if not success and keep_session:
        try:
            await session.stop()
        except Exception:
            pass
        if session_created:
            try:
                await agent.destroy_session(session.uuid)
            except Exception:
                pass
    elif keep_session:
        kept_session: Optional[Session] = session
    else:
        kept_session = None
        try:
            await session.stop()
        except Exception:
            pass
        if session_created:
            try:
                await agent.destroy_session(session.uuid)
            except Exception:
                pass

    return {
        "answer": answer,
        "session": kept_session,
        "history": list(session.chat_history.messages),
        "state": return_state,
    }
