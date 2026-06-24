"""Runner — active loop with message queue, approvals, and the REPL reasoning loop."""

from __future__ import annotations

import asyncio
import json
import uuid as _uuid
from typing import TYPE_CHECKING, Any, Optional

from peteos.chatbot import ChatBot, ChatBotManager, ContentPart, Message

from peteos.utils.activeclass import ActiveClass
from peteos.engine.exec_status import ExecStatus
from peteos.engine.executionenvironment import (
    ApprovalEvent,
    ExecutionEnvironment,
    ToolApprovalStatus,
    ToolCallRecord,
    ToolExecutionStatus,
)
from peteos.utils import get_logger
from peteos.persona.role import Role

if TYPE_CHECKING:
    from peteos.persona.channel import Channel
    from peteos.conversation.session import Session
    from peteos.persona.agent import Agent

_logger = get_logger(__name__)


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
        self._data: dict[str, str] = {}

    def get(self, name: str) -> Optional[str]:
        """Get a variable's value."""
        return self._data.get(name)

    def create(self, name: str, value: str) -> None:
        """Create a new variable."""
        if name in self._data:
            raise ValueError(f"Variable '{name}' already exists in AgenticState")
        if value is None:
            raise ValueError("Value must not be None")
        self._data[name] = value

    def update(self, name: str, old_value: str, new_value: str) -> None:
        """Update a variable with compare-and-swap semantics."""
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
        """Delete a variable."""
        if name not in self._data:
            raise KeyError(f"Variable '{name}' does not exist in AgenticState")
        del self._data[name]

    def list(self) -> list[str]:
        """Return a list of all variable names."""
        return list(self._data.keys())


class Runner(ActiveClass):
    """Active event-loop with message queue and the REPL reasoning loop.

    Wraps ActiveClass's base event loop and adds:

    - Message queue for incoming messages (``queue_message``)
    - Channel subscriptions and notification publishing
    - ``AgenticState`` key-value store
    - ``step()`` reasoning iteration (chatbot -> active_context -> EE -> status)

    Tool call management, approval workflow, and tool execution are
    owned by the ``ExecutionEnvironment``. The Runner delegates to it.

    The ``run()`` loop body consumes events and calls ``step()`` for
    each reasoning iteration.
    """

    def __init__(self, agent: "Agent", session_uuid: "uuid.UUID", chatbot: "ChatBot | None" = None) -> None:
        """Initialize the runner for a specific session.

        Grabs role, tool manager from the agent.
        Selects the chatbot from the ChatBotManager based on role.model.

        Args:
            agent: The Agent owning this session.
            session_uuid: The UUID of the session this runner controls.
            chatbot: Optional chatbot; if None, selects from manager based on role.model.
        """
        super().__init__()
        self._agent = agent
        self._session_uuid = session_uuid
        self.uuid = session_uuid
        self._state = AgenticState()
        self._channels: set[Channel] = set()
        self._idle: asyncio.Event = asyncio.Event()

        # Pull configuration from the agent (not the session data model)
        self._session: Session = agent.get_session(session_uuid)
        if self._session is None:
            raise ValueError(f"Session {session_uuid} not found in agent")

        self._execution_environment = ExecutionEnvironment(
            tool_manager=agent._tool_manager,
            role=agent.role,
            auto_approve_tools=list(agent.role.auto_approve_tools),
            tool_failure_policy="continue",
        )
        self._chatbot: ChatBot = chatbot or self._select_chatbot()

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def agent(self) -> "Agent":
        """Return the owning Agent."""
        return self._agent

    @property
    def session_uuid(self) -> "_uuid.UUID":
        """Return the session UUID this runner controls."""
        return self._session_uuid

    @property
    def role(self) -> "Role":
        """Return the role for this runner's session."""
        return self._agent.role

    @property
    def execution_environment(self) -> ExecutionEnvironment:
        """Return the execution environment for this runner's session."""
        return self._execution_environment

    @property
    def state(self) -> AgenticState:
        """Access the runner's mutable state store."""
        return self._state

    # ------------------------------------------------------------------ #
    # Chatbot selection
    # ------------------------------------------------------------------ #

    def _select_chatbot(self) -> ChatBot:
        """Select a ChatBot from the class-level manager based on role.model."""
        chatbots = ChatBotManager.list_chatbots(self._agent.role.model)
        if not chatbots:
            raise ValueError(
                f"No ChatBot found matching model pattern '{self._agent.role.model}' "
                f"for role '{self._agent.role.name}'"
            )
        return chatbots[0][1]

    # ------------------------------------------------------------------ #
    # Message queue
    # ------------------------------------------------------------------ #

    async def queue_message(self, message: Message) -> None:
        """Queue a message for processing.

        Non-blocking. Pushes to event_queue and starts the event loop if not running.
        The Runner.run() loop processes the message and runs the
        execution environment.

        Args:
            message: The message to queue.
        """
        if not self.is_running():
            await self.start()
        self._idle.clear()
        self.push_event(message)

    async def wait_for_idle(self, timeout: float | None = None) -> bool:
        """Wait until the runner becomes idle.

        Returns True if the runner became idle, False if the timeout expired.
        """
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # ------------------------------------------------------------------ #
    # Channel subscriptions & notifications
    # ------------------------------------------------------------------ #

    def subscribe(self, channel: "Channel") -> bool:
        """Subscribe a channel to notifications for this runner.

        Returns False if the channel is already subscribed.
        """
        if channel in self._channels:
            return False
        self._channels.add(channel)
        return True

    def unsubscribe(self, channel: "Channel") -> bool:
        """Unsubscribe a channel from notifications for this runner.

        Returns False if the channel is not subscribed.
        """
        if channel not in self._channels:
            return False
        self._channels.remove(channel)
        return True

    async def append_and_notify(self, message: Message) -> None:
        """Append a message to active context and publish a notification."""
        self._session.active_context.append(message)
        await self._call_after_message_append(message)
        await self.publish_notification(message)

    async def publish_notification(self, message: Message) -> None:
        """Publish a notification to all subscribed channels."""
        from peteos.persona.channel import NotificationEvent
        await self._call_before_notification_publish(message)
        for channel in self._channels:
            channel.push_event(NotificationEvent(self.uuid, message))

    # ------------------------------------------------------------------ #
    # Hook callbacks (delegated to execution environment)
    # ------------------------------------------------------------------ #

    async def _call_after_message_append(self, message: Message) -> None:
        """Delegate to execution environment's hook system."""
        if self.execution_environment:
            await self.execution_environment.call_hooks("after_message_append", self, message)

    async def _call_before_notification_publish(self, message: Message) -> None:
        """Delegate to execution environment's hook system."""
        if self.execution_environment:
            await self.execution_environment.call_hooks("before_notification_publish", self, message)

    # ------------------------------------------------------------------ #
    # step() — the core reasoning iteration
    # ------------------------------------------------------------------ #

    async def step(self) -> tuple[ExecStatus, dict | None]:
        """Execute one loop iteration: chatbot -> tools -> continue/exit.

        1. Call chatbot (skip on re-entry after tool_pending)
        2. Append assistant response to active_context; create tool groups
           and anchors for tool calls
        3. For each tool call: execute via execution_environment, inject
           result into deferred result message at group anchor
        4. Return a status telling the loop whether to continue, exit, or
           wait for tool approval.

        Returns:
            Tuple of (status, data).
        """
        # --- Phase 1: Call chatbot (skip if re-entering after tool_pending) ---
        has_text_part = False

        if not self._execution_environment.has_unfinished_tool_call():
            self._session.materialize()
            tdm = self._session.active_context.tool_definitions_message
            tool_count = len(tdm.content) if tdm and tdm.content else 0
            _logger.debug(
                "[runner] step(): Materialized context, tool_definitions_message has %d tools",
                tool_count,
            )
            await self._call_hooks("before_send_to_chatbot", self, self._session.active_context)
            response = await self._chatbot.send_context(self._session.active_context)
            async for _ in response:
                pass

            # --- Phase 1.5: Debug output ---
            _logger.debug(
                "[runner] step(): Chatbot response — has_text=%s, content_parts=%d, message_id=%s",
                response.has_text_part,
                len(response.message.content) if response.message else 0,
                response.message.id if response.message else None,
            )
            for i, part in enumerate(response.message.content if response.message else []):
                _logger.debug("[runner] step():   content_part[%d] type=%s", i, part.type)
                if part.type == "text":
                    _logger.debug("[runner] step():   content_part[%d] text=%s", i, json.dumps(part.text))
                elif part.type == "tool_use":
                    _logger.debug("[runner] step():   content_part[%d] tool_use(name=%s, arguments=%s)", i, part.name, part.arguments)

            # --- Phase 2: Error handling ---
            if "error" in response.data:
                _logger.warning("Chatbot returned error, skipping response: %s", response.data["error"])
                return (ExecStatus.ERROR, None)

            # --- Phase 3: Use the normalized message from the chatbot ---
            response_msg: Message = response.message
            await self.append_and_notify(response_msg)
            content_parts = response_msg.content
            has_text_part = response.has_text_part

            # --- Phase 4: Create tool groups and result anchors ---
            if any(cp.type == "tool_use" for cp in content_parts):
                group_id = response_msg.id
                anchor_name = f"{group_id}:tool_result"
                self._execution_environment.create_tool_group(group_id, anchor_name)
                # Add each tool_use call to the group (was inside the for loop
                # in the old session/executionenvironment code)
                for cp in content_parts:
                    if cp.type == "tool_use":
                        self._execution_environment.add_tool_call(cp, group_id)
                # Get the end-iterator from the messages anchor so the
                # negative offset stays correct even when other anchors
                # exist after it
                msg_index = self._session.active_context.get_anchor_msg_index("messages")
                self._session.active_context.add_anchor(anchor_name, msg_index, after_existing=False)

        # --- Phase 5: Execute tool calls (delegate to ExecutionEnvironment) ---
        did_tool_calls = self._execution_environment.has_reviewed_tool_call()
        while self._execution_environment.has_reviewed_tool_call():
            group = self._execution_environment.get_group(group_id := "")
            # Find a group that has a reviewed tool call
            found_group = None
            for g in self._execution_environment._groups.values():
                if g.has_reviewed():
                    found_group = g
                    break
            if found_group is None:
                break
            group = found_group
            group_id = group.id

            record = group.pop_first_reviewed()
            tool_call = record.tool_call
            tool_name = tool_call.name

            # Handle denied tool calls (status set by _handle_approval)
            if record.approval_status == ToolApprovalStatus.DENIED:
                denial_msg = record.denied_reason or "Tool call was denied by user."
                await self._call_hooks("after_tool_execution", self, tool_call, denial_msg, False)
                _logger.debug("[runner] step(): Tool call %s was denied by user", tool_name)
                return (ExecStatus.TOOL_DENIED, None)

            result_str, success = await self._execution_environment.execute_and_inject(tool_call, group_id, runner=self)
            if not success and result_str.startswith("Error: Tool '"):
                _logger.debug("[runner] step(): Tool %s not found", tool_name)
                return (ExecStatus.TOOL_NOT_FOUND, None)
            if not success:
                await self._call_hooks("after_tool_execution", self, tool_call, result_str, False)
                _logger.debug("[runner] step(): Tool %s failed", tool_name)
                return (ExecStatus.TOOL_FAILED, None)
            await self._call_hooks("after_tool_execution", self, tool_call, result_str or "None", True)
            _logger.debug("Tool %s returned: %s", tool_name, result_str)

        # Track whether any tool actually produced a non-None result
        any_tool_produced = any(
            g.result_message is not None
            and len(g.result_message.raw_dict.get("content", [])) > 0
            for g in self._execution_environment._groups.values()
        )
        if did_tool_calls and not self._execution_environment.has_pending_tool_call():
            if any_tool_produced:
                _logger.debug("[runner] step(): Did tool calls with results. Need to continue, such that the ChatBot can see the results.")
                #TODO: result msg is not appended to the context yet (note to take the right group anchor)
                return (ExecStatus.CONTINUE, None)
            _logger.debug("[runner] step(): Did tool calls but all returned None (fire-and-forget). Skipping re-entry.")
            return (ExecStatus.FINISHED, None)

        if has_text_part and not self._execution_environment.has_pending_tool_call():
            if self._agent.role.behavior_policy == "continuous":
                _logger.debug("[runner] step(): Continuous agent produced text, keeping loop active.")
                return (ExecStatus.CONTINUE, None)
            _logger.debug("[runner] step(): Had final answer.")
            return (ExecStatus.FINISHED, None)

        if self._execution_environment.has_pending_tool_call():
            _logger.debug("[runner] step(): Waiting for user review of pending tool calls.")
            return (ExecStatus.PENDING, None)

        # no tool calls, no pending tools, no text part: only reasoning...
        _logger.debug("[runner] step(): Response contained only reasoning part(s).")
        return (ExecStatus.CONTINUE, None)

    async def _call_hooks(self, hook_point: str, *args: Any) -> ExecStatus | None:
        """Delegate to execution environment's hook system."""
        if self.execution_environment:
            return await self.execution_environment.call_hooks(hook_point, *args)
        return None

    # ------------------------------------------------------------------ #
    # Event loop (ActiveClass.run)
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        """Main reasoning loop.

        Drains events, calls ``step()``, and
        re-enters when the step() returns a status that requires more
        processing.
        """
        need_reentry = False

        while self.is_running():
            need_reentry = need_reentry or self.has_event()
            if not need_reentry:
                self._idle.set()
                if not await self._wait_for_event():
                    continue
                self._idle.clear()

            events_processed = 0
            while self.has_event():
                event = self.event_queue.get_nowait()
                if event is None:
                    continue

                events_processed += 1

                if isinstance(event, Message):
                    await self.append_and_notify(event)
                elif isinstance(event, ApprovalEvent):
                    self._execution_environment._handle_approval(event)
                else:
                    events_processed -= 1
                    continue

            if not need_reentry and events_processed == 0:
                continue

            status, _ = await self.step()
            _logger.debug("[runner] step() returned status=%s", status)
            hook_status = await self._call_hooks("after_step", status)
            hook_return = hook_status if hook_status is not None else status
            _logger.debug("[runner] after_step hook returned status=%s, final=%s", hook_status, hook_return)

            need_reentry = False
            if hook_return == ExecStatus.ERROR:
                _logger.debug("[runner] run(): Execution error, exiting loop.")
                break
            if hook_return == ExecStatus.FINISHED:
                _logger.debug("[runner] run(): Step finished, waiting for next events.")
                continue
            if hook_return == ExecStatus.PENDING:
                _logger.debug("[runner] run(): Pending tool approval, re-entering step.")
                continue
            if hook_return in (ExecStatus.TOOL_NOT_FOUND, ExecStatus.TOOL_DENIED):
                _logger.debug("[runner] run(): Tool call failed/denied, re-entering step loop.")
                continue
            if hook_return == ExecStatus.TOOL_FAILED:
                for record in self._execution_environment.get_pending_tool_calls():
                    if record.approval_status == ToolApprovalStatus.PENDING:
                        record.approval_status = ToolApprovalStatus.DENIED
                        record.execution_status = ToolExecutionStatus.ABORTED
                        record.denied_reason = (
                            "Tool call was aborted due to a previous tool failure."
                        )
                _logger.debug("[runner] run(): Aborted all pending tool calls due to tool failure")
                continue
            if hook_return == ExecStatus.CONTINUE:
                need_reentry = True
                continue
            continue

    async def _wait_for_event(self, timeout: float | None = None) -> bool:
        """Block until an event arrives in the queue.

        Fast-path: if an event is already queued, returns True immediately.
        Otherwise waits on asyncio.Event for wake-up signal.

        Uses a retry loop to handle spurious wakeups.
        Returns True if an event is available, False on timeout.
        """
        if not self.event_queue.empty():
            return True

        self._event_trigger.clear()
        loop = asyncio.get_event_loop()
        deadline = (loop.time() + timeout) if timeout else None

        while self.is_running():
            remaining = (deadline - loop.time()) if deadline else None
            try:
                await asyncio.wait_for(
                    self._event_trigger.wait(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                return False

            if not self.event_queue.empty():
                return True

            remaining = (deadline - loop.time()) if deadline else None
            if remaining is not None and remaining <= 0:
                return False
