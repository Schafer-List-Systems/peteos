"""Runner — active loop with message queue, approvals, and the REPL reasoning loop."""

from __future__ import annotations

import asyncio
import json
import uuid as _uuid
from typing import TYPE_CHECKING

from peteos.conversation.session import SessionState
from peteos.sandbox import Sandbox
from peteos.chatbot import ChatBot, ChatBotManager, Message

from peteos.utils.activeclass import ActiveClass
from peteos.engine.exec_status import ExecStatus
from peteos.engine.executionenvironment import (
    ApprovalEvent,
    ExecutionEnvironment,
    ToolApprovalStatus,
    ToolCallGroup,
    ToolCallRecord,
    ToolExecutionStatus,
)
from peteos.utils import get_logger
from peteos.persona.role import Role

if TYPE_CHECKING:
    from peteos.engine.channel import Channel
    from peteos.conversation.session import Session
    from peteos.persona.agent import Agent

_logger = get_logger(__name__)


class Runner(ActiveClass):
    """Active event-loop with message queue and the REPL reasoning loop.

    Wraps ActiveClass's base event loop and adds:

    - Message queue for incoming messages (``queue_message``)
    - Channel subscriptions and notification publishing
    - ``SessionState`` key-value store (via ``state`` property)
    - ``step()`` reasoning iteration (chatbot -> active_context -> EE -> status)

    Tool call management, approval workflow, and tool execution are
    owned by the ``ExecutionEnvironment``. The Runner delegates to it.

    The ``run()`` loop body consumes events and calls ``step()`` for
    each reasoning iteration.
    """

    def __init__(self, agent: "Agent", session_uuid: "_uuid.UUID", chatbot: "ChatBot | None" = None) -> None:
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
        self._sandbox: Sandbox | None = None

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def agent(self) -> "Agent":
        """Return the owning Agent."""
        return self._agent

    @property
    def session(self) -> "Session":
        """Return the Session this runner controls."""
        return self._session

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
    def state(self) -> SessionState:
        """Access the session's mutable state store."""
        return self._session.state

    @property
    def sandbox(self) -> Sandbox | None:
        """Return the sandbox cached on this runner, or None."""
        return self._sandbox

    @sandbox.setter
    def sandbox(self, value: Sandbox | None) -> None:
        self._sandbox = value

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
        from peteos.engine.channel import NotificationEvent
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

    async def step(self) -> tuple[ExecStatus, Message | None]:
        """Execute one reasoning iteration: chatbot -> create tool group.

        If a foreground tool call group already exists, returns PENDING
        so the run loop can finish it before calling step() again.

        Otherwise, calls the chatbot, appends the response, and creates
        a foreground tool call group if tool calls are present.

        Returns:
            Tuple of (status, response_message).
        """
        # --- If there's a foreground tool call group, don't call chatbot ---
        foreground = self._execution_environment.get_foreground_group()
        if foreground is not None:
            _logger.debug("[runner] step(): Foreground tool call group exists, skipping chatbot.")
            return (ExecStatus.PENDING, None)

        # --- Phase 1: Call chatbot ---
        self._session.materialize()
        tdm = self._session.active_context.tool_definitions_message
        tool_count = len(tdm.content) if tdm and tdm.content else 0
        _logger.debug(
            "[runner] step(): Materialized context, tool_definitions_message has %d tools",
            tool_count,
        )
        await self.execution_environment.call_hooks("before_send_to_chatbot", self, self._session.active_context)
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
            elif part.type == "thinking":
                _logger.debug("[runner] step():   content_part[%d] thinking=%s", i, json.dumps(part.text))

        # --- Phase 2: Error handling ---
        if "error" in response.data:
            _logger.warning("Chatbot returned error, skipping response: %s", response.data["error"])
            return (ExecStatus.ERROR, None)

        # --- Phase 3: Use the normalized message from the chatbot ---
        response_msg: Message = response.message
        await self.append_and_notify(response_msg)
        content_parts = response_msg.content
        has_text_part = response.has_text_part

        # --- Phase 4: Create foreground tool call group if tool calls present ---
        if any(cp.type == "tool_use" for cp in content_parts):
            group_id = response_msg.id
            anchor_name = f"{group_id}:tool_result"
            self._execution_environment.create_tool_group(group_id, anchor_name)
            for cp in content_parts:
                if cp.type == "tool_use":
                    self._execution_environment.add_tool_call(cp)
            msg_index = self._session.active_context.get_anchor_msg_index("messages")
            self._session.active_context.add_anchor(anchor_name, msg_index, after_existing=False)

        if any(cp.type == "tool_use" for cp in content_parts):
            _logger.debug("[runner] step(): Tool calls present, returning CONTINUE for tool execution.")
            return (ExecStatus.CONTINUE, response_msg)

        if not has_text_part:
            _logger.debug("[runner] step(): Response contained only reasoning part(s).")
            return (ExecStatus.CONTINUE, response_msg)

        if self._agent.role.behavior_policy == "continuous":
            _logger.debug("[runner] step(): Continuous agent produced text, keeping loop active.")
            return (ExecStatus.CONTINUE, response_msg)

        _logger.debug("[runner] step(): Had final answer.")
        return (ExecStatus.FINISHED, response_msg)

    async def _append_result_message(self, group: "ToolCallGroup") -> bool:
        """Append the group's result message to the active context at the group's anchor.

        Returns True if a message was appended, False if the group had no
        result message (e.g. all tool calls were fire-and-forget).
        """
        result_msg = group.result_message
        if result_msg is not None:
            self._session.active_context.append(result_msg, anchor_point=group.anchor_name)
            return True
        return False

    # ------------------------------------------------------------------ #
    # Tool group processing — called from run loop
    # ------------------------------------------------------------------ #

    async def _handle_tool_group(self) -> bool:
        """Process all sequential approved tool calls.

        Returns True if the foreground group is done and the result message
        was appended to the context. In that case the caller should not
        continue — it should fall through to step().

        Returns False if the group is still pending (waiting for approval).
        In that case the caller should continue.
        """
        foreground = self._execution_environment.get_foreground_group()
        if foreground is None:
            return False

        while foreground.has_reviewed():
            record = foreground.pop_first_reviewed()
            tool_call = record.tool_call
            tool_name = tool_call.name

            # Already denied (e.g., tool not found) — skip hooks and execution
            if record.approval_status == ToolApprovalStatus.DENIED:
                _logger.debug("[runner] Tool call %s already denied, skipping", tool_name)
                break

            # on_tool_call hooks only fire on APPROVED records
            _invocation_hooks = self._session._invocation_hooks or {}
            if "on_tool_call" in _invocation_hooks:
                tool_args = json.loads(tool_call.arguments) if tool_call.arguments else {}
                ctx = {
                    "role": self._agent.role.name,
                    "session": self._session,
                    "tool_name": tool_name,
                    "arguments": tool_args,
                }
                for hook in _invocation_hooks["on_tool_call"]:
                    result = hook(ctx)
                    if result is not None:
                        record.approval_status = ToolApprovalStatus.DENIED
                        record.denied_reason = result
                        foreground.deny_all_remaining(result)
                        break

                if record.approval_status == ToolApprovalStatus.DENIED:
                    break

            result_str, success = await self._execution_environment.execute_and_inject(tool_call, runner=self)
            record.execution_status = ToolExecutionStatus.EXECUTED
            if not success and result_str.startswith("Error: Tool '"):
                _logger.debug("[runner] Tool %s not found", tool_name)
                break
            if not success:
                await self.execution_environment.call_hooks("after_tool_execution", self, tool_call, result_str, False)
                foreground.deny_all_remaining(f"Tool '{tool_name}' execution failed")
                _logger.debug("[runner] Tool %s failed", tool_name)
                break
            await self.execution_environment.call_hooks("after_tool_execution", self, tool_call, result_str or "None", True)
            _logger.debug("Tool %s returned: %s", tool_name, result_str)

        if foreground.is_done():
            self._execution_environment.close_foreground_group()
            return await self._append_result_message(foreground)

        # No reviewed calls — log the first un-reviewed tool call for debugging
        if foreground.records:
            first = foreground.records[0]
            args_preview = first.tool_call.arguments[:200] if first.tool_call.arguments else ""
            _logger.debug(
                "[runner] _handle_tool_group: "
                "approval_status=%s execution_status=%s tool=%s call_id=%s args=%s",
                first.approval_status,
                first.execution_status,
                first.tool_call.name, first.tool_call_id, args_preview,
            )

        return False

    # ------------------------------------------------------------------ #
    # Event loop (ActiveClass.run)
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        """Main reasoning loop.

        Drains the event queue, then calls ``step()`` to get a chatbot
        response and potentially create a foreground tool call group.
        When a foreground group exists, calls ``_handle_tool_group`` for
        each approved tool call.
        """
        have_new_message = False

        while self.is_running():
            foreground = self._execution_environment.get_foreground_group()

            # Drain the event queue
            if not have_new_message and not self.has_event() and (not foreground or not foreground.has_reviewed()):
                self._idle.set()
                if not await self._wait_for_event():
                    continue
                self._idle.clear()

            # drain event queue
            events_processed = 0
            while self.has_event():
                event = self.event_queue.get_nowait()
                if event is None:
                    continue

                events_processed += 1

                if isinstance(event, Message):
                    await self.append_and_notify(event)
                    have_new_message = True
                elif isinstance(event, ApprovalEvent):
                    self._execution_environment._handle_approval(event)
                else:
                    events_processed -= 1
                    continue

            # If a foreground tool group exists, process approved tool calls
            if foreground is not None:
                if not await self._handle_tool_group() and not have_new_message:
                    continue

            # Send context to chatbot and get a response, already passing the tool calls
            try:
                status, response_msg = await self.step()
            except Exception as e:
                _logger.error("[runner] step() raised exception: %s: %r", type(e).__name__, e)
                self._idle.set()
                continue
            finally:
                have_new_message = False

            _logger.debug("[runner] step() returned status=%s", status)
            hook_status = await self.execution_environment.call_hooks("after_step", status)
            hook_return = hook_status if hook_status is not None else status
            _logger.debug("[runner] after_step hook returned status=%s, final=%s", hook_status, hook_return)

            if hook_return == ExecStatus.ERROR:
                _logger.debug("[runner] run(): Execution error, exiting loop.")
                break
            _logger.debug("[runner] run(): Iterating...")

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
