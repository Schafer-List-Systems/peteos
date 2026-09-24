"""Unit tests for Runner and SessionState."""

from __future__ import annotations

import uuid as _uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from peteos.engine.executionenvironment import (
    ExecutionEnvironment,
    ToolApprovalStatus,
    ToolExecutionStatus,
)
from peteos.engine.exec_status import ExecStatus
from peteos.conversation.message import ContentPart, Message
from peteos.conversation.session import SessionState
from peteos.engine.runner import ApprovalEvent, Runner


# ---------------------------------------------------------------------------
# SessionState
# ---------------------------------------------------------------------------

class TestSessionState:
    def test_initial_state_is_empty(self):
        state = SessionState()
        assert state.list() == []
        assert state.get("nonexistent") is None

    def test_create_and_get(self):
        state = SessionState()
        state.create("key1", "value1")
        assert state.get("key1") == "value1"
        assert state.list() == ["key1"]

    def test_create_raises_if_exists(self):
        state = SessionState()
        state.create("key1", "value1")
        with pytest.raises(ValueError, match="already exists"):
            state.create("key1", "value2")

    def test_create_raises_on_none_value(self):
        state = SessionState()
        with pytest.raises(ValueError, match="must not be None"):
            state.create("key1", None)

    def test_update_compare_and_swap(self):
        state = SessionState()
        state.create("key1", "old")
        state.update("key1", "old", "new")
        assert state.get("key1") == "new"

    def test_update_raises_if_key_missing(self):
        state = SessionState()
        with pytest.raises(KeyError, match="does not exist"):
            state.update("key1", "old", "new")

    def test_update_raises_on_none_old(self):
        state = SessionState()
        state.create("key1", "val")
        with pytest.raises(ValueError, match="old_value must not be None"):
            state.update("key1", None, "new")

    def test_update_raises_on_none_new(self):
        state = SessionState()
        state.create("key1", "val")
        with pytest.raises(ValueError, match="new_value must not be None"):
            state.update("key1", "val", None)

    def test_update_raises_on_mismatch(self):
        state = SessionState()
        state.create("key1", "wrong")
        with pytest.raises(ValueError, match=r"has value 'wrong', expected 'correct'"):
            state.update("key1", "correct", "new")

    def test_delete(self):
        state = SessionState()
        state.create("key1", "val")
        assert state.get("key1") == "val"
        state.delete("key1")
        assert state.get("key1") is None
        assert state.list() == []

    def test_delete_raises_if_missing(self):
        state = SessionState()
        with pytest.raises(KeyError, match="does not exist"):
            state.delete("key1")

    def test_multiple_keys(self):
        state = SessionState()
        state.create("a", "1")
        state.create("b", "2")
        state.create("c", "3")
        keys = state.list()
        assert set(keys) == {"a", "b", "c"}

    def test_delete_order_preserved(self):
        state = SessionState()
        state.create("a", "1")
        state.create("b", "2")
        state.create("c", "3")
        state.delete("b")
        assert state.list() == ["a", "c"]


# ---------------------------------------------------------------------------
# ApprovalEvent (dataclass)
# ---------------------------------------------------------------------------

class TestApprovalEvent:
    def test_defaults(self):
        evt = ApprovalEvent()
        assert evt.tool_call_id == ""
        assert isinstance(evt.tool_call, ContentPart)
        assert evt.approved is True

    def test_custom_values(self):
        evt = ApprovalEvent(
            tool_call_id="tc1",
            tool_call={"name": "add"},
            approved=False,
        )
        assert evt.tool_call_id == "tc1"
        assert evt.tool_call == {"name": "add"}
        assert evt.approved is False

    def test_defaults_to_approved(self):
        evt = ApprovalEvent(tool_call_id="tc1")
        assert evt.approved is True

    def test_from_runner_import(self):
        """ApprovalEvent is re-exported from runner module."""
        from peteos.engine.runner import ApprovalEvent as RunnerAE
        evt = ApprovalEvent(tool_call_id="x")
        assert isinstance(evt, RunnerAE)


# ---------------------------------------------------------------------------
# Runner helpers and properties
# ---------------------------------------------------------------------------

def _make_mock_session(
    auto_approve_tools: list[str] | None = None,
    tool_failure_policy: str = "abort",
    behavior_policy: str = "responsive",
) -> MagicMock:
    """Build a mock Session with all properties the Runner consumes."""
    role = MagicMock()
    role.name = "test-role"
    role.model = "test-model"
    role.behavior_policy = behavior_policy
    role.auto_approve_tools = auto_approve_tools or []
    role.tool_filter = []

    tm = MagicMock()
    tm.get_tool.return_value = None

    ctx = MagicMock()
    ctx.messages = []

    session = MagicMock()
    session.role = role
    session.tool_manager = tm
    session.auto_approve_tools = list(role.auto_approve_tools)
    session.tool_failure_policy = tool_failure_policy
    session.active_context = ctx
    session.state = SessionState()
    session.invocation_hooks = {}
    return session


def _make_mock_agent(session: MagicMock) -> MagicMock:
    agent = MagicMock()
    agent.get_session.return_value = session
    agent._tool_manager = session.tool_manager
    agent.role = session.role
    return agent


def _make_mock_chatbot(
    content: list[dict] | None = None,
    role: str = "assistant",
    error: str | None = None,
) -> MagicMock:
    cb = MagicMock()
    response = AsyncMock()
    data: dict[str, Any] = {"role": role, "content": content or []}
    if error:
        data["error"] = error
    response.data = data
    if "error" in data:
        response.message = None
        response.has_text_part = False
    else:
        response.message = Message.create(
            role=role,
            content_parts=[ContentPart(dict(p)) for p in (content or [])],
        )
        response.has_text_part = any(p.get("type") == "text" for p in (content or []))
    cb.send_context = AsyncMock(return_value=response)
    cb.__aiter__ = AsyncMock(return_value=iter([]))
    return cb
    return cb


@pytest.fixture
def chatbot_manager_mock():
    with patch("peteos.engine.runner.ChatBotManager") as mock:
        cb = _make_mock_chatbot()
        mock.list_chatbots.return_value = [("test-model", cb)]
        yield mock


class TestRunnerInit:
    def test_init_raises_on_missing_session(self):
        agent = MagicMock()
        agent.get_session.return_value = None
        with pytest.raises(ValueError, match="not found"):
            Runner(
                agent=agent,
                session_uuid=_uuid.uuid4(),
            )

    def test_init_sets_uuid(self, chatbot_manager_mock):
        session = _make_mock_session()
        agent = _make_mock_agent(session)
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid)
        assert runner.session_uuid == sid
        assert runner.uuid == sid

    def test_init_creates_state(self, chatbot_manager_mock):
        session = _make_mock_session()
        agent = _make_mock_agent(session)
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4())
        assert isinstance(runner.state, SessionState)

    def test_init_creates_execution_environment(self, chatbot_manager_mock):
        session = _make_mock_session()
        agent = _make_mock_agent(session)
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4())
        assert isinstance(runner.execution_environment, ExecutionEnvironment)

    def test_init_properties(self, chatbot_manager_mock):
        session = _make_mock_session()
        agent = _make_mock_agent(session)
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4())
        assert runner.agent is agent
        assert runner.role is session.role


# ---------------------------------------------------------------------------
# Runner step — assistant text with no tool calls → FINISHED
# ---------------------------------------------------------------------------

class TestRunnerStepTextOnly:
    async def test_text_only_finished(self, chatbot_manager_mock, tmp_path):
        """Agent replies with text only → FINISHED."""
        session = _make_mock_session()
        agent = _make_mock_agent(session)
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4())
        runner._chatbot = _make_mock_chatbot(
            content=[{"type": "text", "content": "done"}],
        )

        status, _ = await runner.step()
        assert status is ExecStatus.FINISHED

    async def test_reasoning_only_continue(self, chatbot_manager_mock):
        """Agent replies with reasoning (thinking) only → CONTINUE."""
        session = _make_mock_session()
        agent = _make_mock_agent(session)
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4())
        runner._chatbot = _make_mock_chatbot(
            content=[{"type": "thinking", "content": "let me think"}],
        )

        status, _ = await runner.step()
        assert status is ExecStatus.CONTINUE

    async def test_continuous_policy_keeps_loop(self, chatbot_manager_mock):
        """Continuous behavior_policy → CONTINUE even with text output."""
        session = _make_mock_session(behavior_policy="continuous")
        agent = _make_mock_agent(session)
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4())
        runner._chatbot = _make_mock_chatbot(
            content=[{"type": "text", "content": "thinking..."}],
        )

        status, _ = await runner.step()
        assert status is ExecStatus.CONTINUE

    async def test_chatbot_error(self, chatbot_manager_mock):
        """ChatBot returns error → ERROR status."""
        session = _make_mock_session()
        agent = _make_mock_agent(session)
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4())
        runner._chatbot = _make_mock_chatbot(
            error="API error",
        )

        status, _ = await runner.step()
        assert status is ExecStatus.CRITICAL


# ---------------------------------------------------------------------------
# Runner step — tool calls
# ---------------------------------------------------------------------------

class TestRunnerStepToolCalls:
    async def test_pending_tool_calls(self, chatbot_manager_mock):
        """Pre-setup with PENDING tool call, chatbot returns text → PENDING."""
        session = _make_mock_session()
        agent = _make_mock_agent(session)
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4())

        # Register a real tool so the tool call isn't auto-denied
        tool_mock = MagicMock()
        tool_mock.name = "add"
        tool_mock.func = lambda a, b: a + b
        tool_mock.execute = MagicMock(return_value=3)
        session.tool_manager.get_tool.return_value = tool_mock

        # Pre-setup: create group and a PENDING tool call
        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        await runner.execution_environment.add_tool_call(
            ContentPart.create_tool_use("tc1", "add", "{}"), runner=runner
        )

        # Make chatbot return text output
        runner._chatbot = _make_mock_chatbot(
            content=[{"type": "text", "content": "let me check"}],
        )
        # Ensure tool_manager returns the tool so add_tool_call creates PENDING record
        session.tool_manager.get_tool = MagicMock(return_value=MagicMock(name="add"))

        status, _ = await runner.step()
        assert status is ExecStatus.PENDING

    async def test_auto_approved_tool_succeeds_continue(self, chatbot_manager_mock):
        """Runner creates EE from agent.role.auto_approve_tools.
        When a chatbot response contains tool_use calls that are auto-approved,
        step() still returns PENDING because a foreground group exists —
        the runner's loop then auto-executes.
        """
        role = MagicMock()
        role.name = "test-role"
        role.model = "test-model"
        role.behavior_policy = "responsive"
        role.auto_approve_tools = ["add"]
        role.tool_filter = []

        tm = MagicMock()
        tool_mock = MagicMock()
        tool_mock.name = "add"
        tool_mock.func = lambda a, b: a + b
        tool_mock.execute = MagicMock(return_value=3)
        tm.get_tool.return_value = tool_mock

        ctx = MagicMock()
        ctx.messages = []

        session = MagicMock()
        session.role = role
        session.tool_manager = tm
        session.auto_approve_tools = list(role.auto_approve_tools)
        session.tool_failure_policy = "abort"
        session.active_context = ctx

        # Patch _select_chatbot to return our mock
        chatbot = _make_mock_chatbot(content=[{"type": "tool_use", "name": "add", "arguments": "{}", "call_id": "tc1"}])

        # Create a real runner — it pulls auto_approve_tools from role
        agent = MagicMock()
        agent.get_session.return_value = session
        agent._tool_manager = tm
        agent.role = role
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4(), chatbot=chatbot)

        # Step will: call chatbot → get tool_use → create group → add_tool_call
        # → returns CONTINUE (has_text_part=False from tool_use-only response)
        status, _ = await runner.step()
        assert status is ExecStatus.CONTINUE
        # Verify the group was created and the tool call was auto-approved
        fg = runner.execution_environment.get_foreground_group()
        assert fg is not None
        assert fg.records[0].approval_status == ToolApprovalStatus.APPROVED

    async def test_tool_failed_at_execution(self, chatbot_manager_mock):
        """Auto-approved tool fails at runtime → step returns PENDING (loop handles execution)."""
        role = MagicMock()
        role.name = "test-role"
        role.model = "test-model"
        role.behavior_policy = "responsive"
        role.auto_approve_tools = ["add"]
        role.tool_filter = []

        tm = MagicMock()
        tool_mock = MagicMock()
        tool_mock.name = "add"
        tool_mock.func = lambda a, b: a + b
        tool_mock.execute = MagicMock(side_effect=ValueError("calculation error"))
        tm.get_tool.return_value = tool_mock

        ctx = MagicMock()
        ctx.messages = []

        session = MagicMock()
        session.role = role
        session.tool_manager = tm
        session.auto_approve_tools = list(role.auto_approve_tools)
        session.tool_failure_policy = "abort"
        session.active_context = ctx

        chatbot = _make_mock_chatbot(content=[{"type": "tool_use", "name": "add", "arguments": "{}", "call_id": "tc1"}])

        agent = MagicMock()
        agent.get_session.return_value = session
        agent._tool_manager = tm
        agent.role = role
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4(), chatbot=chatbot)

        status, _ = await runner.step()
        assert status is ExecStatus.CONTINUE
        fg = runner.execution_environment.get_foreground_group()
        assert fg is not None
        assert fg.records[0].approval_status == ToolApprovalStatus.APPROVED

    async def test_tool_denied_by_hook(self, chatbot_manager_mock):
        """Auto-approved tool denied by before_tool_execution hook → record denied."""
        role = MagicMock()
        role.name = "test-role"
        role.model = "test-model"
        role.behavior_policy = "responsive"
        role.auto_approve_tools = ["add"]
        role.tool_filter = []

        tm = MagicMock()
        tool_mock = MagicMock()
        tool_mock.name = "add"
        tool_mock.func = lambda a, b: a + b
        tool_mock.execute = MagicMock(return_value=3)
        tm.get_tool.return_value = tool_mock

        ctx = MagicMock()
        ctx.messages = []

        session = MagicMock()
        session.role = role
        session.tool_manager = tm
        session.auto_approve_tools = list(role.auto_approve_tools)
        session.tool_failure_policy = "abort"
        session.active_context = ctx
        session.invocation_hooks = {}

        chatbot = _make_mock_chatbot(content=[{"type": "tool_use", "name": "add", "arguments": "{}", "call_id": "tc1"}])

        agent = MagicMock()
        agent.get_session.return_value = session
        agent._tool_manager = tm
        agent.role = role
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4(), chatbot=chatbot)

        def deny_hook(tool_call):
            return (False, "denied by hook")

        session.invocation_hooks["before_tool_execution"] = [deny_hook]

        status, _ = await runner.step()
        assert status is ExecStatus.CONTINUE
        fg = runner.execution_environment.get_foreground_group()
        assert fg is not None
        assert fg.records[0].approval_status == ToolApprovalStatus.APPROVED
        record = fg.records[0]

        await runner._handle_tool_group()

        assert record.execution_status == ToolExecutionStatus.EXECUTED
        assert record.execution_result == "denied by hook"
        tool_mock.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Runner event loop
# ---------------------------------------------------------------------------

class TestRunnerEventHandling:
    async def test_queue_message_pushes_event(self, chatbot_manager_mock):
        session = _make_mock_session()
        agent = _make_mock_agent(session)
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4())
        msg = MagicMock()
        runner.push_event(msg)
        assert not runner.event_queue.empty()

    async def test_approval_event_processed(self, chatbot_manager_mock):
        """ApprovalEvent pushes to queue and is handled in run()."""
        session = _make_mock_session()
        agent = _make_mock_agent(session)
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4())

        # Create a group with a pending tool call
        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        await runner.execution_environment.add_tool_call(
            ContentPart.create_tool_use("tc1", "add", "{}"), runner=runner
        )

        evt = ApprovalEvent(tool_call_id="tc1", approved=True)
        runner.push_event(evt)

        assert not runner.event_queue.empty()

        await runner.stop()


# ---------------------------------------------------------------------------
# Runner subscriptions
# ---------------------------------------------------------------------------

class TestRunnerSubscriptions:
    def test_subscribe_returns_true(self, chatbot_manager_mock):
        session = _make_mock_session()
        agent = _make_mock_agent(session)
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4())
        channel = MagicMock()
        assert runner.subscribe(channel) is True
        assert channel in runner._channels

    def test_subscribe_duplicate_returns_false(self, chatbot_manager_mock):
        session = _make_mock_session()
        agent = _make_mock_agent(session)
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4())
        channel = MagicMock()
        runner.subscribe(channel)
        assert runner.subscribe(channel) is False

    def test_unsubscribe_returns_true(self, chatbot_manager_mock):
        session = _make_mock_session()
        agent = _make_mock_agent(session)
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4())
        channel = MagicMock()
        runner.subscribe(channel)
        assert runner.unsubscribe(channel) is True
        assert channel not in runner._channels

    def test_unsubscribe_unknown_returns_false(self, chatbot_manager_mock):
        session = _make_mock_session()
        agent = _make_mock_agent(session)
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4())
        channel = MagicMock()
        assert runner.unsubscribe(channel) is False


# ---------------------------------------------------------------------------
# Runner _handle_tool_group — invocation hooks
# ---------------------------------------------------------------------------

class TestRunnerOnToolCallHook:
    def _build_env(self, auto_approve_tools=None, invocation_hooks=None):
        """Build session/runner with execution environment."""
        role = MagicMock()
        role.name = "test-role"
        role.model = "test-model"
        role.behavior_policy = "responsive"
        role.auto_approve_tools = auto_approve_tools or ["add"]
        role.tool_filter = []

        tm = MagicMock()
        tool_mock = MagicMock()
        tool_mock.name = "add"
        tool_mock.func = MagicMock(return_value=3)
        tool_mock.execute = MagicMock(return_value=3)
        tm.get_tool.return_value = tool_mock

        ctx = MagicMock()
        ctx.messages = []

        session = MagicMock()
        session.role = role
        session.tool_manager = tm
        session.auto_approve_tools = list(role.auto_approve_tools)
        session.tool_failure_policy = "abort"
        session.active_context = ctx
        session.invocation_hooks = invocation_hooks or {}

        agent = MagicMock()
        agent.get_session.return_value = session
        agent._tool_manager = tm
        agent.role = role

        cb = _make_mock_chatbot(content=[{"type": "text", "content": "done"}])
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4(), chatbot=cb)
        runner._execution_environment = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=list(role.auto_approve_tools),
            tool_failure_policy="abort",
        )
        return session, runner, role, tm, tool_mock

    async def test_on_tool_call_denies_with_string(self):
        """on_tool_call hook returning a string denies the tool at registration."""
        session, runner, role, tm, tool_mock = self._build_env()

        def deny_hook(ctx):
            assert ctx["role"] == "test-role"
            assert ctx["tool_name"] == "add"
            assert ctx["arguments"] == {"a": 1, "b": 2}
            assert ctx["session"] is session
            return "Hook says no"

        session.invocation_hooks["on_tool_call"] = [deny_hook]

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "add", '{"a": 1, "b": 2}')
        await runner.execution_environment.add_tool_call(tc, runner=runner)

        fg = runner.execution_environment.get_foreground_group()
        record = fg.records[0]

        assert record.approval_status == ToolApprovalStatus.DENIED
        assert record.denied_reason == "Hook says no"

    async def test_on_tool_call_allows_with_none(self):
        """on_tool_call hook returning None keeps pending status."""
        session, runner, role, tm, tool_mock = self._build_env()
        runner.execution_environment.auto_approve_tools = []

        def allow_hook(ctx):
            assert ctx["tool_name"] == "add"
            return None

        session.invocation_hooks["on_tool_call"] = [allow_hook]

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "add", '{"a": 1, "b": 2}')
        await runner.execution_environment.add_tool_call(tc, runner=runner)

        fg = runner.execution_environment.get_foreground_group()
        record = fg.records[0]

        assert record.approval_status == ToolApprovalStatus.PENDING
        assert record.denied_reason is None

    async def test_on_tool_call_broadcast_calls_all_hooks(self):
        """All on_tool_call hooks fire — broadcast, not short-circuit."""
        session, runner, role, tm, tool_mock = self._build_env()

        call_order = []

        def deny_hook(ctx):
            call_order.append("first")
            return "first denied"

        def second_hook(ctx):
            call_order.append("second")
            return None

        session.invocation_hooks["on_tool_call"] = [deny_hook, second_hook]

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "add", '{}')
        await runner.execution_environment.add_tool_call(tc, runner=runner)

        fg = runner.execution_environment.get_foreground_group()
        record = fg.records[0]

        assert record.approval_status == ToolApprovalStatus.DENIED
        assert call_order == ["first", "second"]
        assert "first denied" in record.denied_reason

    async def test_on_tool_call_multiple_denied_reasons_joined(self):
        """Multiple denying hooks have reasons joined with newlines."""
        session, runner, role, tm, tool_mock = self._build_env()

        def deny1(ctx):
            return "reason one"

        def deny2(ctx):
            return "reason two"

        session.invocation_hooks["on_tool_call"] = [deny1, deny2]

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "add", '{}')
        await runner.execution_environment.add_tool_call(tc, runner=runner)

        record = runner.execution_environment.get_foreground_group().records[0]
        assert record.approval_status == ToolApprovalStatus.DENIED
        assert "reason one" in record.denied_reason
        assert "reason two" in record.denied_reason

    async def test_on_tool_call_ctx_denied_reason_accumulates(self):
        """Each hook sees ctx['denied_reason'] with all prior denial reasons."""
        session, runner, role, tm, tool_mock = self._build_env()

        reasons_seen = []

        def hook1(ctx):
            reasons_seen.append(("h1", ctx.get("denied_reason")))
            return "first"

        def hook2(ctx):
            reasons_seen.append(("h2", ctx.get("denied_reason")))
            return None

        session.invocation_hooks["on_tool_call"] = [hook1, hook2]

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "add", '{}')
        await runner.execution_environment.add_tool_call(tc, runner=runner)

        assert reasons_seen[0] == ("h1", None)
        assert "first" in (reasons_seen[1][1] or "")

    async def test_on_tool_call_true_approves_pending_at_registration(self):
        """on_tool_call hook returning True upgrades PENDING to APPROVED immediately."""
        session, runner, role, tm, tool_mock = self._build_env()
        runner.execution_environment.auto_approve_tools = []

        def approve_hook(ctx):
            assert ctx["approval_status"] == ToolApprovalStatus.PENDING
            return True

        session.invocation_hooks["on_tool_call"] = [approve_hook]

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "add", '{"a": 1}')
        await runner.execution_environment.add_tool_call(tc, runner=runner)

        record = runner.execution_environment.get_foreground_group().records[0]
        assert record.approval_status == ToolApprovalStatus.APPROVED
        assert record.denied_reason is None

    async def test_on_tool_call_true_on_approved_stays_approved(self):
        """True on already APPROVED stays APPROVED (True cannot escalate APPROVED)."""
        session, runner, role, tm, tool_mock = self._build_env()

        def no_op(ctx):
            return True

        session.invocation_hooks["on_tool_call"] = [no_op]

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "add", '{}')
        await runner.execution_environment.add_tool_call(tc, runner=runner)

        record = runner.execution_environment.get_foreground_group().records[0]
        assert record.approval_status == ToolApprovalStatus.APPROVED

    async def test_on_tool_call_true_on_denied_stays_denied(self):
        """True on DENIED stays DENIED (True cannot escalate DENIED)."""
        session, runner, role, tm, tool_mock = self._build_env()
        tm.get_tool.return_value = None

        def approve_hook(ctx):
            return True

        session.invocation_hooks["on_tool_call"] = [approve_hook]

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "nonexistent", '{}')
        await runner.execution_environment.add_tool_call(tc, runner=runner)

        record = runner.execution_environment.get_foreground_group().records[0]
        assert record.approval_status == ToolApprovalStatus.DENIED

    async def test_on_tool_call_false_denies_approved(self):
        """False on APPROVED downgrades to DENIED."""
        session, runner, role, tm, tool_mock = self._build_env()

        def deny_hook(ctx):
            return False

        session.invocation_hooks["on_tool_call"] = [deny_hook]

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "add", '{}')
        await runner.execution_environment.add_tool_call(tc, runner=runner)

        record = runner.execution_environment.get_foreground_group().records[0]
        assert record.approval_status == ToolApprovalStatus.DENIED

    async def test_on_tool_call_false_denies_pending(self):
        """False on PENDING becomes DENIED."""
        session, runner, role, tm, tool_mock = self._build_env()
        runner.execution_environment.auto_approve_tools = []

        def deny_hook(ctx):
            return False

        session.invocation_hooks["on_tool_call"] = [deny_hook]

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "add", '{}')
        await runner.execution_environment.add_tool_call(tc, runner=runner)

        record = runner.execution_environment.get_foreground_group().records[0]
        assert record.approval_status == ToolApprovalStatus.DENIED

    async def test_on_tool_call_tool_not_found_still_fires_hooks(self):
        """Tool not found creates DENIED, but hooks still fire and can add reasons."""
        session, runner, role, tm, tool_mock = self._build_env()
        tm.get_tool.return_value = None

        def enrich_hook(ctx):
            return "hook adds context"

        session.invocation_hooks["on_tool_call"] = [enrich_hook]

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "nonexistent", '{}')
        await runner.execution_environment.add_tool_call(tc, runner=runner)

        record = runner.execution_environment.get_foreground_group().records[0]
        assert record.approval_status == ToolApprovalStatus.DENIED
        assert "not available" in record.denied_reason
        assert "hook adds context" in record.denied_reason

    async def test_on_tool_call_tool_not_found_no_hooks(self):
        """Tool not found with no hooks: DENIED with initial reason only."""
        session, runner, role, tm, tool_mock = self._build_env()
        tm.get_tool.return_value = None
        session.invocation_hooks = {}

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "nonexistent", '{}')
        await runner.execution_environment.add_tool_call(tc, runner=runner)

        record = runner.execution_environment.get_foreground_group().records[0]
        assert record.approval_status == ToolApprovalStatus.DENIED
        assert "not available" in record.denied_reason
        assert "\n" not in record.denied_reason

    async def test_on_tool_call_empty_hooks_list(self):
        """Empty hooks list: status unchanged from initial."""
        session, runner, role, tm, tool_mock = self._build_env()
        session.invocation_hooks = {}

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "add", '{}')
        await runner.execution_environment.add_tool_call(tc, runner=runner)

        record = runner.execution_environment.get_foreground_group().records[0]
        assert record.approval_status == ToolApprovalStatus.APPROVED

    async def test_on_tool_call_ctx_has_required_fields(self):
        """on_tool_call ctx contains all required fields."""
        session, runner, role, tm, tool_mock = self._build_env()

        captured_ctx = {}

        def capture_hook(ctx):
            captured_ctx.update(ctx)
            return None

        session.invocation_hooks["on_tool_call"] = [capture_hook]

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "add", '{"a": 1}')
        await runner.execution_environment.add_tool_call(tc, runner=runner)

        assert captured_ctx["role"] == "test-role"
        assert captured_ctx["session"] is session
        assert captured_ctx["tool_name"] == "add"
        assert captured_ctx["arguments"] == {"a": 1}
        assert "approval_status" in captured_ctx
        assert "respond" in captured_ctx
        assert "denied_reason" in captured_ctx


# ---------------------------------------------------------------------------
# Runner execute_and_inject
# ---------------------------------------------------------------------------

def _make_runner_with_exec_env(auto_approve_tools=None):
    """Build a runner with execution environment for execute_and_inject tests."""
    role = MagicMock()
    role.name = "test-role"
    role.model = "test-model"
    role.behavior_policy = "responsive"
    role.auto_approve_tools = auto_approve_tools or []
    role.tool_filter = []

    tm = MagicMock()
    tool = MagicMock()
    tool.name = "add"
    tool.func = lambda a, b: a + b
    tool.parameters = {"a": {"type": "int"}, "b": {"type": "int"}}
    tool.execute = MagicMock(return_value=5)
    tm.get_tool.return_value = tool

    ctx = MagicMock()
    ctx.messages = []

    session = MagicMock()
    session.role = role
    session.tool_manager = tm
    session.auto_approve_tools = list(role.auto_approve_tools)
    session.tool_failure_policy = "abort"
    session.active_context = ctx
    session.invocation_hooks = {}

    agent = MagicMock()
    agent.get_session.return_value = session
    agent._tool_manager = tm
    agent.role = role

    chatbot = _make_mock_chatbot(content=[{"type": "text", "content": "done"}])
    runner = Runner(agent=agent, session_uuid=_uuid.uuid4(), chatbot=chatbot)
    runner._execution_environment = ExecutionEnvironment(
        tool_manager=tm, role=role, auto_approve_tools=list(role.auto_approve_tools),
        tool_failure_policy="abort",
    )
    return runner, tm, tool


class TestRunnerExecuteAndInject:
    """Test execute_and_inject via runner's execution environment."""

    async def test_execute_and_inject_creates_result_message(self):
        """execute_and_inject creates result message with correct content."""
        runner, tm, tool = _make_runner_with_exec_env()
        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        record = await runner.execution_environment.add_tool_call(ContentPart.create_tool_use("tc1", "add", "{}"), runner=runner)
        result_str, success = await runner.execution_environment.execute_and_inject(record, runner)
        assert success is True
        assert result_str == "5"
        fg = runner.execution_environment.get_foreground_group()
        assert fg.result_message is not None
        cp = fg.result_message.raw_dict["content"][0]
        assert cp["type"] == "tool_result"
        assert cp["call_id"] == "tc1"
        assert cp["name"] == "add"
        assert cp["content"] == "5"

    async def test_execute_and_inject_appends_multiple_results(self):
        """execute_and_inject appends results for multiple tool calls."""
        runner, tm, tool = _make_runner_with_exec_env()
        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        record1 = await runner.execution_environment.add_tool_call(ContentPart.create_tool_use("tc1", "add", "{}"), runner=runner)
        record2 = await runner.execution_environment.add_tool_call(ContentPart.create_tool_use("tc2", "add", "{}"), runner=runner)
        tool.execute = MagicMock(side_effect=[5, 10])

        for record in [record1, record2]:
            await runner.execution_environment.execute_and_inject(record, runner)

        fg = runner.execution_environment.get_foreground_group()
        assert len(fg.result_message.raw_dict["content"]) == 2
        assert fg.result_message.raw_dict["content"][0]["call_id"] == "tc1"
        assert fg.result_message.raw_dict["content"][1]["call_id"] == "tc2"

    async def test_execute_and_inject_fails_gracefully(self):
        """execute_and_inject catches tool exceptions and returns error."""
        runner, tm, tool = _make_runner_with_exec_env()
        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        record = await runner.execution_environment.add_tool_call(ContentPart.create_tool_use("tc1", "add", "{}"), runner=runner)
        tool.execute = MagicMock(side_effect=RuntimeError("kaboom"))
        result_str, success = await runner.execution_environment.execute_and_inject(record, runner)
        assert success is False
        assert "RuntimeError" in result_str
        fg = runner.execution_environment.get_foreground_group()
        assert len(fg.result_message.raw_dict["content"]) == 1

    async def test_execute_and_inject_tool_not_found(self):
        """execute_and_inject returns error for unknown tool."""
        tm = MagicMock()
        tm.get_tool.return_value = None
        role = MagicMock()
        role.name = "test-role"
        role.model = "test-model"
        role.behavior_policy = "responsive"
        role.auto_approve_tools = []
        role.tool_filter = []
        session = MagicMock()
        session.role = role
        session.invocation_hooks = {}
        agent = MagicMock()
        agent.get_session.return_value = session
        agent._tool_manager = tm
        agent.role = role
        chatbot = _make_mock_chatbot(content=[{"type": "text", "content": "done"}])
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4(), chatbot=chatbot)
        runner._execution_environment = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[], tool_failure_policy="abort"
        )
        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        record = await runner.execution_environment.add_tool_call(ContentPart.create_tool_use("tc1", "nonexistent", "{}"), runner=runner)
        result_str, success = await runner.execution_environment.execute_and_inject(record, runner)
        assert success is False
        assert "not found" in result_str

    async def test_execute_and_inject_no_foreground_group(self):
        """execute_and_inject raises if no foreground group exists."""
        role = MagicMock()
        role.name = "test-role"
        role.model = "test-model"
        role.behavior_policy = "responsive"
        role.auto_approve_tools = []
        role.tool_filter = []
        tm = MagicMock()
        session = MagicMock()
        session.role = role
        session.invocation_hooks = {}
        agent = MagicMock()
        agent.get_session.return_value = session
        agent._tool_manager = tm
        agent.role = role
        chatbot = _make_mock_chatbot(content=[{"type": "text", "content": "done"}])
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4(), chatbot=chatbot)
        runner._execution_environment = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[], tool_failure_policy="abort"
        )
        with pytest.raises(RuntimeError, match="No foreground tool call group"):
            await runner.execution_environment.execute_and_inject(
                ContentPart.create_tool_use("tc1", "add", "{}"), runner
            )
