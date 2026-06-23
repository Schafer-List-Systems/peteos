"""Unit tests for Runner and AgenticState."""

import asyncio
import uuid as _uuid
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from peteos.engine.executionenvironment import (
    ExecutionEnvironment,
    ToolApprovalStatus,
    ToolCallRecord,
    ToolExecutionStatus,
)
from peteos.engine.exec_status import ExecStatus
from peteos.conversation.message import ContentPart, Message
from peteos.engine.runner import AgenticState, ApprovalEvent, Runner


# ---------------------------------------------------------------------------
# AgenticState
# ---------------------------------------------------------------------------

class TestAgenticState:
    def test_initial_state_is_empty(self):
        state = AgenticState()
        assert state.list() == []
        assert state.get("nonexistent") is None

    def test_create_and_get(self):
        state = AgenticState()
        state.create("key1", "value1")
        assert state.get("key1") == "value1"
        assert state.list() == ["key1"]

    def test_create_raises_if_exists(self):
        state = AgenticState()
        state.create("key1", "value1")
        with pytest.raises(ValueError, match="already exists"):
            state.create("key1", "value2")

    def test_create_raises_on_none_value(self):
        state = AgenticState()
        with pytest.raises(ValueError, match="must not be None"):
            state.create("key1", None)

    def test_update_compare_and_swap(self):
        state = AgenticState()
        state.create("key1", "old")
        state.update("key1", "old", "new")
        assert state.get("key1") == "new"

    def test_update_raises_if_key_missing(self):
        state = AgenticState()
        with pytest.raises(KeyError, match="does not exist"):
            state.update("key1", "old", "new")

    def test_update_raises_on_none_old(self):
        state = AgenticState()
        state.create("key1", "val")
        with pytest.raises(ValueError, match="old_value must not be None"):
            state.update("key1", None, "new")

    def test_update_raises_on_none_new(self):
        state = AgenticState()
        state.create("key1", "val")
        with pytest.raises(ValueError, match="new_value must not be None"):
            state.update("key1", "val", None)

    def test_update_raises_on_mismatch(self):
        state = AgenticState()
        state.create("key1", "wrong")
        with pytest.raises(ValueError, match=r"has value 'wrong', expected 'correct'"):
            state.update("key1", "correct", "new")

    def test_delete(self):
        state = AgenticState()
        state.create("key1", "val")
        assert state.get("key1") == "val"
        state.delete("key1")
        assert state.get("key1") is None
        assert state.list() == []

    def test_delete_raises_if_missing(self):
        state = AgenticState()
        with pytest.raises(KeyError, match="does not exist"):
            state.delete("key1")

    def test_multiple_keys(self):
        state = AgenticState()
        state.create("a", "1")
        state.create("b", "2")
        state.create("c", "3")
        keys = state.list()
        assert set(keys) == {"a", "b", "c"}

    def test_delete_order_preserved(self):
        state = AgenticState()
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
        assert isinstance(runner.state, AgenticState)

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
        assert status is ExecStatus.ERROR


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
        runner.execution_environment.add_tool_call(
            ContentPart.create_tool_use("tc1", "add", "{}"), "g1"
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
        """Pre-setup with auto-approved tool → CONTINUE after execution."""
        session = _make_mock_session(auto_approve_tools=["add"])
        agent = _make_mock_agent(session)
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4())

        # Register a real tool
        tool_mock = MagicMock()
        tool_mock.name = "add"
        tool_mock.func = lambda a, b: a + b
        tool_mock.execute = MagicMock(return_value=3)
        session.tool_manager.get_tool.return_value = tool_mock

        # Pre-setup: auto-approved tool call
        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        runner.execution_environment.add_tool_call(
            ContentPart.create_tool_use("tc1", "add", "{}"), "g1"
        )

        status, _ = await runner.step()
        assert status is ExecStatus.CONTINUE

    async def test_tool_failed_at_execution(self, chatbot_manager_mock):
        """Pre-setup with auto-approved tool that fails → TOOL_FAILED."""
        session = _make_mock_session(auto_approve_tools=["add"])
        agent = _make_mock_agent(session)
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4())

        tool_mock = MagicMock()
        tool_mock.name = "add"
        tool_mock.func = lambda a, b: a + b
        tool_mock.execute = MagicMock(side_effect=ValueError("calculation error"))
        session.tool_manager.get_tool.return_value = tool_mock

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        runner.execution_environment.add_tool_call(
            ContentPart.create_tool_use("tc1", "add", "{}"), "g1"
        )

        status, _ = await runner.step()
        assert status is ExecStatus.TOOL_FAILED

    async def test_tool_denied_by_hook(self, chatbot_manager_mock):
        """Auto-approved tool denied by before_tool_execution hook → TOOL_FAILED."""
        session = _make_mock_session(auto_approve_tools=["add"])
        agent = _make_mock_agent(session)
        runner = Runner(agent=agent, session_uuid=_uuid.uuid4())

        tool_mock = MagicMock()
        tool_mock.name = "add"
        tool_mock.func = lambda a, b: a + b
        tool_mock.execute = MagicMock(return_value=3)
        session.tool_manager.get_tool.return_value = tool_mock

        def deny_hook(tc):
            return (False, "denied by hook")
        runner.execution_environment.register_hook("before_tool_execution", deny_hook)

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        runner.execution_environment.add_tool_call(
            ContentPart.create_tool_use("tc1", "add", "{}"), "g1"
        )

        status, _ = await runner.step()
        assert status is ExecStatus.TOOL_FAILED


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
        runner.execution_environment.add_tool_call(
            ContentPart.create_tool_use("tc1", "add", "{}"), "g1"
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
