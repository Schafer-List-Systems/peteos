"""Integration tests for Runner using SimpleMockChatBot.

Tests the full run() loop, step() chains, tool execution flows,
message queue, hooks, and state transitions — all using SimpleMockChatBot
so no real LLM is needed.
"""

from __future__ import annotations

import uuid as _uuid

from unittest.mock import MagicMock

import pytest

from peteos.chatbot import ChatBotManager
from peteos.chatbot import (
    SimpleMockChatBot,
)
from peteos.conversation.message import ContentPart, Message
from peteos.conversation.session import Session
from peteos.engine.exec_status import ExecStatus
from peteos.engine.executionenvironment import (
    ApprovalEvent,
    ExecutionEnvironment,
    ToolApprovalStatus,
    ToolCallGroup,
    ToolExecutionStatus,
)
from peteos.engine.runner import Runner
from peteos.persona.role import Role
from peteos.persona.toolmanager import ToolManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_chatbot_manager():
    """Reset ChatBotManager before each test to avoid provider name collisions."""
    ChatBotManager.reset()
    to_remove = [k for k in ChatBotManager._providers if k.startswith("test-mock")]
    for api_type in to_remove:
        ChatBotManager.unregister_provider(api_type)
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(role: str, parts: list[ContentPart]) -> Message:
    return Message.create(role, parts)


def _make_role(
    auto_approve_tools: list[str] | None = None,
    behavior_policy: str = "responsive",
    tool_names: list[str] | None = None,
) -> Role:
    """Create a minimal Role with a ToolManager.

    Args:
        auto_approve_tools: Tools auto-approved by the runner.
        behavior_policy: Behavior policy ("responsive" or "continuous").
        tool_names: If provided, registers mock tools with these names in the ToolManager.
    """
    tm = ToolManager()
    for tn in tool_names or []:
        tool = MagicMock()
        tool.name = tn
        tool.func = MagicMock(return_value="result")
        tool.execute = MagicMock(return_value="result")
        tm.register_tool(tool)
    role = Role(name="test-role", system_prompt="", model="test-model")
    role.auto_approve_tools = auto_approve_tools or []
    role.behavior_policy = behavior_policy
    role.tool_filter = []
    role._tool_manager = tm
    return role


def _make_runner_with_chatbot(
    role: Role,
    bot: SimpleMockChatBot,
) -> Runner:
    """Build a Runner from a role and bot."""
    agent = MagicMock()
    agent._tool_manager = role._tool_manager
    agent.role = role
    sid = _uuid.uuid4()
    return Runner(agent=agent, session_uuid=sid, chatbot=bot)


# ---------------------------------------------------------------------------
# Runner init
# ---------------------------------------------------------------------------


class TestRunnerInit:
    """Runner constructor and property tests."""

    def test_init_with_explicit_chatbot(self):
        """Runner accepts a chatbot directly, skipping ChatBotManager lookup."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()

        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)
        assert runner._chatbot is bot

    @pytest.mark.asyncio
    async def test_init_raises_when_no_chatbot_match(self):
        """Runner raises if ChatBotManager has no matching chatbot."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        ChatBotManager.reset()  # empty manager
        sid = _uuid.uuid4()

        with pytest.raises(ValueError, match="No ChatBot found"):
            Runner(agent=agent, session_uuid=sid)


# ---------------------------------------------------------------------------
# step() — full chains
# ---------------------------------------------------------------------------


class TestRunnerStepChains:
    """Tests for step() multi-turn chains with SimpleMockChatBot."""

    @pytest.mark.asyncio
    async def test_text_only_finished(self):
        """Text response → FINISHED."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("done")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        status, _ = await runner.step()
        assert status is ExecStatus.FINISHED

    @pytest.mark.asyncio
    async def test_reasoning_then_text(self):
        """Reasoning-only → CONTINUE, then text → FINISHED."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg1 = _make_message("assistant", [ContentPart.create_thinking("thinking")])
        msg2 = _make_message("assistant", [ContentPart.create_text("done")])
        bot = SimpleMockChatBot([msg1, msg2])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        # First call: reasoning only → CONTINUE
        status, resp = await runner.step()
        assert status is ExecStatus.CONTINUE
        assert resp is not None
        assert resp.content[0].type == "thinking"

        # Second call: text → FINISHED
        status, resp = await runner.step()
        assert status is ExecStatus.FINISHED
        assert resp is not None
        assert resp.content[0].type == "text"

    @pytest.mark.asyncio
    async def test_tool_use_auto_approved_then_text(self):
        """Auto-approved tool → CONTINUE, then text → FINISHED."""
        role = _make_role(auto_approve_tools=["search"], tool_names=["search"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg1 = _make_message("assistant", [
            ContentPart.create_tool_use("tc1", "search", '{"q":"hello"}'),
        ])
        msg2 = _make_message("assistant", [ContentPart.create_text("done")])
        bot = SimpleMockChatBot([msg1, msg2])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        # Step 1: tool_use → CONTINUE, foreground group created
        status, resp = await runner.step()
        assert status is ExecStatus.CONTINUE
        fg = runner.execution_environment.get_foreground_group()
        assert fg is not None

        # Step 2: _handle_tool_group executes the auto-approved tool and closes
        # the group, so the next step() will call the chatbot again
        await runner._handle_tool_group()
        status, resp = await runner.step()
        assert status is ExecStatus.FINISHED

    @pytest.mark.asyncio
    async def test_tool_use_manual_not_auto_approved(self):
        """Tool call that isn't auto-approved → group stays pending."""
        tm = ToolManager()
        tool = MagicMock()
        tool.name = "search"
        tool.func = MagicMock(return_value="result")
        tool.execute = MagicMock(return_value="result")
        tm.register_tool(tool)

        role = _make_role(auto_approve_tools=[], tool_names=["search"])
        role._tool_manager = tm

        agent = MagicMock()
        agent._tool_manager = tm
        agent.role = role

        msg1 = _make_message("assistant", [
            ContentPart.create_tool_use("tc1", "search", '{"q":"hello"}'),
        ])
        bot = SimpleMockChatBot([msg1])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        status, resp = await runner.step()
        assert status is ExecStatus.CONTINUE
        fg = runner.execution_environment.get_foreground_group()
        assert fg is not None
        # Tool is registered but not auto-approved → PENDING
        assert fg.records[0].approval_status == ToolApprovalStatus.PENDING

    @pytest.mark.asyncio
    async def test_error_response(self):
        """Chatbot exhausts → RuntimeError is raised."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        bot = SimpleMockChatBot([])  # empty → RuntimeError
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        with pytest.raises(RuntimeError, match="HTTP 503"):
            await runner.step()

    @pytest.mark.asyncio
    async def test_tool_use_and_text_in_same_response(self):
        """Response with both text and tool_use → CONTINUE (tool_use takes priority)."""
        role = _make_role(auto_approve_tools=["search"], tool_names=["search"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [
            ContentPart.create_text("Let me search"),
            ContentPart.create_tool_use("tc1", "search", '{"q":"hello"}'),
        ])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        status, _ = await runner.step()
        assert status is ExecStatus.CONTINUE
        fg = runner.execution_environment.get_foreground_group()
        assert fg is not None

    @pytest.mark.asyncio
    async def test_continuous_policy_keeps_running_with_text(self):
        """Continuous behavior_policy → CONTINUE even with text output."""
        role = _make_role(behavior_policy="continuous")
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("keep going")])
        msg2 = _make_message("assistant", [ContentPart.create_text("still going")])
        bot = SimpleMockChatBot([msg, msg2])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        status, _ = await runner.step()
        assert status is ExecStatus.CONTINUE

        status, _ = await runner.step()
        assert status is ExecStatus.CONTINUE  # still continuous

        # After messages exhausted, RuntimeError
        with pytest.raises(RuntimeError, match="HTTP 503"):
            await runner.step()

    @pytest.mark.asyncio
    async def test_chatbot_error_skips_response(self):
        """Chatbot exhausts → RuntimeError is raised."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        bot = SimpleMockChatBot([])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        with pytest.raises(RuntimeError, match="HTTP 503"):
            await runner.step()


# ---------------------------------------------------------------------------
# run() loop — full integration
# ---------------------------------------------------------------------------


class TestRunnerRunLoop:
    """Tests for the full run() event loop."""

    @pytest.mark.asyncio
    async def test_run_text_only_finishes(self):
        """Single text response → FINISHED → run exits."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("done")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        test_msg = _make_message("user", [ContentPart.create_text("hello")])
        await runner.queue_message(test_msg)

        idle = await runner.wait_for_idle(timeout=2.0)
        assert idle is True
        await runner.stop()

    @pytest.mark.asyncio
    async def test_run_tool_use_auto_approved(self):
        """Auto-approved tool executes, then text → FINISHED."""
        role = _make_role(auto_approve_tools=["search"], tool_names=["search"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg1 = _make_message("assistant", [
            ContentPart.create_tool_use("tc1", "search", '{"q":"hello"}'),
        ])
        msg2 = _make_message("assistant", [ContentPart.create_text("done")])
        bot = SimpleMockChatBot([msg1, msg2])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        test_msg = _make_message("user", [ContentPart.create_text("hello")])
        await runner.queue_message(test_msg)

        idle = await runner.wait_for_idle(timeout=2.0)
        assert idle is True
        await runner.stop()

    @pytest.mark.asyncio
    async def test_run_error_exits_loop(self):
        """ERROR status → run loop breaks (runner breaks without setting idle)."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        bot = SimpleMockChatBot([])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        test_msg = _make_message("user", [ContentPart.create_text("hello")])
        await runner.queue_message(test_msg)

        # Run loop breaks on ERROR without setting idle — stop it directly
        await runner.stop()


# ---------------------------------------------------------------------------
# Message queue operations
# ---------------------------------------------------------------------------


class TestRunnerMessageQueue:
    """Tests for queue_message and wait_for_idle."""

    @pytest.mark.asyncio
    async def test_queue_message_pushes_to_queue(self):
        """queue_message pushes a Message to the event queue."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        user_msg = _make_message("user", [ContentPart.create_text("hello")])
        await runner.queue_message(user_msg)

        assert not runner.event_queue.empty()
        await runner.stop()

    @pytest.mark.asyncio
    async def test_queue_message_starts_loop(self):
        """queue_message starts the active loop if not already running."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        assert not runner.is_running()
        user_msg = _make_message("user", [ContentPart.create_text("hello")])
        await runner.queue_message(user_msg)
        await runner.stop()

    @pytest.mark.asyncio
    async def test_wait_for_idle_becomes_idle(self):
        """Runner becomes idle after processing a message."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        user_msg = _make_message("user", [ContentPart.create_text("hello")])
        await runner.queue_message(user_msg)

        idle = await runner.wait_for_idle(timeout=2.0)
        assert idle is True
        await runner.stop()


# ---------------------------------------------------------------------------
# _handle_tool_group — tool execution
# ---------------------------------------------------------------------------


class TestHandleToolGroup:
    """Tests for _handle_tool_group processing."""

    @pytest.mark.asyncio
    async def test_approved_tool_executes(self):
        """Auto-approved tool executes and runs to completion."""
        role = _make_role(auto_approve_tools=["add"], tool_names=["add"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("done")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        tc = ContentPart.create_tool_use("tc1", "add", '{"a":1,"b":2}')
        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        runner.execution_environment.add_tool_call(tc)

        fg = runner.execution_environment.get_foreground_group()
        assert fg is not None
        assert fg.records[0].approval_status == ToolApprovalStatus.APPROVED

        result_appended = await runner._handle_tool_group()
        assert result_appended is True

    @pytest.mark.asyncio
    async def test_result_message_appended(self):
        """create_result_message creates placeholder entries in the group's result message."""
        role = _make_role(auto_approve_tools=["search"], tool_names=["search"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("done")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        tc = ContentPart.create_tool_use("tc1", "search", '{"q":"x"}')
        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        runner.execution_environment.add_tool_call(tc)

        fg = runner.execution_environment.get_foreground_group()
        result_msg = fg.result_message
        assert len(result_msg.raw_dict["content"]) == 1
        assert result_msg.raw_dict["content"][0]["type"] == "tool_result"
        assert result_msg.raw_dict["content"][0]["call_id"] == "tc1"
        assert result_msg.raw_dict["content"][0]["content"] == ""

    @pytest.mark.asyncio
    async def test_result_message_none_fire_and_forget(self):
        """Group with all fire-and-forget results → _handle_tool_group returns False."""
        role = _make_role(auto_approve_tools=["add"], tool_names=["add"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("done")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        tc = ContentPart.create_tool_use("tc1", "add", '{"a":1,"b":2}')
        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        runner.execution_environment.add_tool_call(tc)

        # Make the tool return None (fire-and-forget)
        role._tool_manager.get_tool("add").execute = MagicMock(return_value=None)

        result_appended = await runner._handle_tool_group()
        assert result_appended is False  # all fire-and-forget, so idle

    @pytest.mark.asyncio
    async def test_no_foreground_group(self):
        """_handle_tool_group returns False when no foreground group."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        result = await runner._handle_tool_group()
        assert result is False

    @pytest.mark.asyncio
    async def test_denied_tool_in_group_breaks_loop(self):
        """A denied tool call breaks the tool group processing."""
        role = _make_role(tool_names=["nonexistent"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        tc = ContentPart.create_tool_use("tc1", "nonexistent", "{}")
        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        runner.execution_environment.add_tool_call(tc)

        fg = runner.execution_environment.get_foreground_group()
        fg.records[0].approval_status = ToolApprovalStatus.DENIED

        result_appended = await runner._handle_tool_group()
        fg2 = runner.execution_environment.get_foreground_group()
        # A single denied tool is a final state — group is done and closed
        assert fg2 is None
        assert result_appended is False


# ---------------------------------------------------------------------------
# After-step hook modifying status
# ---------------------------------------------------------------------------


class TestAfterStepHook:
    """Tests for after_step hook on status."""

    @pytest.mark.asyncio
    async def test_after_step_returns_error_stops_run(self):
        """after_step hook returning ExecStatus.CRITICAL → run loop breaks."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        runner.execution_environment.register_hook(
            "after_step",
            lambda status: ExecStatus.CRITICAL,
        )

        user_msg = _make_message("user", [ContentPart.create_text("hello")])
        await runner.queue_message(user_msg)

        # after_step hook returns ERROR → run loop breaks without setting idle
        idle = await runner.wait_for_idle(timeout=2.0)
        assert idle is False
        await runner.stop()


# ---------------------------------------------------------------------------
# Channel notification publishing
# ---------------------------------------------------------------------------


class TestRunnerNotifications:
    """Tests for publish_notification and channel subscriptions."""

    @pytest.mark.asyncio
    async def test_publish_notification_to_channels(self):
        """Notification is pushed to all subscribed channels."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        ch = MagicMock()
        ch.push_event = MagicMock()
        runner.subscribe(ch)

        test_msg = _make_message("user", [ContentPart.create_text("hello")])
        await runner.publish_notification(test_msg)

        ch.push_event.assert_called_once()
        await runner.stop()

    @pytest.mark.asyncio
    async def test_append_and_notify(self):
        """append_and_notify appends message and publishes notification."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        # Create a real Session so that append_and_notify actually stores messages
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            session = Session.create(parent_dir=tmpdir, system_prompt_message=None)
            agent.get_session.return_value = session

            msg = _make_message("assistant", [ContentPart.create_text("hi")])
            bot = SimpleMockChatBot([msg])
            sid = session.uuid
            runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

            ch = MagicMock()
            ch.push_event = MagicMock()
            runner.subscribe(ch)

            test_msg = _make_message("user", [ContentPart.create_text("hello")])
            await runner.append_and_notify(test_msg)

            assert len(runner.session.active_context.messages) > 0
            ch.push_event.assert_called_once()
            await runner.stop()


# ---------------------------------------------------------------------------
# Foreground group state machine
# ---------------------------------------------------------------------------


class TestForegroundGroupStates:
    """Tests for foreground group state transitions."""

    @pytest.mark.asyncio
    async def test_group_create_pending(self):
        """Group starts with no records."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        fg = runner.execution_environment.get_foreground_group()
        assert fg is not None
        assert fg.id == "g1"
        assert fg.anchor_name == "g1:tool_result"
        assert len(fg.records) == 0

    @pytest.mark.asyncio
    async def test_group_add_pending_record(self):
        """Adding a non-auto-approved tool creates a PENDING record."""
        role = _make_role(auto_approve_tools=[], tool_names=["search"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "search", '{"q":"x"}')
        record = runner.execution_environment.add_tool_call(tc)

        assert record.approval_status == ToolApprovalStatus.PENDING
        assert record.execution_status == ToolExecutionStatus.WAITING_FOR_APPROVAL
        assert record.tool_call_id == "tc1"

    @pytest.mark.asyncio
    async def test_group_add_auto_approved_record(self):
        """Adding an auto-approved tool creates an APPROVED record."""
        role = _make_role(auto_approve_tools=["search"], tool_names=["search"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "search", '{"q":"x"}')
        record = runner.execution_environment.add_tool_call(tc)

        assert record.approval_status == ToolApprovalStatus.APPROVED
        assert record.execution_status == ToolExecutionStatus.EXECUTING

    @pytest.mark.asyncio
    async def test_group_add_unknown_tool_denied(self):
        """Adding a tool not in tool manager creates a DENIED record."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "nonexistent", "{}")
        record = runner.execution_environment.add_tool_call(tc)

        assert record.approval_status == ToolApprovalStatus.DENIED
        assert record.denied_reason is not None

    @pytest.mark.asyncio
    async def test_group_deny_all_remaining(self):
        """deny_all_remaining denies unexecuted tools in group."""
        role = _make_role(auto_approve_tools=[], tool_names=["search"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc1 = ContentPart.create_tool_use("tc1", "search", '{"q":"x"}')
        tc2 = ContentPart.create_tool_use("tc2", "search", '{"q":"y"}')
        runner.execution_environment.add_tool_call(tc1)
        runner.execution_environment.add_tool_call(tc2)

        fg = runner.execution_environment.get_foreground_group()
        fg.deny_all_remaining("user cancelled")

        assert fg.records[0].approval_status == ToolApprovalStatus.DENIED
        assert fg.records[1].approval_status == ToolApprovalStatus.DENIED
        assert fg.records[0].denied_reason == "Tool group denied: user cancelled"
        assert fg.records[1].denied_reason == "Tool group denied: user cancelled"

    @pytest.mark.asyncio
    async def test_group_is_done_all_executed(self):
        """Group is_done when all records are executed or denied."""
        role = _make_role(auto_approve_tools=["search"], tool_names=["search"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "search", '{"q":"x"}')
        runner.execution_environment.add_tool_call(tc)

        fg = runner.execution_environment.get_foreground_group()
        fg.records[0].execution_status = ToolExecutionStatus.EXECUTED
        assert fg.is_done() is True

    @pytest.mark.asyncio
    async def test_group_is_done_not_done(self):
        """Group not done when a record is still pending."""
        role = _make_role(auto_approve_tools=[], tool_names=["search"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "search", '{"q":"x"}')
        runner.execution_environment.add_tool_call(tc)

        fg = runner.execution_environment.get_foreground_group()
        assert fg.is_done() is False

    @pytest.mark.asyncio
    async def test_close_foreground_group(self):
        """close_foreground_group removes the group."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        assert runner.execution_environment.get_foreground_group() is not None

        runner.execution_environment.close_foreground_group()
        assert runner.execution_environment.get_foreground_group() is None


# ---------------------------------------------------------------------------
# Tool call record lifecycle
# ---------------------------------------------------------------------------


class TestToolCallRecordLifecycle:
    """Tests for ToolCallRecord through approval and execution."""

    @pytest.mark.asyncio
    async def test_record_pending_to_approved(self):
        """Record transitions from PENDING to APPROVED via ApprovalEvent."""
        role = _make_role(auto_approve_tools=[], tool_names=["search"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "search", '{"q":"x"}')
        runner.execution_environment.add_tool_call(tc)

        fg = runner.execution_environment.get_foreground_group()
        assert fg.records[0].approval_status == ToolApprovalStatus.PENDING

        evt = ApprovalEvent(tool_call_id="tc1", approved=True)
        approved, _ = runner.execution_environment._handle_approval(evt)
        assert approved is True
        assert fg.records[0].approval_status == ToolApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_record_denied_cascades(self):
        """Denying first tool denies all remaining in group."""
        role = _make_role(auto_approve_tools=[], tool_names=["search"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc1 = ContentPart.create_tool_use("tc1", "search", '{"q":"x"}')
        tc2 = ContentPart.create_tool_use("tc2", "search", '{"q":"y"}')
        runner.execution_environment.add_tool_call(tc1)
        runner.execution_environment.add_tool_call(tc2)

        fg = runner.execution_environment.get_foreground_group()

        evt = ApprovalEvent(tool_call_id="tc1", approved=False)
        approved, _ = runner.execution_environment._handle_approval(evt)
        assert approved is False

        assert fg.records[0].approval_status == ToolApprovalStatus.DENIED
        assert fg.records[1].approval_status == ToolApprovalStatus.DENIED

    @pytest.mark.asyncio
    async def test_find_pending_record(self):
        """find_pending_record returns the matching record."""
        role = _make_role(auto_approve_tools=[], tool_names=["search"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "search", '{"q":"x"}')
        runner.execution_environment.add_tool_call(tc)

        record = runner.execution_environment.find_pending_record("tc1")
        assert record is not None
        assert record.tool_call_id == "tc1"

    @pytest.mark.asyncio
    async def test_find_pending_record_not_found(self):
        """find_pending_record returns None for unknown ID."""
        role = _make_role(auto_approve_tools=[], tool_names=["search"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "search", '{"q":"x"}')
        runner.execution_environment.add_tool_call(tc)

        record = runner.execution_environment.find_pending_record("nonexistent")
        assert record is None

    @pytest.mark.asyncio
    async def test_get_pending_tool_calls(self):
        """get_pending_tool_calls returns all pending records."""
        role = _make_role(auto_approve_tools=["search"], tool_names=["search", "other"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc1 = ContentPart.create_tool_use("tc1", "search", '{"q":"x"}')  # auto-approved
        tc2 = ContentPart.create_tool_use("tc2", "other", '{"q":"y"}')  # not in auto-approve
        runner.execution_environment.add_tool_call(tc1)
        runner.execution_environment.add_tool_call(tc2)

        pending = runner.execution_environment.get_pending_tool_calls()
        assert len(pending) == 1
        assert pending[0].tool_call_id == "tc2"

    @pytest.mark.asyncio
    async def test_tool_call_missing_call_id_raises(self):
        """add_tool_call raises if tool_call has no call_id."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart({"type": "tool_use", "name": "search", "arguments": "{}"})

        with pytest.raises(ValueError, match="call_id"):
            runner.execution_environment.add_tool_call(tc)


# ---------------------------------------------------------------------------
# ExecutionEnvironment — hook system
# ---------------------------------------------------------------------------


class TestEEHooks:
    """Tests for ExecutionEnvironment hook registration and invocation."""

    def test_register_hook_valid_point(self):
        """register_hook succeeds for a valid hook point."""
        tm = MagicMock()
        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )
        ee.register_hook("before_send_to_chatbot", lambda *a: None)

    def test_register_hook_invalid_point_raises(self):
        """register_hook raises for an unknown hook point."""
        tm = MagicMock()
        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )
        with pytest.raises(ValueError, match="Unknown hook point"):
            ee.register_hook("nonexistent", lambda *a: None)

    def test_deregister_hook(self):
        """deregister_hook removes the callback."""
        tm = MagicMock()
        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )

        def callback(*a):
            pass

        ee.register_hook("before_send_to_chatbot", callback)
        ee.deregister_hook("before_send_to_chatbot", callback)

    def test_deregister_hook_not_found_raises(self):
        """deregister_hook raises if callback not registered."""
        tm = MagicMock()
        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )

        def callback(*a):
            pass

        with pytest.raises(ValueError, match="Callback not found"):
            ee.deregister_hook("before_send_to_chatbot", callback)

    def test_deregister_all_hooks(self):
        """deregister_all_hooks clears all hooks for a point."""
        tm = MagicMock()
        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )

        ee.register_hook("before_send_to_chatbot", lambda *a: None)
        ee.register_hook("before_send_to_chatbot", lambda *a: None)
        ee.deregister_all_hooks("before_send_to_chatbot")

    @pytest.mark.asyncio
    async def test_call_hooks_invokes_all(self):
        """call_hooks invokes all registered callbacks in order."""
        tm = MagicMock()
        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )

        calls = []
        ee.register_hook("before_send_to_chatbot", lambda *a: calls.append(1))
        ee.register_hook("before_send_to_chatbot", lambda *a: calls.append(2))

        await ee.call_hooks("before_send_to_chatbot")
        assert calls == [1, 2]

    @pytest.mark.asyncio
    async def test_call_hooks_with_async_callback(self):
        """call_hooks awaits async callbacks."""
        tm = MagicMock()
        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )

        result = []

        async def async_callback(*a):
            result.append("async")

        ee.register_hook("before_send_to_chatbot", async_callback)
        await ee.call_hooks("before_send_to_chatbot")
        assert result == ["async"]

    @pytest.mark.asyncio
    async def test_call_hooks_merge_exec_status(self):
        """call_hooks merges ExecStatus returns from hooks."""
        tm = MagicMock()
        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )

        ee.register_hook("after_step", lambda *a: ExecStatus.CRITICAL)

        merged = await ee.call_hooks("after_step")
        assert merged is ExecStatus.CRITICAL

    @pytest.mark.asyncio
    async def test_call_hooks_deny(self):
        """call_hooks_deny returns first deny result."""
        tm = MagicMock()
        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )

        ee.register_hook("before_tool_execution", lambda *a: (True, "allow"))
        ee.register_hook("before_tool_execution", lambda *a: (False, "deny"))
        ee.register_hook("before_tool_execution", lambda *a: (True, "allow"))

        result = await ee.call_hooks_deny("before_tool_execution")
        assert result == (False, "deny")

    @pytest.mark.asyncio
    async def test_call_hooks_deny_no_deny(self):
        """call_hooks_deny returns None when no hooks deny."""
        tm = MagicMock()
        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )

        ee.register_hook("before_tool_execution", lambda *a: (True, "allow"))
        ee.register_hook("before_tool_execution", lambda *a: (True, "also allow"))

        result = await ee.call_hooks_deny("before_tool_execution")
        assert result is None


# ---------------------------------------------------------------------------
# ExecutionEnvironment — execute_tool
# ---------------------------------------------------------------------------


class TestEEExecuteTool:
    """Tests for ExecutionEnvironment.execute_tool."""

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """execute_tool returns error when tool is not found."""
        tm = MagicMock()
        tm.get_tool.return_value = None
        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )

        tc = ContentPart.create_tool_use("tc1", "nonexistent", "{}")
        result, success = await ee.execute_tool(tc)
        assert success is False
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """execute_tool succeeds and returns result string."""
        tm = MagicMock()
        tool = MagicMock()
        tool.name = "add"
        tool.execute = MagicMock(return_value=42)
        tm.get_tool.return_value = tool

        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )

        tc = ContentPart.create_tool_use("tc1", "add", '{"a":1,"b":2}')
        result, success = await ee.execute_tool(tc)
        assert success is True
        assert result == "42"

    @pytest.mark.asyncio
    async def test_execute_tool_exception(self):
        """execute_tool catches tool exceptions and returns error string."""
        tm = MagicMock()
        tool = MagicMock()
        tool.name = "add"
        tool.execute = MagicMock(side_effect=ValueError("calculation error"))
        tm.get_tool.return_value = tool

        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )

        tc = ContentPart.create_tool_use("tc1", "add", '{"a":1,"b":2}')
        result, success = await ee.execute_tool(tc)
        assert success is False
        assert "ValueError" in result

    @pytest.mark.asyncio
    async def test_execute_tool_returns_none_fire_and_forget(self):
        """execute_tool returns (None, True) when tool returns None."""
        tm = MagicMock()
        tool = MagicMock()
        tool.name = "notify"
        tool.execute = MagicMock(return_value=None)
        tm.get_tool.return_value = tool

        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )

        tc = ContentPart.create_tool_use("tc1", "notify", "{}")
        result, success = await ee.execute_tool(tc)
        assert success is True
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_tool_before_deny_hook(self):
        """before_tool_execution hook can deny the tool."""
        tm = MagicMock()
        tool = MagicMock()
        tool.name = "add"
        tool.execute = MagicMock(return_value=42)
        tm.get_tool.return_value = tool

        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )

        ee.register_hook("before_tool_execution", lambda *a: (False, "not allowed"))

        tc = ContentPart.create_tool_use("tc1", "add", '{"a":1,"b":2}')
        result, success = await ee.execute_tool(tc)
        assert success is False
        assert result == "not allowed"
        tool.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_and_inject(self):
        """execute_and_inject creates result message with tool_result content."""
        tm = MagicMock()
        tool = MagicMock()
        tool.name = "add"
        tool.execute = MagicMock(return_value=42)
        tm.get_tool.return_value = tool

        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )

        ee.create_tool_group("g1", "g1:tool_result")
        ee.add_tool_call(ContentPart.create_tool_use("tc1", "add", "{}"))

        tc = ContentPart.create_tool_use("tc1", "add", '{"a":1,"b":2}')
        result_str, success = await ee.execute_and_inject(tc)

        assert success is True
        assert result_str == "42"
        fg = ee.get_foreground_group()
        assert len(fg.result_message.content) == 1
        assert fg.result_message.content[0].type == "tool_result"

    @pytest.mark.asyncio
    async def test_execute_and_inject_none_result(self):
        """execute_and_inject with None tool result does not create message."""
        tm = MagicMock()
        tool = MagicMock()
        tool.name = "notify"
        tool.execute = MagicMock(return_value=None)
        tm.get_tool.return_value = tool

        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )

        ee.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "notify", "{}")
        result_str, success = await ee.execute_and_inject(tc)

        assert success is True
        assert result_str is None
        fg = ee.get_foreground_group()
        assert fg.result_message is not None

    @pytest.mark.asyncio
    async def test_execute_and_inject_no_foreground_raises(self):
        """execute_and_inject raises RuntimeError with no foreground group."""
        tm = MagicMock()
        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )

        tc = ContentPart.create_tool_use("tc1", "add", "{}")
        with pytest.raises(RuntimeError, match="No foreground"):
            await ee.execute_and_inject(tc)

    @pytest.mark.asyncio
    async def test_execute_tool_async_result(self):
        """execute_tool awaits async tool results."""
        tm = MagicMock()

        async def async_exec(**kwargs):
            return "async_result"

        tool = MagicMock()
        tool.name = "async_tool"
        tool.execute = async_exec
        tm.get_tool.return_value = tool

        role = _make_role()
        ee = ExecutionEnvironment(
            tool_manager=tm, role=role, auto_approve_tools=[],
            tool_failure_policy="abort",
        )

        tc = ContentPart.create_tool_use("tc1", "async_tool", "{}")
        result, success = await ee.execute_tool(tc)
        assert success is True
        assert result == "async_result"


# ---------------------------------------------------------------------------
# Full run() cycle: message → chatbot → tool → chatbot → FINISHED
# ---------------------------------------------------------------------------


class TestRunnerFullCycle:
    """End-to-end tests for the complete run loop cycle."""

    @pytest.mark.asyncio
    async def test_full_cycle_single_tool(self):
        """Message → text → tool_use → text → FINISHED."""
        role = _make_role(auto_approve_tools=["add"], tool_names=["add"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg1 = _make_message("assistant", [
            ContentPart.create_tool_use("tc1", "add", '{"a":1,"b":2}'),
        ])
        msg2 = _make_message("assistant", [ContentPart.create_text("done")])
        bot = SimpleMockChatBot([msg1, msg2])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        user_msg = _make_message("user", [ContentPart.create_text("hello")])
        await runner.queue_message(user_msg)

        idle = await runner.wait_for_idle(timeout=2.0)
        assert idle is True
        await runner.stop()

    @pytest.mark.asyncio
    async def test_full_cycle_tool_denied(self):
        """Tool denied by hook → loop exits gracefully."""
        role = _make_role(auto_approve_tools=["add"], tool_names=["add"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg1 = _make_message("assistant", [
            ContentPart.create_tool_use("tc1", "add", '{"a":1,"b":2}'),
        ])
        msg2 = _make_message("assistant", [ContentPart.create_text("done")])
        bot = SimpleMockChatBot([msg1, msg2])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        def deny_add(ctx):
            return "denied by hook"

        runner._session._invocation_hooks = {"on_tool_call": [deny_add]}

        user_msg = _make_message("user", [ContentPart.create_text("hello")])
        await runner.queue_message(user_msg)

        idle = await runner.wait_for_idle(timeout=2.0)
        assert idle is True
        await runner.stop()

    @pytest.mark.asyncio
    async def test_step_with_foreground_group_skips_chatbot(self):
        """step() with existing foreground group returns PENDING, not calling chatbot."""
        role = _make_role(tool_names=["search"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg = _make_message("assistant", [ContentPart.create_text("hi")])
        bot = SimpleMockChatBot([msg])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        runner.execution_environment.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "search", '{"q":"x"}')
        runner.execution_environment.add_tool_call(tc)

        status, resp = await runner.step()
        assert status is ExecStatus.PENDING
        assert resp is None

        # Chatbot was not called — step() returned PENDING due to existing foreground group
        # Verify no foreground group remains after step() (it should still be active for _handle_tool_group)
        fg = runner.execution_environment.get_foreground_group()
        assert fg is not None

    @pytest.mark.asyncio
    async def test_step_after_tool_group_executed(self):
        """After foreground group finishes, step() calls chatbot again."""
        role = _make_role(auto_approve_tools=["add"], tool_names=["add"])
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        msg1 = _make_message("assistant", [
            ContentPart.create_tool_use("tc1", "add", '{"a":1,"b":2}'),
        ])
        msg2 = _make_message("assistant", [ContentPart.create_text("done")])
        bot = SimpleMockChatBot([msg1, msg2])
        sid = _uuid.uuid4()
        runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

        status, _ = await runner.step()
        assert status is ExecStatus.CONTINUE
        fg = runner.execution_environment.get_foreground_group()
        assert fg is not None
        assert fg.records[0].approval_status == ToolApprovalStatus.APPROVED
        assert fg.records[0].execution_status == ToolExecutionStatus.EXECUTING

        # _handle_tool_group executes the auto-approved tool and closes the group
        await runner._handle_tool_group()

        status, _ = await runner.step()
        assert status is ExecStatus.FINISHED

    @pytest.mark.asyncio
    async def test_queue_message_append_and_notify(self):
        """queue_message appends the user message and publishes notification."""
        role = _make_role()
        agent = MagicMock()
        agent._tool_manager = role._tool_manager
        agent.role = role

        # Create a real Session so that append_and_notify actually stores messages
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            session = Session.create(parent_dir=tmpdir, system_prompt_message=None)
            agent.get_session.return_value = session

            msg = _make_message("assistant", [ContentPart.create_text("hi")])
            bot = SimpleMockChatBot([msg])
            sid = session.uuid
            runner = Runner(agent=agent, session_uuid=sid, chatbot=bot)

            ch = MagicMock()
            ch.push_event = MagicMock()
            runner.subscribe(ch)

            user_msg = _make_message("user", [ContentPart.create_text("hello")])
            await runner.queue_message(user_msg)

            # Wait for the runner loop to process the event (append_and_notify + chatbot)
            idle = await runner.wait_for_idle(timeout=1.0)
            assert idle is True

            ctx = runner.session.active_context
            found = False
            for m in ctx.messages:
                if m.content and m.content[0].type == "text" and m.content[0].text == "hello":
                    found = True
                    break
            assert found is True

            ch.push_event.assert_called()
            assert idle is True
            await runner.stop()
