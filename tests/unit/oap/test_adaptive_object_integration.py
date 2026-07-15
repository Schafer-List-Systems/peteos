"""Tests for AdaptiveObject tool definitions in materialized messages.

Verifies that defined functions appear correctly in the ToolDefinitionsMessage
that the chatbot receives after session.materialize().
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from peteos.chatbot import (
    ChatBotManager,
    SimpleMockBackendProvider,
    SimpleMockChatBot,
)
from peteos.conversation.message import ContentPart, Message
from peteos.conversation.session import Session
from peteos.persona.agent import Agent
from peteos.persona.role import Role
from peteos.oap import AdaptiveObject, agentic_object, tool


def _mock_runner():
    runner = MagicMock()
    runner._execution_environment = MagicMock()
    runner._execution_environment.auto_approve_tools = []
    return runner


def _wire_runner(obj, runner):
    """Wire runner.sandbox_builder and runner.sandbox."""
    config = {"invoke_sub_agents": False}
    sandbox_builder = obj._create_sandbox_builder(config, runner=runner)
    runner.sandbox_builder = sandbox_builder
    runner.sandbox = sandbox_builder.get_sandbox(freeze_namespaces=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_chatbot_manager():
    """Reset ChatBotManager before each test to avoid provider name collisions."""
    ChatBotManager.reset()
    to_remove = [k for k in ChatBotManager._providers if k.startswith("test-adaptive")]
    for api_type in to_remove:
        ChatBotManager.unregister_provider(api_type)
    yield


def _make_message(role: str, parts: list[ContentPart]) -> Message:
    return Message.create(role, parts)


# ---------------------------------------------------------------------------
# Capturing mock chatbot
# ---------------------------------------------------------------------------


class AdaptiveMockChatBot(SimpleMockChatBot):
    """Mock chatbot that captures materialized tool definitions from each send_context call."""

    def __init__(self, responses: List[Message]):
        super().__init__(responses)
        self._captured_tool_defs: List[List[Dict[str, Any]]] = []

    async def send_context(self, context, *_args, **_kwargs):
        if hasattr(context, "active_context"):
            ctx = context.active_context
        else:
            ctx = context

        tdm = getattr(ctx, "tool_definitions_message", None)
        if tdm is not None:
            content = getattr(tdm, "content", None)
            if content:
                try:
                    # content is a list of ContentPart objects
                    self._captured_tool_defs.append([
                        cp.name for cp in content if cp.type == "tool"
                    ])
                except (TypeError, AttributeError):
                    pass

        return await super().send_context(context, *_args, **_kwargs)

    def captured_tools_at_step(self, step: int) -> List[str]:
        """Return the tool names captured at the given step (0-indexed)."""
        if step < len(self._captured_tool_defs):
            return self._captured_tool_defs[step]
        return []


# ---------------------------------------------------------------------------
# Test objects
# ---------------------------------------------------------------------------


@agentic_object(allow_code_execution=True)
class AdaptiveTestObj(AdaptiveObject):
    """AdaptiveObject with a static tool for collision testing."""

    @tool(name="add", description="Add two numbers")
    def add(self, a: int, b: int) -> int:
        return a + b


# ---------------------------------------------------------------------------
# Session → materialize → tool defs (the core integration path)
# ---------------------------------------------------------------------------


def _make_session_with_agent(obj: AdaptiveObject) -> tuple[Session, Agent, Role]:
    """Create a real Agent+Session+Role wired to the object's ToolManager."""
    tm = obj._oap_tool_manager
    role = Role(name="test-adaptive-role", system_prompt="", model="test-adaptive")
    role.tool_filter = []
    role.auto_approve_tools = []

    agent = Agent(role=role, tool_manager=tm)
    from peteos.conversation.system_prompt_message import SystemPromptMessage
    from peteos.conversation.tool_definitions_message import ToolDefinitionsMessage

    system_prompt_msg = SystemPromptMessage.create(role.system_prompt)
    tool_defs_msg = ToolDefinitionsMessage()
    session = Session.create(agent.agent_dir, system_prompt_msg, tool_defs_msg)
    session.session_dir.mkdir(exist_ok=True)
    session.register_hook(tool_defs_msg, ToolDefinitionsMessage.TOOL_LIST_HOOK_NAME, agent._tool_list_hook)
    session.register_hook(tool_defs_msg, ToolDefinitionsMessage.TOOL_FILTER_HOOK_NAME, agent._tool_filter_hook)

    return session, agent, role


class TestToolDefsAfterMaterialize:
    """Verify tool definitions are correct after session.materialize()."""

    def test_initial_tools_visible_after_materialize(self):
        """First materialize should include define_function, remove_function, and static add."""
        obj = AdaptiveTestObj()
        session, _, _ = _make_session_with_agent(obj)

        session.materialize()
        content = session.active_context.tool_definitions_message.content
        tool_names = [t.name for t in content if t.type == "tool"]

        assert "define_function" in tool_names
        assert "remove_function" in tool_names
        assert "add" in tool_names

    def test_defined_function_visible_after_materialize(self):
        """After define_function, the new tool should appear in tool defs."""
        obj = AdaptiveTestObj()
        obj._define_function(
            "def square(x: int) -> int:\n    return x * x",
            "Square a number",
            _mock_runner(),
        )
        session, _, _ = _make_session_with_agent(obj)

        session.materialize()
        content = session.active_context.tool_definitions_message.content
        tool_names = [t.name for t in content if t.type == "tool"]

        assert "square" in tool_names
        assert "define_function" in tool_names
        assert "add" in tool_names

    def test_removed_tool_disappears_after_materialize(self):
        """After remove_function, the removed tool should be gone from tool defs."""
        obj = AdaptiveTestObj()
        runner = _mock_runner()
        obj._define_function(
            "def triple(x: int) -> int:\n    return x * 3",
            "Triple a number",
            runner,
        )
        obj.remove_function("triple", runner)
        session, _, _ = _make_session_with_agent(obj)

        session.materialize()
        content = session.active_context.tool_definitions_message.content
        tool_names = [t.name for t in content if t.type == "tool"]

        assert "triple" not in tool_names
        assert "define_function" in tool_names
        assert "add" in tool_names

    def test_define_then_remove_and_verify(self):
        """Define, materialize (visible), remove, materialize (gone)."""
        obj = AdaptiveTestObj()
        runner = _mock_runner()
        obj._define_function(
            "def double(x: int) -> int:\n    return x * 2",
            "Double a number",
            runner,
        )

        session, _, _ = _make_session_with_agent(obj)
        session.materialize()
        content = session.active_context.tool_definitions_message.content
        tool_names_before = [t.name for t in content if t.type == "tool"]
        assert "double" in tool_names_before

        obj.remove_function("double", runner)
        session.materialize()
        content = session.active_context.tool_definitions_message.content
        tool_names_after = [t.name for t in content if t.type == "tool"]

        assert "double" not in tool_names_after
        assert "define_function" in tool_names_after


class TestMockChatBotCapturesToolDefs:
    """Verify the mock chatbot captures tool definitions from the context."""

    async def test_mock_captures_initial_tool_defs(self):
        """First send_context should capture the initial tool list."""
        obj = AdaptiveTestObj()
        bot = AdaptiveMockChatBot([
            _make_message("assistant", [ContentPart.create_text("Hello")]),
        ])
        session, _, _ = _make_session_with_agent(obj)

        session.materialize()
        session.is_active = True

        await bot.send_context(session, None, None)
        tools = bot.captured_tools_at_step(0)

        assert "define_function" in tools
        assert "remove_function" in tools
        assert "add" in tools

    async def test_mock_captures_updated_tool_defs_after_persist(self):
        """After define_function, second send_context should capture the new tool."""
        obj = AdaptiveTestObj()
        bot = AdaptiveMockChatBot([
            _make_message("assistant", [ContentPart.create_text("Step 1")]),
            _make_message("assistant", [ContentPart.create_text("Step 2")]),
        ])
        session, _, _ = _make_session_with_agent(obj)

        # Step 1: initial tools
        session.is_active = True
        session.materialize()
        await bot.send_context(session, None, None)
        tools_0 = bot.captured_tools_at_step(0)
        assert "define_function" in tools_0

        # Step 2: persist and re-materialize
        obj._define_function(
            "def triple(x: int) -> int:\n    return x * 3",
            "Triple a number",
            _mock_runner(),
        )
        session.materialize()
        await bot.send_context(session, None, None)
        tools_1 = bot.captured_tools_at_step(1)

        assert "triple" in tools_1
        assert "define_function" in tools_1
        assert "add" in tools_1


class TestSingleStepDefineAndExec:
    """Unit test: one mock message with define_function + python_exec in the same step."""

    @pytest.fixture(autouse=True)
    def _reset_manager(self):
        ChatBotManager.reset()
        to_remove = [k for k in ChatBotManager._providers if k.startswith("simple-mock")]
        for api_type in to_remove:
            ChatBotManager.unregister_provider(api_type)

    @pytest.mark.asyncio
    async def test_define_and_call_in_same_message(self):
        """invoke_agent processes define_function then python_exec from one mock response."""
        messages = [
            Message.create(
                "assistant",
                [
                    ContentPart.create_tool_use(
                        "call_1", "define_function",
                        json.dumps({
                            "code": "def double_it(x: int) -> int:\n    return x * 2",
                            "docstring": "Double a number"
                        }),
                    ),
                    ContentPart.create_tool_use(
                        "call_2", "python_exec",
                        json.dumps({
                            "function": "def func(self):\n    return self.produce_output(self.double_it(21))\n",
                        }),
                    ),
                ],
            ),
        ]
        ChatBotManager.register_provider("simple-mock", SimpleMockBackendProvider(messages))
        await ChatBotManager.add_backend("simple-mock", api_type="simple-mock")

        obj = AdaptiveTestObj()
        result = await obj.invoke_agent("do it")
        assert result == 42


class TestFullLifecycle:
    """End-to-end lifecycle: persist → tool visible → call → remove."""

    def test_persist_then_call_via_toolmanager(self):
        """Persist a function and call it through the ToolManager."""
        obj = AdaptiveTestObj()
        runner = _mock_runner()
        _wire_runner(obj, runner)
        result = obj._define_function(
            "def multiply(a: int, b: int) -> int:\n    return a * b",
            "Multiply two numbers",
            runner,
        )
        assert result == "OK: registered as 'multiply'"

        tool = obj._oap_tool_manager.get_tool("multiply")
        assert tool is not None
        result = tool.execute(a=6, b=7, runner=runner)
        assert result == 42

        obj.remove_function("multiply", runner)
        assert obj._oap_tool_manager.get_tool("multiply") is None
