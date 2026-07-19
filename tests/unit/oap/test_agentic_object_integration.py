"""Integration tests for AgenticObject invoke_agent with mock chatbots.

Tests the invoke_agent path end-to-end using SimpleMockChatBot to verify
the agentic object handles various chatbot responses correctly — including
error cases and empty responses that may cause infinite loops.
"""

from __future__ import annotations

import pytest

from peteos.chatbot import (
    ChatBotManager,
    SimpleMockBackendProvider,
)
from peteos.conversation.message import ContentPart, Message
from peteos.conversation.session import Session
from peteos.oap import AgenticObject, agentic_object
from peteos.persona.agent import Agent
from peteos.persona.role import Role


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_chatbot_manager():
    """Reset ChatBotManager before each test to avoid provider name collisions."""
    ChatBotManager.reset()
    for api_type in list(ChatBotManager._providers.keys()):
        ChatBotManager.unregister_provider(api_type)
    yield


# ---------------------------------------------------------------------------
# Test object
# ---------------------------------------------------------------------------


@agentic_object(allow_code_execution=True)
class SimpleTestObj(AgenticObject):
    """Minimal AgenticObject with just the produce_output tool."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_with_agent(obj: AgenticObject) -> tuple[Session, Agent, Role]:
    """Create a real Agent+Session+Role wired to the object's ToolManager."""
    tm = obj._oap_tool_manager
    role = Role(name="test-ao-role", system_prompt="", model="test-ao")
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


# ---------------------------------------------------------------------------
# invoke_agent with empty chatbot — infinite loop detection
# ---------------------------------------------------------------------------


class TestInvokeAgentEmptyChatBot:
    """Tests for invoke_agent with empty/non-functional mock chatbot backends."""

    @pytest.mark.asyncio
    async def test_invoke_agent_with_empty_mock_returns_error(self):
        """invoke_agent with an empty SimpleMockChatBot should NOT hang.

        An empty chatbot (no messages) returns an error response from send_context.
        The runner step() should return ExecStatus.ERROR, and invoke_agent should
        return an Error object, not loop forever.
        """
        # Empty chatbot — no messages
        ChatBotManager.register_provider("simple-mock", SimpleMockBackendProvider([]))
        await ChatBotManager.add_backend("simple-mock", url="http://localhost:9999", api_type="simple-mock")

        obj = SimpleTestObj()
        with pytest.raises(RuntimeError, match="HTTP 503"):
            await obj.invoke_agent("hello")

    @pytest.mark.asyncio
    async def test_invoke_agent_with_text_only_response(self):
        """invoke_agent with a text-only mock chatbot response should return Error.

        A chatbot that only returns text (no produce_output) should eventually
        be prompted by the after_step hook reminder, but must still not hang
        if it keeps returning text-only responses.
        """
        msg = Message.create("assistant", [ContentPart.create_text("I can't do that")])
        ChatBotManager.register_provider("simple-mock", SimpleMockBackendProvider([msg]))
        await ChatBotManager.add_backend("simple-mock", url="http://localhost:9999", api_type="simple-mock")

        obj = SimpleTestObj()
        # First call returns text-only, after_step hook queues reminder.
        # Second call exhausts mock → HTTP 503 → RuntimeError raised.
        with pytest.raises(RuntimeError, match="HTTP 503"):
            await obj.invoke_agent("hello")
