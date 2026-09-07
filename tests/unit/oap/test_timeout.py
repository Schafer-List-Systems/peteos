"""Integration tests for AgenticObject invoke_agent timeout behavior.

Tests that invoke_agent properly raises TimeoutError when the agent does not
produce output within the timeout period.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Any, List

import pytest

from peteos.chatbot import (
    ChatBotManager,
    SimpleMockBackendProvider,
    SimpleMockChatBot,
    SimpleMockChatBotResponse,
)
from peteos.chatbot.backendprovider import BackendProvider
from peteos.chatbot.chatbot import ChatBot
from peteos.chatbot.chatbotconfig import ChatBotConfig
from peteos.conversation.message import ContentPart, Message
from peteos.oap import AgenticObject, agentic_object, tool


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
# Infinite mock chatbot
# ---------------------------------------------------------------------------


class InfiniteMockChatBot(SimpleMockChatBot):
    """A chatbot that always returns the same message without exhausting."""

    def __init__(self, message: Message) -> None:
        super().__init__([message])
        self._message = message

    async def send_context(
        self,
        _context: object,
        _generation_config: Dict[str, Any] | None = None,
        _streaming: bool | None = None,
    ) -> object:
        return SimpleMockChatBotResponse(self._message)


class InfiniteMockBackendProvider(BackendProvider):
    """BackendProvider that creates InfiniteMockChatBot instances."""

    def __init__(self, response: Message) -> None:
        self._response = response

    async def list_models(self, url: str, api_key: str | None = None) -> List[str]:
        return ["infinite-mock"]

    def create_chatbot(
        self,
        _http_client: object | None,
        _config: ChatBotConfig,
    ) -> ChatBot:
        return InfiniteMockChatBot(self._response)


# ---------------------------------------------------------------------------
# Test object
# ---------------------------------------------------------------------------


@agentic_object(allow_code_execution=True)
class TimeoutTestObj(AgenticObject):
    """Test agent with a tool that sleeps."""

    @tool
    async def test_sleep(self) -> str:
        """Sleep for 1 second using asyncio to allow the event loop to continue."""
        await asyncio.sleep(1)
        return "done"


# ---------------------------------------------------------------------------
# Timeout tests
# ---------------------------------------------------------------------------


class TestInvokeAgentTimeout:
    """Tests for invoke_agent timeout behavior."""

    @pytest.mark.asyncio
    async def test_timeout_raises_timeouterror_on_long_tool(self):
        """invoke_agent raises TimeoutError when a tool takes longer than the timeout.

        The async tool sleeps for 1s. invoke_agent has timeout=0.5s.
        Because the tool uses asyncio.sleep, the event loop continues and
        invoke_agent's timeout check fires after 0.5s, raising TimeoutError.
        """
        msg = Message.create(
            "assistant",
            [ContentPart.create_tool_use(
                call_id="call-1",
                name="test_sleep",
                arguments="{}",
            )],
        )
        ChatBotManager.register_provider("infinite-mock", InfiniteMockBackendProvider(msg))
        await ChatBotManager.add_backend("infinite-mock", url="http://localhost:9999", api_type="infinite-mock")

        obj = TimeoutTestObj()

        with pytest.raises(TimeoutError, match="Agent did not produce output within 0.5s timeout"):
            await obj.invoke_agent("Call test_sleep", timeout=0.5)
