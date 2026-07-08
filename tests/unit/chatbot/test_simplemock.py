"""Unit tests for SimpleMockChatBot, SimpleMockChatBotResponse, and SimpleMockBackendProvider."""

import pytest

from peteos.chatbot import (
    ChatBotManager,
    SimpleMockChatBot,
    SimpleMockChatBotResponse,
    SimpleMockBackendProvider,
)
from peteos.conversation.message import Message
from peteos.conversation import ContentPart


def _make_message(role: str, parts: list[ContentPart]) -> Message:
    return Message.create(role, parts)


class TestSimpleMockChatBotResponse:
    """Tests for SimpleMockChatBotResponse."""

    def test_build_message_returns_stored_message(self):
        """_build_message returns the Message passed to constructor."""
        parts = [ContentPart.create_text("Hello")]
        msg = _make_message("assistant", parts)
        response = SimpleMockChatBotResponse(msg)
        assert response._build_message() is msg

    def test_data_contains_role(self):
        """Response data contains the role from the Message."""
        parts = [ContentPart.create_text("Hello")]
        msg = _make_message("assistant", parts)
        response = SimpleMockChatBotResponse(msg)
        assert response["role"] == "assistant"

    def test_data_contains_content(self):
        """Response data contains the content parts from the Message."""
        parts = [ContentPart.create_text("Hello")]
        msg = _make_message("assistant", parts)
        response = SimpleMockChatBotResponse(msg)
        assert response["content"] == [{"type": "text", "content": "Hello"}]

    def test_has_text_part_true_for_text_message(self):
        """has_text_part is True when message has text content."""
        parts = [ContentPart.create_text("Hello")]
        msg = _make_message("assistant", parts)
        response = SimpleMockChatBotResponse(msg)
        assert response.has_text_part is True

    def test_has_text_part_false_for_tool_use(self):
        """has_text_part is False when message has only tool_use content."""
        parts = [ContentPart.create_tool_use("call_1", "search", '{"q":"x"}')]
        msg = _make_message("assistant", parts)
        response = SimpleMockChatBotResponse(msg)
        assert response.has_text_part is False

    def test_message_cache_is_stored_message(self):
        """message property returns the stored Message via cache."""
        parts = [ContentPart.create_text("Hello")]
        msg = _make_message("assistant", parts)
        response = SimpleMockChatBotResponse(msg)
        assert response.message is msg


class TestSimpleMockChatBot:
    """Tests for SimpleMockChatBot."""

    def test_list_available_models(self):
        """list_available_models returns ["simple-mock"]."""
        bot = SimpleMockChatBot([])
        assert bot.list_available_models() == ["simple-mock"]

    def test_model_returns_simple_mock(self):
        """model property returns 'simple-mock'."""
        bot = SimpleMockChatBot([])
        assert bot.model == "simple-mock"

    @pytest.mark.asyncio
    async def test_returns_messages_in_order(self):
        """Messages are returned in the order they were added."""
        parts_a = [ContentPart.create_text("First")]
        parts_b = [ContentPart.create_text("Second")]
        msg_a = _make_message("assistant", parts_a)
        msg_b = _make_message("assistant", parts_b)

        bot = SimpleMockChatBot([msg_a, msg_b])

        resp1 = await bot.send_context(None)
        assert resp1.message.content[0].text == "First"

        resp2 = await bot.send_context(None)
        assert resp2.message.content[0].text == "Second"

    @pytest.mark.asyncio
    async def test_returns_error_when_exhausted(self):
        """After all messages are consumed, returns an error response."""
        bot = SimpleMockChatBot([])

        resp = await bot.send_context(None)
        assert "error" in resp
        assert resp["error"] == "unexpected request — no more mock responses"

    @pytest.mark.asyncio
    async def test_pops_messages_from_list(self):
        """Messages are popped, allowing error response after exhaustion."""
        bot = SimpleMockChatBot([_make_message("assistant", [ContentPart.create_text("X")])])

        # First call succeeds
        resp1 = await bot.send_context(None)
        assert resp1.message.content[0].text == "X"

        # Second call fails
        resp2 = await bot.send_context(None)
        assert "error" in resp2

    @pytest.mark.asyncio
    async def test_tool_use_content_part(self):
        """Messages with tool_use content parts work correctly."""
        parts = [ContentPart.create_tool_use("call_1", "search", '{"q":"hello"}')]
        msg = _make_message("assistant", parts)
        bot = SimpleMockChatBot([msg])

        resp = await bot.send_context(None)
        assert resp["content"][0]["type"] == "tool_use"
        assert resp["content"][0]["name"] == "search"
        assert resp["content"][0]["arguments"] == '{"q":"hello"}'


class TestSimpleMockBackendProvider:
    """Tests for SimpleMockBackendProvider."""

    @pytest.mark.asyncio
    async def test_list_models_returns_simple_mock(self):
        """list_models returns ["simple-mock"]."""
        provider = SimpleMockBackendProvider([])
        assert await provider.list_models("http://localhost") == ["simple-mock"]

    def test_create_chatbot_returns_simple_mock_with_responses(self):
        """create_chatbot returns a SimpleMockChatBot pre-loaded with responses."""
        parts = [ContentPart.create_text("Hello from mock")]
        msg = _make_message("assistant", parts)
        provider = SimpleMockBackendProvider([msg])

        bot = provider.create_chatbot(None, None)
        assert isinstance(bot, SimpleMockChatBot)
        assert bot._responses[0].content[0].text == "Hello from mock"

    @pytest.mark.asyncio
    async def test_provider_chatbot_produces_responses(self):
        """ChatBot created by provider produces the expected responses."""
        parts = [ContentPart.create_text("Hello from mock")]
        msg = _make_message("assistant", parts)
        provider = SimpleMockBackendProvider([msg])

        bot = provider.create_chatbot(None, None)
        resp = await bot.send_context(None)
        assert resp.message.content[0].text == "Hello from mock"


class TestChatBotManagerIntegration:
    """Tests for SimpleMockBackendProvider with ChatBotManager."""

    @pytest.fixture(autouse=True)
    def _reset_manager(self):
        ChatBotManager.reset()
        # Remove any previously registered mock providers
        to_remove = [k for k in ChatBotManager._providers if k.startswith("simple-mock")]
        for api_type in to_remove:
            ChatBotManager.unregister_provider(api_type)

    @pytest.mark.asyncio
    async def test_register_and_use_provider(self):
        """Provider is registered and used by ChatBotManager."""
        parts = [ContentPart.create_text("Hello")]
        msg = _make_message("assistant", parts)
        provider = SimpleMockBackendProvider([msg])

        ChatBotManager.register_provider("simple-mock-1", provider)

        await ChatBotManager.add_backend(
            "mock-backend-1",
            api_type="simple-mock-1",
        )

        # list_models returns "simple-mock", so regex must match that
        chatbots = ChatBotManager.list_chatbots("simple-mock")
        assert len(chatbots) == 1
        model_id, chatbot = chatbots[0]
        assert model_id == "simple-mock"

        resp = await chatbot.send_context(None)
        assert resp.message.content[0].text == "Hello"

    @pytest.mark.asyncio
    async def test_unregister_provider(self):
        """Unregister removes the provider."""
        ChatBotManager.register_provider("simple-mock-2", SimpleMockBackendProvider([]))
        assert ChatBotManager.unregister_provider("simple-mock-2") is True
        assert "simple-mock-2" not in ChatBotManager._providers

    @pytest.mark.asyncio
    async def test_unregistered_provider_raises_error(self):
        """Adding backend with unregistered api_type raises RuntimeError."""
        ChatBotManager.register_provider("simple-mock-3", SimpleMockBackendProvider([]))
        await ChatBotManager.add_backend(
            "mock-backend-3",
            api_type="simple-mock-3",
        )
        assert ChatBotManager.remove_backend("mock-backend-3")
        ChatBotManager.unregister_provider("simple-mock-3")

        with pytest.raises(RuntimeError, match="No provider registered"):
            await ChatBotManager.add_backend(
                "mock-backend-4",
                api_type="simple-mock-4",
            )


