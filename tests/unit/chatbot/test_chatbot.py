"""Unit tests for ChatBot abstract base class."""

import pytest
from unittest.mock import MagicMock

from peteos.chatbot.chatbot import ChatBot
from peteos.chatbot.chatbotconfig import ChatBotConfig
from peteos.chatbot.httpclient import HTTPClient
from peteos.chatbot.chatbotresponse import ChatBotResponse
from peteos.conversation.context import Context


class ConcreteChatBot(ChatBot):
    """Concrete ChatBot for testing the base class."""

    async def send_context(self, context, generation_config=None, streaming=None):
        return ChatBotResponse(MagicMock())

    def list_available_models(self):
        return ["test-model"]


class TestChatBotBaseClass:
    """Tests for ChatBot abstract base class behavior."""

    def test_cannot_instantiate_abstract_class(self):
        """ChatBot cannot be instantiated directly."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        with pytest.raises(TypeError):
            ChatBot(http_client, config)

    def test_subclass_must_implement_send_context(self):
        """A subclass that doesn't implement send_context raises TypeError."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")

        class PartialChatBot(ChatBot):
            def list_available_models(self):
                return []

        with pytest.raises(TypeError):
            PartialChatBot(http_client, config)

    def test_subclass_must_implement_list_available_models(self):
        """A subclass that doesn't implement list_available_models raises TypeError."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")

        class PartialChatBot(ChatBot):
            async def send_context(self, context, generation_config=None, streaming=None):
                return ChatBotResponse(MagicMock())

        with pytest.raises(TypeError):
            PartialChatBot(http_client, config)

    def test_http_client_stored(self):
        """ChatBot stores the provided HTTP client."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        bot = ConcreteChatBot(http_client, config)
        assert bot._http_client is http_client

    def test_config_stored(self):
        """ChatBot stores the provided config."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        bot = ConcreteChatBot(http_client, config)
        assert bot._config is config

    def test_model_property_returns_config_model(self):
        """model property returns the model from config."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="my-model")
        bot = ConcreteChatBot(http_client, config)
        assert bot.model == "my-model"

    def test_model_property_setter_updates_config(self):
        """model setter updates the config."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="model-a")
        bot = ConcreteChatBot(http_client, config)
        bot.model = "model-b"
        assert bot._config.model == "model-b"
        assert bot.model == "model-b"

    def test_list_available_models(self):
        """Concrete subclass can call list_available_models."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        bot = ConcreteChatBot(http_client, config)
        models = bot.list_available_models()
        assert models == ["test-model"]

    def test_send_context_signature_accepts_context(self):
        """send_context accepts a Context argument."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        bot = ConcreteChatBot(http_client, config)
        ctx = Context({"messages": []})
        # Just verify the method accepts the argument without error
        result = bot.send_context(ctx)
        assert result is not None
