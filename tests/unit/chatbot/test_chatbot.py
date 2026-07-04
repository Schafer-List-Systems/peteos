"""Unit tests for ChatBot abstract base class."""

import inspect

import pytest
from unittest.mock import MagicMock

from peteos.chatbot.chatbot import ChatBot
from peteos.chatbot.chatbotconfig import ChatBotConfig
from peteos.chatbot.chatbotresponse import ChatBotResponse


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
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        with pytest.raises(TypeError):
            ChatBot(config)

    def test_subclass_must_implement_send_context(self):
        """A subclass that doesn't implement send_context raises TypeError."""
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")

        class PartialChatBot(ChatBot):
            def list_available_models(self):
                return []

        with pytest.raises(TypeError):
            PartialChatBot(config)

    def test_subclass_must_implement_list_available_models(self):
        """A subclass that doesn't implement list_available_models raises TypeError."""
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")

        class PartialChatBot(ChatBot):
            async def send_context(self, context, generation_config=None, streaming=None):
                return ChatBotResponse(MagicMock())

        with pytest.raises(TypeError):
            PartialChatBot(config)

    def test_config_stored(self):
        """ChatBot stores the provided config."""
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        bot = ConcreteChatBot(config)
        assert bot._config is config

    def test_model_property_returns_config_model(self):
        """model property returns the model from config."""
        config = ChatBotConfig(name="test", url="http://test:8000", model="my-model")
        bot = ConcreteChatBot(config)
        assert bot.model == "my-model"

    def test_model_property_setter_updates_config(self):
        """model setter updates the config."""
        config = ChatBotConfig(name="test", url="http://test:8000", model="model-a")
        bot = ConcreteChatBot(config)
        bot.model = "model-b"
        assert bot._config.model == "model-b"
        assert bot.model == "model-b"

    def test_list_available_models(self):
        """Concrete subclass can call list_available_models."""
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        bot = ConcreteChatBot(config)
        models = bot.list_available_models()
        assert models == ["test-model"]

    def test_send_context_signature_accepts_context(self):
        """send_context accepts a Context argument."""
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        bot = ConcreteChatBot(config)
        # Verify the method exists and accepts a Context argument
        sig = inspect.signature(bot.send_context)
        params = list(sig.parameters.keys())
        assert "context" in params
