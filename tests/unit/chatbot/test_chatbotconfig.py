"""Unit tests for ChatBotConfig."""

from peteos.chatbot.chatbotconfig import ChatBotConfig


class TestChatBotConfig:
    """Tests for ChatBotConfig dataclass."""

    def test_from_dict_minimal(self):
        """Minimal config dict applies defaults."""
        data = {"name": "test", "url": "http://test:8000"}
        config = ChatBotConfig.from_dict(data)
        assert config.name == "test"
        assert config.url == "http://test:8000"
        assert config.api_type is None
        assert config.chat_endpoint is None
        assert config.models_endpoint is None
        assert config.streaming is False
        assert isinstance(config.max_tokens, int)
        assert config.max_tokens > 0
        assert config.model is None
        assert config.response_translations is None
        assert config.request_translations is None

    def test_from_dict_all_fields(self):
        """Config dict with all fields overrides defaults."""
        data = {
            "name": "full",
            "url": "http://full:9000",
            "api_type": "openai",
            "chat_endpoint": "/custom/chat",
            "models_endpoint": "/custom/models",
            "streaming": False,
            "max_tokens": 8192,
            "model": "gpt-4",
            "response_translations": {"key": "value"},
            "request_translations": {"req": "key"},
        }
        config = ChatBotConfig.from_dict(data)
        assert config.name == "full"
        assert config.url == "http://full:9000"
        assert config.api_type == "openai"
        assert config.chat_endpoint == "/custom/chat"
        assert config.models_endpoint == "/custom/models"
        assert config.streaming is False
        assert config.max_tokens == 8192
        assert config.model == "gpt-4"
        assert config.response_translations == {"key": "value"}
        assert config.request_translations == {"req": "key"}

    def test_from_dict_rejects_non_field_keys(self):
        """Keys not in the dataclass are ignored."""
        data = {"name": "test", "url": "http://test:8000", "unknown_key": "ignore"}
        config = ChatBotConfig.from_dict(data)
        assert config.name == "test"
        assert config.url == "http://test:8000"

    def test_from_dict_partial_override(self):
        """Partial dict applies only specified overrides."""
        data = {"name": "test", "url": "http://test:8000", "streaming": False}
        config = ChatBotConfig.from_dict(data)
        assert config.streaming is False
        assert config.model is None  # still default

    def test_model_default_is_none(self):
        """model defaults to None when not provided."""
        config = ChatBotConfig.from_dict({"name": "t", "url": "http://t:80"})
        assert config.model is None

    def test_streaming_default_is_false(self):
        """streaming defaults to False."""
        config = ChatBotConfig.from_dict({"name": "t", "url": "http://t:80"})
        assert config.streaming is False

    def test_max_tokens_default(self):
        """max_tokens defaults to a positive integer."""
        config = ChatBotConfig.from_dict({"name": "t", "url": "http://t:80"})
        assert isinstance(config.max_tokens, int)
        assert config.max_tokens > 0

    def test_response_translations_default(self):
        """response_translations defaults to None."""
        config = ChatBotConfig.from_dict({"name": "t", "url": "http://t:80"})
        assert config.response_translations is None

    def test_request_translations_default(self):
        """request_translations defaults to None."""
        config = ChatBotConfig.from_dict({"name": "t", "url": "http://t:80"})
        assert config.request_translations is None

    def test_priority_default_is_zero(self):
        """priority defaults to 0 when not provided."""
        config = ChatBotConfig.from_dict({"name": "t", "url": "http://t:80"})
        assert config.priority == 0

    def test_priority_explicit_value(self):
        """priority can be explicitly set."""
        config = ChatBotConfig.from_dict(
            {"name": "t", "url": "http://t:80", "model": "gpt-4", "priority": 42}
        )
        assert config.priority == 42
