"""Unit tests for BackendConfig."""

import pytest

from peteos.chatbot.backendconfig import BackendConfig


class TestBackendConfig:
    """Tests for BackendConfig dataclass."""

    def test_from_dict_required_fields_only(self):
        """Minimal config with only required fields."""
        data = {"name": "test", "url": "http://test:8000"}
        config = BackendConfig.from_dict(data)
        assert config.name == "test"
        assert config.url == "http://test:8000"
        assert config.api_type is None
        assert config.chat_endpoint is None
        assert config.models_endpoint is None
        assert config.streaming is True
        assert config.max_tokens == 4096
        assert config.retry_delays is None

    def test_from_dict_with_api_type(self):
        """Config with explicit api_type."""
        data = {"name": "test", "url": "http://test:8000", "api_type": "anthropic"}
        config = BackendConfig.from_dict(data)
        assert config.api_type == "anthropic"

    def test_from_dict_with_retry_delays(self):
        """Config with custom retry delays."""
        data = {"name": "test", "url": "http://test:8000", "retry_delays": [1, 2, 3, 5]}
        config = BackendConfig.from_dict(data)
        assert config.retry_delays == [1, 2, 3, 5]

    def test_from_dict_with_all_fields(self):
        """Config with all fields specified."""
        data = {
            "name": "full",
            "url": "http://full:9000",
            "api_type": "openai",
            "chat_endpoint": "/v/custom",
            "models_endpoint": "/v/custom/models",
            "streaming": False,
            "max_tokens": 16384,
            "retry_delays": [0.5, 1.0, 2.0],
        }
        config = BackendConfig.from_dict(data)
        assert config.api_type == "openai"
        assert config.chat_endpoint == "/v/custom"
        assert config.models_endpoint == "/v/custom/models"
        assert config.streaming is False
        assert config.max_tokens == 16384
        assert config.retry_delays == [0.5, 1.0, 2.0]

    def test_from_dict_rejects_non_field_keys(self):
        """Keys not in dataclass are ignored."""
        data = {"name": "test", "url": "http://t:80", "bogus": 123}
        config = BackendConfig.from_dict(data)
        assert hasattr(config, "name")
        assert not hasattr(config, "bogus")

    def test_from_dict_partial_override(self):
        """Partial override preserves other defaults."""
        data = {"name": "test", "url": "http://t:80", "max_tokens": 100}
        config = BackendConfig.from_dict(data)
        assert config.max_tokens == 100
        assert config.streaming is True  # still default

    def test_streaming_default_true(self):
        """streaming defaults to True."""
        config = BackendConfig.from_dict({"name": "t", "url": "http://t:80"})
        assert config.streaming is True

    def test_max_tokens_default(self):
        """max_tokens defaults to 4096."""
        config = BackendConfig.from_dict({"name": "t", "url": "http://t:80"})
        assert config.max_tokens == 4096

    def test_retry_delays_default_none(self):
        """retry_delays defaults to None."""
        config = BackendConfig.from_dict({"name": "t", "url": "http://t:80"})
        assert config.retry_delays is None
