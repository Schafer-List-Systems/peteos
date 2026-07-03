"""Unit tests for ChatBotManager."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from peteos.chatbot.manager import ChatBotManager, BackendInfo
from peteos.chatbot import OpenAIChatBot, AnthropicChatBot
from peteos.chatbot.chatbotconfig import ChatBotConfig


def _chatbot_config(**overrides):
    """Create a minimal ChatBotConfig for tests."""
    return ChatBotConfig(
        name="test",
        url="http://test:8000",
        **overrides,
    )


@pytest.fixture(autouse=True)
def _reset_chatbot_manager():
    """Reset ChatBotManager class-level state before and after each test."""
    ChatBotManager.reset()
    yield
    ChatBotManager.reset()


def _mock_providers(openai_models=None, anthropic_models=None, gemini_models=None):
    """Return a dict of mocked providers with configured list_models."""
    mock_providers = {}
    for api_type, models in [
        ("openai", openai_models),
        ("anthropic", anthropic_models),
        ("gemini", gemini_models),
    ]:
        p = MagicMock()
        p.list_models = AsyncMock(return_value=models or [])
        p.create_chatbot = MagicMock()
        mock_providers[api_type] = p
    return mock_providers


@pytest.mark.asyncio
async def test_add_backend_openai():
    """Test adding OpenAI-compatible backend."""
    mock_providers = _mock_providers(openai_models=["model-1", "model-2"])
    mock_providers["openai"].create_chatbot = MagicMock(return_value=OpenAIChatBot(MagicMock(), _chatbot_config()))

    with patch.dict(ChatBotManager._providers, mock_providers):
        backend = await ChatBotManager.add_backend("test-backend", "http://test:8000")

    assert backend.name == "test-backend"
    assert backend.url == "http://test:8000"
    assert backend.api_type == "openai"
    assert len(backend.models) == 2
    assert "model-1" in backend.models
    assert "model-2" in backend.models
    assert isinstance(backend.models["model-1"], OpenAIChatBot)


@pytest.mark.asyncio
async def test_add_backend_anthropic():
    """Test adding Anthropic-compatible backend."""
    mock_providers = _mock_providers(anthropic_models=["anthropic-model-1", "anthropic-model-2"])
    from peteos.chatbot import AnthropicChatBot

    mock_providers["anthropic"].create_chatbot = MagicMock(return_value=AnthropicChatBot(MagicMock(), _chatbot_config()))

    with patch.dict(ChatBotManager._providers, mock_providers):
        backend = await ChatBotManager.add_backend("anthropic-backend", "http://test:8000")

    assert backend.api_type == "anthropic"
    assert len(backend.models) == 2
    assert "anthropic-model-1" in backend.models
    assert "anthropic-model-2" in backend.models


@pytest.mark.asyncio
async def test_add_backend_gemini():
    """Test adding Gemini-compatible backend."""
    mock_providers = _mock_providers(gemini_models=["models/gemini-2.5-flash", "models/gemini-2.0-flash"])
    from peteos.chatbot import GeminiChatBot

    mock_providers["gemini"].create_chatbot = MagicMock(return_value=GeminiChatBot(MagicMock(), _chatbot_config()))

    with patch.dict(ChatBotManager._providers, mock_providers):
        backend = await ChatBotManager.add_backend("gemini-backend", "https://generativelanguage.googleapis.com")

    assert backend.api_type == "gemini"
    assert len(backend.models) == 2
    assert "models/gemini-2.5-flash" in backend.models


@pytest.mark.asyncio
async def test_add_backend_explicit_api_type():
    """Backend with explicit api_type skips auto-detection."""
    mock_providers = _mock_providers(openai_models=["model-1"])
    mock_providers["openai"].create_chatbot = MagicMock(return_value=OpenAIChatBot(MagicMock(), _chatbot_config()))

    with patch.dict(ChatBotManager._providers, mock_providers):
        backend = await ChatBotManager.add_backend(
            "explicit-backend", "http://test:8000", api_type="openai"
        )

    assert backend.api_type == "openai"
    assert len(backend.models) == 1
    # Verify list_models was called with the api_key kwarg
    mock_providers["openai"].list_models.assert_called_once_with("http://test:8000", None)


@pytest.mark.asyncio
async def test_add_backend_invalid_api_type():
    """Adding backend with invalid api_type raises ValueError."""
    with pytest.raises(ValueError, match="Invalid api_type"):
        await ChatBotManager.add_backend("bad", "http://test:8000", api_type="foobar")


@pytest.mark.asyncio
async def test_add_backend_duplicate_name():
    """Adding backend with duplicate name raises error."""
    mock_providers = _mock_providers(openai_models=["model-1"])
    mock_providers["openai"].create_chatbot = MagicMock(return_value=OpenAIChatBot(MagicMock(), _chatbot_config()))

    with patch.dict(ChatBotManager._providers, mock_providers):
        await ChatBotManager.add_backend("test", "http://test:8000")

    with pytest.raises(ValueError, match="Backend 'test' already exists"):
        await ChatBotManager.add_backend("test", "http://test:8000")


@pytest.mark.asyncio
async def test_add_backend_empty_models():
    """Adding backend with no models raises error."""
    mock_providers = _mock_providers(openai_models=[])

    # For auto-detect, _detect_api_and_list_models falls through all providers
    with patch.dict(ChatBotManager._providers, mock_providers):
        with pytest.raises(RuntimeError, match="Failed to detect API type"):
            await ChatBotManager.add_backend("empty-backend", "http://test:8000")


@pytest.mark.asyncio
async def test_add_backend_no_provider():
    """Adding backend with no registered provider raises error."""
    with patch.object(ChatBotManager, "_providers", {"openai": MagicMock()}):
        with pytest.raises(RuntimeError, match="No provider registered"):
            await ChatBotManager.add_backend(
                "unknown-backend", "http://test:8000", api_type="anthropic"
            )


@pytest.mark.asyncio
async def test_add_backend_auto_detect_opens_ai():
    """Auto-detection tries OpenAI first and succeeds."""
    mock_providers = _mock_providers(openai_models=["model-1"])
    mock_providers["openai"].create_chatbot = MagicMock(return_value=OpenAIChatBot(MagicMock(), _chatbot_config()))

    with patch.dict(ChatBotManager._providers, mock_providers):
        backend = await ChatBotManager.add_backend("auto-backend", "http://test:8000")

    assert backend.api_type == "openai"


@pytest.mark.asyncio
async def test_remove_backend_success():
    """Test successful backend removal."""
    mock_providers = _mock_providers(openai_models=["model-1"])
    mock_providers["openai"].create_chatbot = MagicMock(return_value=OpenAIChatBot(MagicMock(), _chatbot_config()))

    with patch.dict(ChatBotManager._providers, mock_providers):
        await ChatBotManager.add_backend("test", "http://test:8000")

    assert "test" in ChatBotManager._backends
    assert ChatBotManager.remove_backend("test") is True
    assert "test" not in ChatBotManager._backends
    assert "test" not in ChatBotManager._clients


@pytest.mark.asyncio
async def test_remove_backend_not_found():
    """Removing non-existent backend returns False."""
    assert ChatBotManager.remove_backend("nonexistent") is False
    assert len(ChatBotManager._backends) == 0


@pytest.mark.asyncio
async def test_list_chatbots_all():
    """Test listing all chatbots with wildcard pattern."""
    mock_providers = _mock_providers(openai_models=["model-a", "model-b"])
    mock_providers["openai"].create_chatbot = MagicMock(return_value=OpenAIChatBot(MagicMock(), _chatbot_config()))

    with patch.dict(ChatBotManager._providers, mock_providers):
        await ChatBotManager.add_backend("backend1", "http://test1:8000")

    results = ChatBotManager.list_chatbots(".*")

    assert len(results) == 2
    assert results[0][0] == "model-a"
    assert results[1][0] == "model-b"


@pytest.mark.asyncio
async def test_list_chatbots_regex_filter():
    """Test filtering chatbots by regex pattern."""
    mock_providers = _mock_providers(openai_models=["qwen-7b", "qwen-14b", "mistral-7b"])
    mock_providers["openai"].create_chatbot = MagicMock(return_value=OpenAIChatBot(MagicMock(), _chatbot_config()))

    with patch.dict(ChatBotManager._providers, mock_providers):
        await ChatBotManager.add_backend("llm-backend", "http://test:8000")

    results = ChatBotManager.list_chatbots("qwen.*")

    assert len(results) == 2
    assert all("qwen" in model_id for model_id, _ in results)
    assert all("mistral" not in model_id for model_id, _ in results)


@pytest.mark.asyncio
async def test_list_chatbots_no_match():
    """Test listing chatbots with no matches."""
    mock_providers = _mock_providers(openai_models=["model-1"])
    mock_providers["openai"].create_chatbot = MagicMock(return_value=OpenAIChatBot(MagicMock(), _chatbot_config()))

    with patch.dict(ChatBotManager._providers, mock_providers):
        await ChatBotManager.add_backend("test", "http://test:8000")

    results = ChatBotManager.list_chatbots("nonexistent.*")

    assert len(results) == 0


@pytest.mark.asyncio
async def test_list_chatbots_multiple_backends():
    """Test listing chatbots from multiple backends."""
    from peteos.chatbot import AnthropicChatBot

    mock_providers = {}
    for api_type, models in [("openai", ["openai-model"]), ("anthropic", ["anthropic-model"])]:
        p = MagicMock()
        p.create_chatbot = MagicMock(return_value=OpenAIChatBot(MagicMock(), _chatbot_config()) if api_type == "openai" else AnthropicChatBot(MagicMock(), _chatbot_config()))
        p.list_models = AsyncMock(return_value=models)
        mock_providers[api_type] = p

    def auto_detect(backend_name, api_key):
        """Return the api_type matching the backend name."""
        if "openai" in backend_name:
            return "openai", ["openai-model"]
        return "anthropic", ["anthropic-model"]

    with patch.object(ChatBotManager, "_detect_api_and_list_models", side_effect=auto_detect), \
         patch.dict(ChatBotManager._providers, mock_providers):
        await ChatBotManager.add_backend("openai-backend", "http://openai:8000")
        await ChatBotManager.add_backend("anthropic-backend", "http://anthropic:8000")

    results = ChatBotManager.list_chatbots(".*")

    assert len(results) == 2
    model_ids = [model_id for model_id, _ in results]
    assert "openai-model" in model_ids
    assert "anthropic-model" in model_ids


@pytest.mark.asyncio
async def test_load_from_json_openai():
    """Test loading OpenAI backend from JSON."""
    mock_providers = _mock_providers(openai_models=["local-model-1", "local-model-2"])
    mock_providers["openai"].create_chatbot = MagicMock(return_value=OpenAIChatBot(MagicMock(), _chatbot_config()))

    json_obj = {"backends": [{"name": "local-openai", "url": "http://localhost:8000"}]}

    with patch.dict(ChatBotManager._providers, mock_providers):
        await ChatBotManager.load_from_json(json_obj)

    assert "local-openai" in ChatBotManager._backends
    backend = ChatBotManager._backends["local-openai"]
    assert backend.api_type == "openai"
    assert len(backend.models) == 2
    assert "local-model-1" in backend.models


@pytest.mark.asyncio
async def test_load_from_json_anthropic():
    """Test loading Anthropic backend from JSON."""
    from peteos.chatbot import AnthropicChatBot

    mock_providers = _mock_providers(anthropic_models=["claude-3-opus"])
    mock_providers["anthropic"].create_chatbot = MagicMock(return_value=AnthropicChatBot(MagicMock(), _chatbot_config()))

    json_obj = {"backends": [{"name": "local-anthropic", "url": "http://localhost:8001"}]}

    with patch.dict(ChatBotManager._providers, mock_providers):
        await ChatBotManager.load_from_json(json_obj)

    assert "local-anthropic" in ChatBotManager._backends
    backend = ChatBotManager._backends["local-anthropic"]
    assert backend.api_type == "anthropic"
    assert len(backend.models) == 1
    assert "claude-3-opus" in backend.models


@pytest.mark.asyncio
async def test_load_from_json_multiple_backends():
    """Test loading multiple backends from JSON."""
    from peteos.chatbot import AnthropicChatBot

    mock_providers = {}
    for api_type, models in [("openai", ["model1", "model2"]), ("anthropic", ["anthropic-model"])]:
        p = MagicMock()
        p.create_chatbot = MagicMock(return_value=OpenAIChatBot(MagicMock(), _chatbot_config()) if api_type == "openai" else AnthropicChatBot(MagicMock(), _chatbot_config()))
        p.list_models = AsyncMock(return_value=models)
        mock_providers[api_type] = p

    json_obj = {
        "backends": [
            {"name": "backend1", "url": "http://backend1:8000"},
            {"name": "backend2", "url": "http://backend2:8000"},
        ]
    }

    def auto_detect(backend_name, api_key):
        if "backend1" in backend_name:
            return "openai", ["model1", "model2"]
        return "anthropic", ["anthropic-model"]

    with patch.object(ChatBotManager, "_detect_api_and_list_models", side_effect=auto_detect), \
         patch.dict(ChatBotManager._providers, mock_providers):
        await ChatBotManager.load_from_json(json_obj)

    assert len(ChatBotManager._backends) == 2
    assert "backend1" in ChatBotManager._backends
    assert "backend2" in ChatBotManager._backends
    assert ChatBotManager._backends["backend1"].api_type == "openai"
    assert ChatBotManager._backends["backend2"].api_type == "anthropic"


@pytest.mark.asyncio
async def test_load_from_json_clears_existing():
    """Test that load_from_json clears existing backends."""
    mock_providers = _mock_providers(openai_models=["old-model"])
    mock_providers["openai"].create_chatbot = MagicMock(return_value=OpenAIChatBot(MagicMock(), _chatbot_config()))

    with patch.dict(ChatBotManager._providers, mock_providers):
        await ChatBotManager.add_backend("old", "http://old:8000")

    assert "old" in ChatBotManager._backends

    json_obj = {"backends": [{"name": "new", "url": "http://new:8000"}]}
    mock_providers2 = _mock_providers(openai_models=["new-model"])
    mock_providers2["openai"].create_chatbot = MagicMock(return_value=OpenAIChatBot(MagicMock(), _chatbot_config()))

    with patch.dict(ChatBotManager._providers, mock_providers2):
        await ChatBotManager.load_from_json(json_obj)

    assert "old" not in ChatBotManager._backends
    assert "new" in ChatBotManager._backends


@pytest.mark.asyncio
async def test_load_from_json_empty():
    """Test loading empty JSON object."""
    json_obj = {"backends": []}
    await ChatBotManager.load_from_json(json_obj)

    assert len(ChatBotManager._backends) == 0
    assert len(ChatBotManager._clients) == 0


@pytest.mark.asyncio
async def test_load_from_file(tmp_path):
    """Test loading from JSON file."""
    json_file = tmp_path / "backends.json"
    json_file.write_text('{"backends": [{"name": "file-backend", "url": "http://file:8000"}]}')

    mock_providers = _mock_providers(openai_models=["file-model"])
    mock_providers["openai"].create_chatbot = MagicMock(return_value=OpenAIChatBot(MagicMock(), _chatbot_config()))

    with patch.dict(ChatBotManager._providers, mock_providers):
        await ChatBotManager.load_from_file(str(json_file))

    assert "file-backend" in ChatBotManager._backends
    assert ChatBotManager._backends["file-backend"].api_type == "openai"
    assert "file-model" in ChatBotManager._backends["file-backend"].models


@pytest.mark.asyncio
async def test_load_from_file_invalid_json(tmp_path):
    """Test loading from file with invalid JSON."""
    json_file = tmp_path / "bad.json"
    json_file.write_text("{invalid json")

    with pytest.raises(ValueError):
        await ChatBotManager.load_from_file(str(json_file))


@pytest.mark.asyncio
async def test_reset_clears_all_state():
    """reset() clears backends and clients."""
    mock_providers = _mock_providers(openai_models=["model-1"])
    mock_providers["openai"].create_chatbot = MagicMock(return_value=OpenAIChatBot(MagicMock(), _chatbot_config()))

    with patch.dict(ChatBotManager._providers, mock_providers):
        await ChatBotManager.add_backend("test", "http://test:8000")

    assert len(ChatBotManager._backends) == 1
    assert len(ChatBotManager._clients) == 1

    ChatBotManager.reset()

    assert len(ChatBotManager._backends) == 0
    assert len(ChatBotManager._clients) == 0


@pytest.mark.asyncio
async def test_reset_allows_reload():
    """After reset, same backend name can be added again."""
    mock_providers = _mock_providers(openai_models=["model-1"])
    mock_providers["openai"].create_chatbot = MagicMock(return_value=OpenAIChatBot(MagicMock(), _chatbot_config()))

    with patch.dict(ChatBotManager._providers, mock_providers):
        await ChatBotManager.add_backend("test", "http://test:8000")

    ChatBotManager.reset()

    with patch.dict(ChatBotManager._providers, mock_providers):
        backend = await ChatBotManager.add_backend("test", "http://test:8000")

    assert backend.name == "test"


@pytest.mark.asyncio
async def test_init_sets_timeout():
    """Instantiating ChatBotManager sets the class-level timeout."""
    ChatBotManager(timeout=60.0)
    assert ChatBotManager._timeout == 60.0


@pytest.mark.asyncio
async def test_init_none_keeps_existing():
    """Passing None to __init__ doesn't change timeout."""
    ChatBotManager._timeout = 30.0
    manager = ChatBotManager()
    assert ChatBotManager._timeout == 30.0


class TestBackendInfo:
    """Tests for BackendInfo dataclass."""

    def test_backend_info_creation(self):
        """BackendInfo can be instantiated."""
        info = BackendInfo(name="test", url="http://test:8000", api_type="openai", models={})
        assert info.name == "test"
        assert info.url == "http://test:8000"
        assert info.api_type == "openai"
        assert info.models == {}

    def test_backend_info_with_models(self):
        """BackendInfo stores models dict."""
        models = {"m1": object(), "m2": object()}
        info = BackendInfo(name="test", url="http://t:8000", api_type="openai", models=models)
        assert info.models == models


class TestChatBotPriority:
    """Tests for model priority feature."""

    @pytest.mark.asyncio
    async def test_add_backend_with_model_priorities(self):
        """Test adding backend with model_priorities applies priorities to each model."""
        mock_providers = _mock_providers(openai_models=["gpt-4", "gpt-3.5"])
        # Use the real OpenAI provider's create_chatbot to verify priority flows through
        from peteos.chatbot.openaiprovider import OpenAIChatBotProvider

        mock_providers["openai"].create_chatbot = OpenAIChatBotProvider().create_chatbot

        with patch.dict(ChatBotManager._providers, mock_providers):
            backend = await ChatBotManager.add_backend(
                "prio-backend",
                "http://test:8000",
                model_priorities={"gpt-4": 10, "gpt-3.5": 0},
            )

        assert backend.models["gpt-4"].priority == 10
        assert backend.models["gpt-3.5"].priority == 0

    @pytest.mark.asyncio
    async def test_add_backend_model_priorities_missing_model_defaults_to_zero(self):
        """Model not in model_priorities dict gets priority 0."""
        mock_providers = _mock_providers(openai_models=["gpt-4", "gpt-5"])
        from peteos.chatbot.openaiprovider import OpenAIChatBotProvider

        mock_providers["openai"].create_chatbot = OpenAIChatBotProvider().create_chatbot

        with patch.dict(ChatBotManager._providers, mock_providers):
            # Only gpt-4 has priority; gpt-5 should default to 0
            backend = await ChatBotManager.add_backend(
                "partial-prio-backend",
                "http://test:8000",
                model_priorities={"gpt-4": 5},
            )

        assert backend.models["gpt-4"].priority == 5
        assert backend.models["gpt-5"].priority == 0

    @pytest.mark.asyncio
    async def test_load_from_json_with_model_priorities(self):
        """Test load_from_json applies model_priorities from config."""
        mock_providers = _mock_providers(openai_models=["premium", "standard"])
        from peteos.chatbot.openaiprovider import OpenAIChatBotProvider

        mock_providers["openai"].create_chatbot = OpenAIChatBotProvider().create_chatbot

        json_obj = {
            "backends": [
                {
                    "name": "prio-json-backend",
                    "url": "http://test:8000",
                    "model_priorities": {"premium": 20, "standard": 0},
                }
            ]
        }

        with patch.dict(ChatBotManager._providers, mock_providers):
            await ChatBotManager.load_from_json(json_obj)

        assert ChatBotManager._backends["prio-json-backend"].models["premium"].priority == 20
        assert ChatBotManager._backends["prio-json-backend"].models["standard"].priority == 0

    def test_list_chatbots_sorted_by_priority(self):
        """Test list_chatbots returns models sorted by priority descending."""
        mock_providers = _mock_providers(openai_models=["a", "b", "c"])
        mock_providers["openai"].create_chatbot = MagicMock(
            return_value=OpenAIChatBot(MagicMock(), _chatbot_config())
        )

        with patch.dict(ChatBotManager._providers, mock_providers):
            ChatBotManager._backends["t"] = BackendInfo(
                name="t", url="http://t:8000", api_type="openai", models={}
            )
            for mid, prio in [("a", 0), ("b", 10), ("c", 5)]:
                config = _chatbot_config(model=mid, priority=prio)
                mock_providers["openai"].create_chatbot.return_value = OpenAIChatBot(
                    MagicMock(), config
                )
                ChatBotManager._backends["t"].models[mid] = mock_providers[
                    "openai"
                ].create_chatbot.return_value

        results = ChatBotManager.list_chatbots(".*")
        assert len(results) == 3
        assert results[0][0] == "b"  # priority 10
        assert results[1][0] == "c"  # priority 5
        assert results[2][0] == "a"  # priority 0

    @pytest.mark.asyncio
    async def test_list_chatbots_tiebreaker_is_alphabetical(self):
        """When priorities are equal, models are sorted alphabetically."""
        mock_providers = _mock_providers(openai_models=["zebra", "alpha"])
        mock_providers["openai"].create_chatbot = MagicMock(
            return_value=OpenAIChatBot(MagicMock(), _chatbot_config())
        )

        with patch.dict(ChatBotManager._providers, mock_providers):
            await ChatBotManager.add_backend("tie-backend", "http://test:8000")

        results = ChatBotManager.list_chatbots(".*")
        assert len(results) == 2
        assert results[0][0] == "alpha"  # alphabetical tiebreaker
        assert results[1][0] == "zebra"
