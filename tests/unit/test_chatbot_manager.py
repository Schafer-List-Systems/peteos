"""Tests for ChatBotManager."""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from peteos.chatbot import ChatBotManager, BackendInfo
from peteos.chatbot import OpenAIChatBot, AnthropicChatBot


class _MockResponse:
    """Minimal mock HTTP response for httpx.AsyncClient."""

    def __init__(self, json_data):
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        pass


def _make_httpx_mock(return_value=None, side_effect=None):
    """Create a mock httpx.AsyncClient that returns pre-set responses.

    Usage:
        with patch("httpx.AsyncClient", _make_httpx_mock(return_value={"data": [...]})):
            ...
    """
    if side_effect is not None:
        responses = list(side_effect)
        _idx = [0]

        class _MockClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def get(self, url, **kwargs):
                idx = _idx[0]
                _idx[0] += 1
                if idx < len(responses):
                    return _MockResponse(responses[idx])
                raise StopAsyncIteration

        return _MockClient

    class _MockClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def get(self, url, **kwargs):
            return _MockResponse(return_value)

    return _MockClient


@pytest.fixture(autouse=True)
def _reset_chatbot_manager():
    """Reset ChatBotManager class-level state before and after each test."""
    ChatBotManager.reset()
    yield
    ChatBotManager.reset()


class TestChatBotManagerAddBackend:
    """Tests for add_backend method."""

    @pytest.mark.asyncio
    async def test_add_backend_openai(self):
        """Test adding OpenAI-compatible backend."""

        mock_response = {
            "data": [
                {"id": "model-1"},
                {"id": "model-2"},
            ]
        }

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            backend = await ChatBotManager.add_backend("test-backend", "http://test:8000")

        assert backend.name == "test-backend"
        assert backend.url == "http://test:8000"
        assert backend.api_type == "openai"
        assert len(backend.models) == 2
        assert "model-1" in backend.models
        assert "model-2" in backend.models
        assert isinstance(backend.models["model-1"], OpenAIChatBot)
        assert isinstance(backend.models["model-2"], OpenAIChatBot)

    @pytest.mark.asyncio
    async def test_add_backend_anthropic(self):
        """Test adding Anthropic-compatible backend."""

        # Anthropic provider returns {"models": [...]} — but _detect_api_and_list_models
        # tries OpenAI first (which looks for "data" key). We must also return "data" in the
        # OpenAI probe so it succeeds and the anthropic response goes unused.
        # However the Anthropic provider also tries /v1/models looking for "data" key,
        # so we need to return "data" as well.
        mock_response = {
            "data": [
                {"id": "anthropic-model-1"},
                {"id": "anthropic-model-2"},
            ]
        }

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            backend = await ChatBotManager.add_backend(
                "anthropic-backend", "http://test:8000", api_type="anthropic"
            )

        assert backend.api_type == "anthropic"
        assert len(backend.models) == 2
        assert "anthropic-model-1" in backend.models
        assert "anthropic-model-2" in backend.models
        assert isinstance(backend.models["anthropic-model-1"], AnthropicChatBot)

    @pytest.mark.asyncio
    async def test_add_backend_duplicate_name(self):
        """Test adding backend with duplicate name raises error."""

        mock_response = {"data": [{"id": "model-1"}]}

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            await ChatBotManager.add_backend("test", "http://test:8000")

        with pytest.raises(ValueError, match="Backend 'test' already exists"):
            with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
                await ChatBotManager.add_backend("test", "http://test:8000")

    @pytest.mark.asyncio
    async def test_add_backend_empty_models(self):
        """Test adding backend with no models raises error."""

        mock_response = {"data": []}

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            with pytest.raises(RuntimeError, match="Failed to detect API type"):
                await ChatBotManager.add_backend("empty-backend", "http://test:8000")

    @pytest.mark.asyncio
    async def test_add_backend_unrecognized_format(self):
        """Test adding backend with unrecognized response format raises error."""

        mock_response = {"unknown": "format"}

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            with pytest.raises(RuntimeError, match="Failed to detect API type"):
                await ChatBotManager.add_backend("unknown-backend", "http://test:8000")


class TestChatBotManagerRemoveBackend:
    """Tests for remove_backend method."""

    @pytest.mark.asyncio
    async def test_remove_backend_success(self):
        """Test successful backend removal."""

        mock_response = {"data": [{"id": "model-1"}]}

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            await ChatBotManager.add_backend("test", "http://test:8000")

        assert "test" in ChatBotManager._backends
        assert ChatBotManager.remove_backend("test") is True
        assert "test" not in ChatBotManager._backends

    @pytest.mark.asyncio
    async def test_remove_backend_not_found(self):
        """Test removing non-existent backend returns False."""

        assert ChatBotManager.remove_backend("nonexistent") is False
        assert len(ChatBotManager._backends) == 0


class TestChatBotManagerListChatbots:
    """Tests for list_chatbots method."""

    @pytest.mark.asyncio
    async def test_list_chatbots_all(self):
        """Test listing all chatbots with wildcard pattern."""

        mock_response = {"data": [{"id": "model-a"}, {"id": "model-b"}]}

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            await ChatBotManager.add_backend("backend1", "http://test1:8000")

        results = ChatBotManager.list_chatbots(".*")

        assert len(results) == 2
        assert results[0] == ("model-a", AnyChatBot())
        assert results[1] == ("model-b", AnyChatBot())

    @pytest.mark.asyncio
    async def test_list_chatbots_regex_filter(self):
        """Test filtering chatbots by regex pattern."""

        mock_response = {
            "data": [
                {"id": "qwen-7b"},
                {"id": "qwen-14b"},
                {"id": "mistral-7b"},
            ]
        }

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            await ChatBotManager.add_backend("llm-backend", "http://test:8000")

        results = ChatBotManager.list_chatbots("qwen.*")

        assert len(results) == 2
        assert all("qwen" in model_id for model_id, _ in results)
        assert all("mistral" not in model_id for model_id, _ in results)

    @pytest.mark.asyncio
    async def test_list_chatbots_multiple_backends(self):
        """Test listing chatbots from multiple backends."""

        responses_side_effect = [
            {"data": [{"id": "openai-model"}]},
            {"data": [{"id": "anthropic-model"}]},
        ]

        with patch(
            "httpx.AsyncClient",
            _make_httpx_mock(side_effect=responses_side_effect),
        ):
            await ChatBotManager.add_backend("openai-backend", "http://openai:8000")
            await ChatBotManager.add_backend("anthropic-backend", "http://anthropic:8000")

        results = ChatBotManager.list_chatbots(".*")

        assert len(results) == 2
        model_ids = [model_id for model_id, _ in results]
        assert "openai-model" in model_ids
        assert "anthropic-model" in model_ids

    @pytest.mark.asyncio
    async def test_list_chatbots_no_match(self):
        """Test listing chatbots with no matches."""

        mock_response = {"data": [{"id": "model-1"}]}

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            await ChatBotManager.add_backend("test", "http://test:8000")

        results = ChatBotManager.list_chatbots("nonexistent.*")

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_list_chatbots_sorted_by_priority(self):
        """Test that list_chatbots sorts by priority descending, then model_id."""

        mock_response = {
            "data": [
                {"id": "gpt-3.5-turbo"},
                {"id": "gpt-4"},
                {"id": "gpt-4o"},
            ]
        }

        with patch(
            "httpx.AsyncClient",
            _make_httpx_mock(return_value=mock_response),
        ):
            await ChatBotManager.add_backend(
                "priority-backend",
                "http://test:8000",
                model_priorities={"gpt-4o": 10, "gpt-4": 5},
            )

        results = ChatBotManager.list_chatbots("gpt")

        assert len(results) == 3
        assert results[0][0] == "gpt-4o"  # priority 10
        assert results[1][0] == "gpt-4"   # priority 5
        assert results[2][0] == "gpt-3.5-turbo"  # priority 0 (default)

    @pytest.mark.asyncio
    async def test_list_chatbots_priority_breaks_alphabetical_tie(self):
        """When priorities differ, they override alphabetical ordering."""

        mock_response = {
            "data": [
                {"id": "gpt-3.5-turbo"},
                {"id": "gpt-4"},
            ]
        }

        with patch(
            "httpx.AsyncClient",
            _make_httpx_mock(return_value=mock_response),
        ):
            # gpt-3.5 is alphabetically first, but gpt-4 has higher priority
            await ChatBotManager.add_backend(
                "tie-breaker-backend",
                "http://test:8000",
                model_priorities={"gpt-4": 1},
            )

        results = ChatBotManager.list_chatbots("gpt")

        assert results[0][0] == "gpt-4"  # higher priority, despite later alphabet
        assert results[1][0] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_list_chatbots_default_priority_is_zero(self):
        """Models without explicit priority get 0 and stay alphabetical."""

        mock_response = {
            "data": [
                {"id": "model-b"},
                {"id": "model-a"},
            ]
        }

        with patch(
            "httpx.AsyncClient",
            _make_httpx_mock(return_value=mock_response),
        ):
            await ChatBotManager.add_backend(
                "default-priority-backend",
                "http://test:8000",
            )

        results = ChatBotManager.list_chatbots(".*")

        assert len(results) == 2
        assert results[0][0] == "model-a"  # alphabetical tiebreaker
        assert results[1][0] == "model-b"

    @pytest.mark.asyncio
    async def test_chatbot_priority_property(self):
        """ChatBot instances expose priority via .priority property."""

        mock_response = {"data": [{"id": "priority-model"}]}

        with patch(
            "httpx.AsyncClient",
            _make_httpx_mock(return_value=mock_response),
        ):
            await ChatBotManager.add_backend(
                "prop-backend",
                "http://test:8000",
                model_priorities={"priority-model": 42},
            )

        chatbot = ChatBotManager._backends["prop-backend"].models["priority-model"]
        assert chatbot.priority == 42


class TestChatBotManagerLoadFromJson:
    """Tests for load_from_json method."""

    @pytest.mark.asyncio
    async def test_load_from_json_openai(self):
        """Test loading OpenAI backend from JSON."""

        json_obj = {
            "backends": [
                {
                    "name": "local-openai",
                    "url": "http://localhost:8000",
                }
            ]
        }

        mock_response = {
            "data": [
                {"id": "local-model-1"},
                {"id": "local-model-2"},
            ]
        }

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            await ChatBotManager.load_from_json(json_obj)

        assert "local-openai" in ChatBotManager._backends
        backend = ChatBotManager._backends["local-openai"]
        assert backend.api_type == "openai"
        assert len(backend.models) == 2
        assert "local-model-1" in backend.models

    @pytest.mark.asyncio
    async def test_load_from_json_anthropic(self):
        """Test loading Anthropic backend from JSON."""

        json_obj = {
            "backends": [
                {
                    "name": "local-anthropic",
                    "url": "http://localhost:8001",
                    "api_type": "anthropic",
                }
            ]
        }

        mock_response = {
            "data": [
                {"id": "claude-3-opus"},
            ]
        }

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            await ChatBotManager.load_from_json(json_obj)

        assert "local-anthropic" in ChatBotManager._backends
        backend = ChatBotManager._backends["local-anthropic"]
        assert backend.api_type == "anthropic"
        assert len(backend.models) == 1
        assert "claude-3-opus" in backend.models

    @pytest.mark.asyncio
    async def test_load_from_json_multiple_backends(self):
        """Test loading multiple backends from JSON."""

        json_obj = {
            "backends": [
                {
                    "name": "backend1",
                    "url": "http://backend1:8000",
                },
                {
                    "name": "backend2",
                    "url": "http://backend2:8000",
                    "api_type": "anthropic",
                },
            ]
        }

        responses_side_effect = [
            {"data": [{"id": "model1"}, {"id": "model2"}]},
            {"data": [{"id": "anthropic-model"}]},
        ]

        with patch(
            "httpx.AsyncClient",
            _make_httpx_mock(side_effect=responses_side_effect),
        ):
            await ChatBotManager.load_from_json(json_obj)

        assert len(ChatBotManager._backends) == 2
        assert "backend1" in ChatBotManager._backends
        assert "backend2" in ChatBotManager._backends
        assert ChatBotManager._backends["backend1"].api_type == "openai"
        assert ChatBotManager._backends["backend2"].api_type == "anthropic"

    @pytest.mark.asyncio
    async def test_load_from_json_clears_existing(self):
        """Test that load_from_json clears existing backends."""

        mock_response = {"data": [{"id": "old-model"}]}
        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            await ChatBotManager.add_backend("old", "http://old:8000")

        assert "old" in ChatBotManager._backends

        # Load new JSON
        json_obj = {
            "backends": [
                {
                    "name": "new",
                    "url": "http://new:8000",
                }
            ]
        }

        mock_new_response = {"data": [{"id": "new-model"}]}
        with patch(
            "httpx.AsyncClient", _make_httpx_mock(return_value=mock_new_response)
        ):
            await ChatBotManager.load_from_json(json_obj)

        assert "old" not in ChatBotManager._backends
        assert "new" in ChatBotManager._backends

    @pytest.mark.asyncio
    async def test_load_from_json_empty(self):
        """Test loading empty JSON object."""

        json_obj = {"backends": []}

        await ChatBotManager.load_from_json(json_obj)

        assert len(ChatBotManager._backends) == 0

    @pytest.mark.asyncio
    async def test_load_from_json_with_model_priorities(self):
        """Test loading backend with model_priorities from JSON."""

        json_obj = {
            "backends": [
                {
                    "name": "prioritized-backend",
                    "url": "http://prioritized:8000",
                    "model_priorities": {
                        "premium-model": 10,
                        "standard-model": 0,
                    },
                }
            ]
        }

        mock_response = {
            "data": [
                {"id": "premium-model"},
                {"id": "standard-model"},
            ]
        }

        with patch(
            "httpx.AsyncClient",
            _make_httpx_mock(return_value=mock_response),
        ):
            await ChatBotManager.load_from_json(json_obj)

        results = ChatBotManager.list_chatbots(".*")

        assert len(results) == 2
        assert results[0][0] == "premium-model"  # priority 10
        assert results[1][0] == "standard-model"  # priority 0

    @pytest.mark.asyncio
    async def test_load_from_json_with_empty_model_priorities(self):
        """Test loading backend with empty model_priorities dict."""

        json_obj = {
            "backends": [
                {
                    "name": "no-priority-backend",
                    "url": "http://noprio:8000",
                    "model_priorities": {},
                }
            ]
        }

        mock_response = {
            "data": [
                {"id": "model-b"},
                {"id": "model-a"},
            ]
        }

        with patch(
            "httpx.AsyncClient",
            _make_httpx_mock(return_value=mock_response),
        ):
            await ChatBotManager.load_from_json(json_obj)

        results = ChatBotManager.list_chatbots(".*")

        assert len(results) == 2
        assert results[0][0] == "model-a"  # alphabetical tiebreaker
        assert results[1][0] == "model-b"


class TestChatBotManagerLoadFromFile:
    """Tests for load_from_file method."""

    @pytest.mark.asyncio
    async def test_load_from_file(self, tmp_path):
        """Test loading from JSON file."""

        # Create test JSON file
        json_file = tmp_path / "backends.json"
        json_file.write_text(
            '{"backends": [{"name": "file-backend", "url": "http://file:8000"}]}'
        )

        mock_response = {"data": [{"id": "file-model"}]}

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            await ChatBotManager.load_from_file(str(json_file))

        assert "file-backend" in ChatBotManager._backends
        assert ChatBotManager._backends["file-backend"].api_type == "openai"
        assert "file-model" in ChatBotManager._backends["file-backend"].models


class TestChatBotConfigDiscovery:
    """Tests for load_from_config and config resolution."""

    @pytest.mark.asyncio
    async def test_load_from_config_no_file(self):
        """When no config file exists, logs warning and returns gracefully."""
        with patch.object(ChatBotManager, "_resolve_config_path", return_value=None):
            await ChatBotManager.load_from_config()

        assert ChatBotManager._config_dir is None
        assert len(ChatBotManager._backends) == 0

    @pytest.mark.asyncio
    async def test_load_from_config_from_env_var(self, tmp_path):
        """PETEOS_CONFIG env var takes highest priority."""
        config_file = tmp_path / "peteos.json"
        config_file.write_text('{"backends": [{"name": "env-backend", "url": "http://env:8000"}]}')

        mock_response = {"data": [{"id": "env-model"}]}

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            with patch.object(ChatBotManager, "_resolve_config_path", return_value=str(config_file)):
                await ChatBotManager.load_from_config()

        assert ChatBotManager._config_dir == str(tmp_path)
        assert "env-backend" in ChatBotManager._backends

    @pytest.mark.asyncio
    async def test_load_from_config_sets_config_dir(self, tmp_path):
        """Config directory is set to the parent of the loaded peteos.json."""
        config_dir = tmp_path / "peteos_config"
        config_dir.mkdir()
        config_file = config_dir / "peteos.json"
        config_file.write_text('{"backends": [{"name": "dir-backend", "url": "http://dir:8000"}]}')

        mock_response = {"data": [{"id": "dir-model"}]}

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            with patch.object(ChatBotManager, "_resolve_config_path", return_value=str(config_file)):
                await ChatBotManager.load_from_config()

        assert ChatBotManager._config_dir == str(config_dir)

    @pytest.mark.asyncio
    async def test_config_dir_reset_on_reset(self):
        """_config_dir is cleared on ChatBotManager.reset()."""
        ChatBotManager._config_dir = "/test/path"
        ChatBotManager.reset()
        assert ChatBotManager._config_dir is None

    def test_resolve_config_path_env_first(self, tmp_path):
        """PETEOS_CONFIG env var resolves before XDG/system/local."""
        env_file = tmp_path / "env_peteos.json"
        env_file.write_text('{"backends": []}')

        with patch("os.environ.get", side_effect=lambda key, default=None: {
            "PETEOS_CONFIG": str(env_file),
        }.get(key, default)):
            resolved = ChatBotManager._resolve_config_path()

        assert resolved == str(env_file)

    def test_resolve_config_path_xdg_config(self, tmp_path):
        """$XDG_CONFIG_HOME/peteos/peteos.json is resolved."""
        xdg_path = tmp_path / "my_config" / "peteos" / "peteos.json"
        xdg_path.parent.mkdir(parents=True)
        xdg_path.write_text('{"backends": []}')

        xdg_str = str(tmp_path / "my_config")

        def fake_is_file(self):
            return str(self) == str(xdg_path)

        with patch("os.environ.get", side_effect=lambda key, default=None: {
            "XDG_CONFIG_HOME": xdg_str,
        }.get(key, default)):
            with patch("pathlib.Path.is_file", fake_is_file):
                resolved = ChatBotManager._resolve_config_path()

        assert resolved == str(xdg_path)

    def test_resolve_config_path_no_env(self, tmp_path):
        """When no env vars set, returns None if no local peteos.json exists."""
        with patch("os.environ.get", side_effect=lambda key, default=None: default):
            with patch("pathlib.Path.is_file", return_value=False):
                resolved = ChatBotManager._resolve_config_path()

        assert resolved is None


class AnyChatBot:
    """Helper class for matching any ChatBot instance in tests."""

    def __eq__(self, other):
        return hasattr(other, "__class__") and "ChatBot" in other.__class__.__name__
