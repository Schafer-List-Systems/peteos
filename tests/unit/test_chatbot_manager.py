"""Tests for ChatBotManager."""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from peteos.chatbot import ChatBotManager, BackendInfo
from peteos.chatbot import OpenAIChatBot, AnthropicChatBot


def _make_http_client_mock(return_value=None, side_effect=None):
    """Create a side_effect-compatible HTTPClient mock factory.

    Each call creates independent mocks so side_effect lists work per-backend.
    """
    stored_side_effect = list(side_effect) if side_effect else None
    if stored_side_effect is not None:
        _side_effect_idx = [0]
        async def _shared_get(*args, **kwargs):
            idx = _side_effect_idx[0]
            _side_effect_idx[0] += 1
            if idx < len(stored_side_effect):
                return stored_side_effect[idx]
            raise StopAsyncIteration
        def factory(*args, **kwargs):
            client = MagicMock()
            client.get = AsyncMock(side_effect=_shared_get)
            return client
        return factory
    else:
        def factory(*args, **kwargs):
            client = MagicMock()
            if return_value is not None:
                client.get = AsyncMock(return_value=return_value)
            else:
                client.get = AsyncMock()
            return client
        return factory


class TestChatBotManagerAddBackend:
    """Tests for add_backend method."""

    @pytest.mark.asyncio
    async def test_add_backend_openai(self):
        """Test adding OpenAI-compatible backend."""
        manager = ChatBotManager(timeout=5.0)

        mock_response = {
            "data": [
                {"id": "model-1"},
                {"id": "model-2"},
            ]
        }

        mock_factory = _make_http_client_mock(return_value=mock_response)

        with patch(
            "peteos.chatbot.manager.HTTPClient",
            side_effect=mock_factory,
        ):
            backend = await manager.add_backend("test-backend", "http://test:8000")

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
        manager = ChatBotManager(timeout=5.0)

        mock_response = {
            "models": [
                {"id": "anthropic-model-1"},
                {"id": "anthropic-model-2"},
            ]
        }

        mock_factory = _make_http_client_mock(return_value=mock_response)

        with patch(
            "peteos.chatbot.manager.HTTPClient",
            side_effect=mock_factory,
        ):
            backend = await manager.add_backend("anthropic-backend", "http://test:8000")

        assert backend.api_type == "anthropic"
        assert len(backend.models) == 2
        assert "anthropic-model-1" in backend.models
        assert "anthropic-model-2" in backend.models
        assert isinstance(backend.models["anthropic-model-1"], AnthropicChatBot)

    @pytest.mark.asyncio
    async def test_add_backend_duplicate_name(self):
        """Test adding backend with duplicate name raises error."""
        manager = ChatBotManager(timeout=5.0)

        mock_response = {"data": [{"id": "model-1"}]}
        mock_factory = _make_http_client_mock(return_value=mock_response)

        with patch(
            "peteos.chatbot.manager.HTTPClient",
            side_effect=mock_factory,
        ):
            await manager.add_backend("test", "http://test:8000")

        with pytest.raises(ValueError, match="Backend 'test' already exists"):
            with patch(
                "peteos.chatbot.manager.HTTPClient",
                side_effect=mock_factory,
            ):
                await manager.add_backend("test", "http://test:8000")

    @pytest.mark.asyncio
    async def test_add_backend_empty_models(self):
        """Test adding backend with no models raises error."""
        manager = ChatBotManager(timeout=5.0)

        mock_response = {"data": []}
        mock_factory = _make_http_client_mock(return_value=mock_response)

        with patch(
            "peteos.chatbot.manager.HTTPClient",
            side_effect=mock_factory,
        ):
            with pytest.raises(RuntimeError, match="No models found"):
                await manager.add_backend("empty-backend", "http://test:8000")

    @pytest.mark.asyncio
    async def test_add_backend_unrecognized_format(self):
        """Test adding backend with unrecognized response format raises error."""
        manager = ChatBotManager(timeout=5.0)

        mock_response = {"unknown": "format"}
        mock_factory = _make_http_client_mock(return_value=mock_response)

        with patch(
            "peteos.chatbot.manager.HTTPClient",
            side_effect=mock_factory,
        ):
            with pytest.raises(RuntimeError, match="Could not detect API type"):
                await manager.add_backend("unknown-backend", "http://test:8000")


class TestChatBotManagerRemoveBackend:
    """Tests for remove_backend method."""

    @pytest.mark.asyncio
    async def test_remove_backend_success(self):
        """Test successful backend removal."""
        manager = ChatBotManager(timeout=5.0)

        mock_response = {"data": [{"id": "model-1"}]}
        mock_factory = _make_http_client_mock(return_value=mock_response)

        with patch(
            "peteos.chatbot.manager.HTTPClient",
            side_effect=mock_factory,
        ):
            await manager.add_backend("test", "http://test:8000")

        assert "test" in manager._backends
        assert manager.remove_backend("test") is True
        assert "test" not in manager._backends
        assert "test" not in manager._clients

    @pytest.mark.asyncio
    async def test_remove_backend_not_found(self):
        """Test removing non-existent backend returns False."""
        manager = ChatBotManager(timeout=5.0)

        assert manager.remove_backend("nonexistent") is False
        assert len(manager._backends) == 0


class TestChatBotManagerListChatbots:
    """Tests for list_chatbots method."""

    @pytest.mark.asyncio
    async def test_list_chatbots_all(self):
        """Test listing all chatbots with wildcard pattern."""
        manager = ChatBotManager(timeout=5.0)

        mock_response = {"data": [{"id": "model-a"}, {"id": "model-b"}]}
        mock_factory = _make_http_client_mock(return_value=mock_response)

        with patch(
            "peteos.chatbot.manager.HTTPClient",
            side_effect=mock_factory,
        ):
            await manager.add_backend("backend1", "http://test1:8000")

        results = manager.list_chatbots(".*")

        assert len(results) == 2
        assert results[0] == ("model-a", AnyChatBot())
        assert results[1] == ("model-b", AnyChatBot())

    @pytest.mark.asyncio
    async def test_list_chatbots_regex_filter(self):
        """Test filtering chatbots by regex pattern."""
        manager = ChatBotManager(timeout=5.0)

        mock_response = {"data": [{"id": "qwen-7b"}, {"id": "qwen-14b"}, {"id": "mistral-7b"}]}
        mock_factory = _make_http_client_mock(return_value=mock_response)

        with patch(
            "peteos.chatbot.manager.HTTPClient",
            side_effect=mock_factory,
        ):
            await manager.add_backend("llm-backend", "http://test:8000")

        results = manager.list_chatbots("qwen.*")

        assert len(results) == 2
        assert all("qwen" in model_id for model_id, _ in results)
        assert all("mistral" not in model_id for model_id, _ in results)

    @pytest.mark.asyncio
    async def test_list_chatbots_multiple_backends(self):
        """Test listing chatbots from multiple backends."""
        manager = ChatBotManager(timeout=5.0)

        mock_responses_side_effect = [
            {"data": [{"id": "openai-model"}]},
            {"models": [{"id": "anthropic-model"}]},
        ]

        mock_factory = _make_http_client_mock(side_effect=mock_responses_side_effect)

        with patch(
            "peteos.chatbot.manager.HTTPClient",
            side_effect=mock_factory,
        ):
            await manager.add_backend("openai-backend", "http://openai:8000")
            await manager.add_backend("anthropic-backend", "http://anthropic:8000")

        results = manager.list_chatbots(".*")

        assert len(results) == 2
        model_ids = [model_id for model_id, _ in results]
        assert "openai-model" in model_ids
        assert "anthropic-model" in model_ids

    @pytest.mark.asyncio
    async def test_list_chatbots_no_match(self):
        """Test listing chatbots with no matches."""
        manager = ChatBotManager(timeout=5.0)

        mock_response = {"data": [{"id": "model-1"}]}
        mock_factory = _make_http_client_mock(return_value=mock_response)

        with patch(
            "peteos.chatbot.manager.HTTPClient",
            side_effect=mock_factory,
        ):
            await manager.add_backend("test", "http://test:8000")

        results = manager.list_chatbots("nonexistent.*")

        assert len(results) == 0


class TestChatBotManagerLoadFromJson:
    """Tests for load_from_json method."""

    @pytest.mark.asyncio
    async def test_load_from_json_openai(self):
        """Test loading OpenAI backend from JSON."""
        manager = ChatBotManager(timeout=5.0)

        json_obj = {
            "backends": [
                {
                    "name": "local-openai",
                    "url": "http://localhost:8000"
                }
            ]
        }

        mock_response = {
            "data": [
                {"id": "local-model-1"},
                {"id": "local-model-2"},
            ]
        }
        mock_factory = _make_http_client_mock(return_value=mock_response)

        with patch(
            "peteos.chatbot.manager.HTTPClient",
            side_effect=mock_factory,
        ):
            await manager.load_from_json(json_obj)

        assert "local-openai" in manager._backends
        backend = manager._backends["local-openai"]
        assert backend.api_type == "openai"
        assert len(backend.models) == 2
        assert "local-model-1" in backend.models

    @pytest.mark.asyncio
    async def test_load_from_json_anthropic(self):
        """Test loading Anthropic backend from JSON."""
        manager = ChatBotManager(timeout=5.0)

        json_obj = {
            "backends": [
                {
                    "name": "local-anthropic",
                    "url": "http://localhost:8001"
                }
            ]
        }

        mock_response = {
            "models": [
                {"id": "claude-3-opus"},
            ]
        }
        mock_factory = _make_http_client_mock(return_value=mock_response)

        with patch(
            "peteos.chatbot.manager.HTTPClient",
            side_effect=mock_factory,
        ):
            await manager.load_from_json(json_obj)

        assert "local-anthropic" in manager._backends
        backend = manager._backends["local-anthropic"]
        assert backend.api_type == "anthropic"
        assert len(backend.models) == 1
        assert "claude-3-opus" in backend.models

    @pytest.mark.asyncio
    async def test_load_from_json_multiple_backends(self):
        """Test loading multiple backends from JSON."""
        manager = ChatBotManager(timeout=5.0)

        json_obj = {
            "backends": [
                {
                    "name": "backend1",
                    "url": "http://backend1:8000"
                },
                {
                    "name": "backend2",
                    "url": "http://backend2:8000"
                }
            ]
        }

        mock_responses_side_effect = [
            {"data": [{"id": "model1"}, {"id": "model2"}]},
            {"models": [{"id": "anthropic-model"}]},
        ]
        mock_factory = _make_http_client_mock(side_effect=mock_responses_side_effect)

        with patch(
            "peteos.chatbot.manager.HTTPClient",
            side_effect=mock_factory,
        ):
            await manager.load_from_json(json_obj)

        assert len(manager._backends) == 2
        assert "backend1" in manager._backends
        assert "backend2" in manager._backends
        assert manager._backends["backend1"].api_type == "openai"
        assert manager._backends["backend2"].api_type == "anthropic"

    @pytest.mark.asyncio
    async def test_load_from_json_clears_existing(self):
        """Test that load_from_json clears existing backends."""
        manager = ChatBotManager(timeout=5.0)

        # Add initial backend
        mock_response = {"data": [{"id": "old-model"}]}
        mock_factory = _make_http_client_mock(return_value=mock_response)
        with patch(
            "peteos.chatbot.manager.HTTPClient",
            side_effect=mock_factory,
        ):
            await manager.add_backend("old", "http://old:8000")

        assert "old" in manager._backends

        # Load new JSON
        json_obj = {
            "backends": [
                {
                    "name": "new",
                    "url": "http://new:8000"
                }
            ]
        }

        mock_new_response = {"models": [{"id": "new-model"}]}
        new_mock_factory = _make_http_client_mock(return_value=mock_new_response)
        with patch(
            "peteos.chatbot.manager.HTTPClient",
            side_effect=new_mock_factory,
        ):
            await manager.load_from_json(json_obj)

        assert "old" not in manager._backends
        assert "new" in manager._backends

    @pytest.mark.asyncio
    async def test_load_from_json_empty(self):
        """Test loading empty JSON object."""
        manager = ChatBotManager(timeout=5.0)

        json_obj = {"backends": []}

        await manager.load_from_json(json_obj)

        assert len(manager._backends) == 0
        assert len(manager._clients) == 0


class TestChatBotManagerLoadFromFile:
    """Tests for load_from_file method."""

    @pytest.mark.asyncio
    async def test_load_from_file(self, tmp_path):
        """Test loading from JSON file."""
        manager = ChatBotManager(timeout=5.0)

        # Create test JSON file
        json_file = tmp_path / "backends.json"
        json_file.write_text('{"backends": [{"name": "file-backend", "url": "http://file:8000"}]}')

        mock_response = {"data": [{"id": "file-model"}]}
        mock_factory = _make_http_client_mock(return_value=mock_response)

        with patch(
            "peteos.chatbot.manager.HTTPClient",
            side_effect=mock_factory,
        ):
            await manager.load_from_file(str(json_file))

        assert "file-backend" in manager._backends
        assert manager._backends["file-backend"].api_type == "openai"
        assert "file-model" in manager._backends["file-backend"].models


class AnyChatBot:
    """Helper class for matching any ChatBot instance in tests."""

    def __eq__(self, other):
        return hasattr(other, '__class__') and 'ChatBot' in other.__class__.__name__
