"""Unit tests for ChatBotResponse and GenericChatBotResponse."""

import json
import pytest

from peteos.chatbot.chatbotresponse import ChatBotResponse, GenericChatBotResponse


class TestChatBotResponse:
    """Tests for the base ChatBotResponse class."""

    @pytest.mark.asyncio
    async def test_data_property_empty_initially(self):
        """Response starts with empty data dict."""
        async def mock_stream():
            yield "[DONE]"

        response = ChatBotResponse(mock_stream())
        assert response.data == {}

    @pytest.mark.asyncio
    async def test_getitem_returns_none_for_missing_key(self):
        """__getitem__ returns None for non-existent key."""
        async def mock_stream():
            yield "[DONE]"

        response = ChatBotResponse(mock_stream())
        assert response["nonexistent"] is None

    @pytest.mark.asyncio
    async def test_contains_false_for_missing_key(self):
        """__contains__ returns False for non-existent key."""
        async def mock_stream():
            yield "[DONE]"

        response = ChatBotResponse(mock_stream())
        assert "missing" not in response

    @pytest.mark.asyncio
    async def test_contains_true_for_existing_key(self):
        """__contains__ returns True for existing key."""
        async def mock_stream():
            yield "[DONE]"

        response = ChatBotResponse(mock_stream())
        response._data["key"] = "value"
        assert "key" in response


class TestGenericChatBotResponse:
    """Tests for GenericChatBotResponse with path-based translations."""

    @pytest.mark.asyncio
    async def test_accumulates_text_chunks(self):
        """Generic response accumulates text content across SSE events."""
        translations = {"path": "content"}
        async def mock_stream():
            yield 'data: {"path": "Hello"}'
            yield 'data: {"path": " World"}'
            yield "[DONE]"

        response = GenericChatBotResponse(mock_stream(), translations)
        async for _ in response:
            pass

        assert response.data["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_from_json_creates_stream(self):
        """from_json converts a JSON response to a stream."""
        data = {"choices": [{"delta": {"content": "test"}}]}
        translations = {}
        response = GenericChatBotResponse.from_json(data, translations)
        assert isinstance(response, GenericChatBotResponse)

    @pytest.mark.asyncio
    async def test_handles_invalid_json_line(self):
        """Invalid JSON lines are logged but don't crash."""
        translations = {}
        async def mock_stream():
            yield 'data: not valid json'
            yield "[DONE]"

        response = GenericChatBotResponse(mock_stream(), translations)
        async for _ in response:
            pass  # Should not raise

    @pytest.mark.asyncio
    async def test_error_in_response(self):
        """Error dicts are stored in response.data."""
        translations = {}
        async def mock_stream():
            yield '{"error": "something broke"}'
            yield "[DONE]"

        response = GenericChatBotResponse(mock_stream(), translations)
        async for _ in response:
            pass

        assert response.data.get("error") == "something broke"

    @pytest.mark.asyncio
    async def test_event_marker_ignored(self):
        """event: markers are skipped during parsing."""
        translations = {}
        async def mock_stream():
            yield 'event: message_start'
            yield 'data: {"path": "Hello"}'
            yield "[DONE]"

        response = GenericChatBotResponse(mock_stream(), translations)
        async for _ in response:
            pass

        assert "path" not in response.data  # No data accumulated from event marker

    @pytest.mark.asyncio
    async def test_multiple_keys_from_one_event(self):
        """One SSE event can yield multiple key-value pairs."""
        translations = {"a": "key_a", "b": "key_b"}
        async def mock_stream():
            yield 'data: {"a": "val_a", "b": "val_b"}'
            yield "[DONE]"

        response = GenericChatBotResponse(mock_stream(), translations)
        chunks = []
        async for chunk in response:
            chunks.append(chunk)

        keys = [c[0] for c in chunks]
        assert "key_a" in keys
        assert "key_b" in keys

    @pytest.mark.asyncio
    async def test_none_chunks_dropped(self):
        """None values from translation are not accumulated."""
        translations = {"a": "key_a", "b": "key_b"}
        async def mock_stream():
            yield 'data: {"a": "val_a", "b": null}'
            yield "[DONE]"

        response = GenericChatBotResponse(mock_stream(), translations)
        async for _ in response:
            pass

        assert "key_a" in response.data
        assert "key_b" not in response.data
