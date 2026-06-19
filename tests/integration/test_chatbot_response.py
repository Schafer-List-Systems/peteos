"""Integration tests for ChatBot response format.

This tests that ChatBot implementations correctly translate API responses
to the uniform ChatBotResponse format with required fields.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from aiohttp import web

from peteos.chatbot import OpenAIChatBot, AnthropicChatBot, GenericChatBotResponse, ChatHistory
from peteos.chatbot.httpclient import HTTPClient
from peteos.chatbot.chatbotconfig import ChatBotConfig


class TestOpenAIChatBotResponse:
    """Tests for OpenAIChatBot response handling."""

    @pytest.mark.asyncio
    async def test_openai_response_has_role_field(self):
        """Test that OpenAIChatBot adds role to response.data."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = OpenAIChatBot(http_client, config)
        chat_history = ChatHistory()

        # Mock SSE response for OpenAI format
        mock_sse_data = [
            'data: {"choices": [{"delta": {"role": "assistant", "content": "Hello"}}]}\n',
            'data: {"choices": [{"delta": {"content": " World"}}]}\n',
            'data: [DONE]\n'
        ]

        async def mock_stream():
            for line in mock_sse_data:
                yield line

        with patch.object(http_client, 'stream_post', return_value=mock_stream()):
            response = await chatbot.send_message(chat_history=chat_history, streaming=True)
            async for _ in response:
                pass

            # Verify role is present in response data
            assert "role" in response.data, f"OpenAIChatBot response missing 'role' field: {response.data.keys()}"
            assert response.data["role"] == "assistant"
            # Contract: content array with text item
            assert "content" in response.data
            assert response.data["content"][0]["type"] == "text"
            assert response.data["content"][0]["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_openai_response_non_streaming_has_role(self):
        """Test that non-streaming OpenAI response also has role."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = OpenAIChatBot(http_client, config)
        chat_history = ChatHistory()

        # Mock JSON response with role in message - needs to match OpenAI format
        mock_json = {
            "choices": [{
                "message": {"role": "assistant", "content": "Test response"}
            }]
        }

        with patch.object(http_client, 'post', return_value=mock_json):
            response = await chatbot.send_message(chat_history=chat_history, streaming=False)

            # Need to iterate through response to populate response.data
            async for _ in response:
                pass

            # Role and content should be present per the unified contract
            assert "role" in response.data
            assert response.data["role"] == "assistant"
            assert "content" in response.data
            assert response.data["content"][0]["type"] == "text"
            assert response.data["content"][0]["content"] == "Test response"


class TestAnthropicChatBotResponse:
    """Tests for AnthropicChatBot response handling."""

    @pytest.mark.asyncio
    async def test_anthropic_response_has_role_field(self):
        """Test that AnthropicChatBot adds role to response.data."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = AnthropicChatBot(http_client, config)
        chat_history = ChatHistory()

        # Mock SSE response for Anthropic format (includes content_block_start to set type)
        mock_sse_data = [
            'data: {"type": "message_start", "message": {"role": "assistant"}}\n',
            'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}\n',
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}\n',
            'data: [DONE]\n'
        ]

        async def mock_stream():
            for line in mock_sse_data:
                yield line

        with patch.object(http_client, 'stream_post', return_value=mock_stream()):
            response = await chatbot.send_message(chat_history=chat_history, streaming=True)
            async for _ in response:
                pass

            # Verify role and content per unified contract
            assert "role" in response.data
            assert response.data["role"] == "assistant"
            assert "content" in response.data
            assert response.data["content"][0]["type"] == "text"
            assert response.data["content"][0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_anthropic_response_has_role_field_2(self):
        """Test Anthropic with simple content_block_delta (no index)."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = AnthropicChatBot(http_client, config)
        chat_history = ChatHistory()

        mock_sse_data = [
            'data: {"type": "message_start", "message": {"role": "assistant"}}\n',
            'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}\n',
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "World"}}\n',
            'data: [DONE]\n'
        ]

        async def mock_stream():
            for line in mock_sse_data:
                yield line

        with patch.object(http_client, 'stream_post', return_value=mock_stream()):
            response = await chatbot.send_message(chat_history=chat_history, streaming=True)
            async for _ in response:
                pass

            # Verify role and content per unified contract
            assert "role" in response.data
            assert response.data["role"] == "assistant"
            assert "content" in response.data
            assert response.data["content"][0]["type"] == "text"
            assert response.data["content"][0]["content"] == "World"


class TestChatBotResponseIntegration:
    """Integration tests that ChatBot responses work with ExecutionEnvironment."""

    @pytest.mark.asyncio
    async def test_response_valid_for_execution_environment(self):
        """Test that response format satisfies ExecutionEnvironment requirements."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = OpenAIChatBot(http_client, config)
        chat_history = ChatHistory()

        # Mock response
        mock_sse_data = [
            'data: {"choices": [{"delta": {"role": "assistant", "content": "Hello"}}]}\n',
            'data: [DONE]\n'
        ]

        async def mock_stream():
            for line in mock_sse_data:
                yield line

        with patch.object(http_client, 'stream_post', return_value=mock_stream()):
            response = await chatbot.send_message(chat_history=chat_history, streaming=True)
            async for _ in response:
                pass

            # Unified contract: both role and content must be present
            assert "role" in response.data, \
                f"ExecutionEnvironment requires 'role' in response.data, but got: {response.data.keys()}"
            assert response.data["role"] in ("assistant", "user", "system"), \
                f"Invalid role value: {response.data['role']}"
            assert "content" in response.data, \
                f"ExecutionEnvironment requires 'content' array in response.data"
            assert response.data["content"][0]["type"] == "text"
            assert response.data["content"][0]["content"] == "Hello"
