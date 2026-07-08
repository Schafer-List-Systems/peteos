"""Tests for ChatBotResponse classes."""

import pytest
from peteos.chatbot import (
    ChatBotResponse,
    AnthropicChatBotResponse,
    GenericChatBotResponse,
)
from peteos.chatbot import OpenAIChatBot, AnthropicChatBot
from peteos.chatbot.openaichatbot import OpenAIChatBotResponse


class TestGenericChatBotResponseOpenAI:
    """Tests for GenericChatBotResponse with OpenAI translation configuration."""

    @pytest.mark.asyncio
    async def test_openai_streaming_response(self):
        """Test streaming OpenAI response with incremental content."""
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"role": "assistant", "content": "Hello"}}]}'
            yield 'data: {"choices": [{"delta": {"content": " World"}}]}'
            yield "[DONE]"

        response = OpenAIChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        assert response.data["role"] == "assistant"
        assert "content" in response.data
        assert response.data["content"][0]["type"] == "text"
        assert response.data["content"][0]["content"] == "Hello World"
        assert "reasoning" not in response.data

    @pytest.mark.asyncio
    async def test_openai_with_thinking_content(self):
        """Test OpenAI response with reasoning/thinking content."""
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"reasoning": "Let me think..."}}]}'
            yield 'data: {"choices": [{"delta": {"role": "assistant", "content": "Hello"}}]}'
            yield 'data: {"choices": [{"delta": {"content": " World"}}]}'
            yield "[DONE]"

        response = OpenAIChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        # Verify reasoning → thinking block in content array
        assert "content" in response.data
        content = response.data["content"]
        assert content[0]["type"] == "thinking"
        assert content[0]["content"] == "Let me think..."
        assert content[1]["type"] == "text"
        assert content[1]["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_openai_with_reasoning(self):
        """Test OpenAI response with incremental reasoning (Qwen-style)."""
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"reasoning": "Thinking"}}]}'
            yield 'data: {"choices": [{"delta": {"reasoning": " step by step"}}]}'
            yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
            yield 'data: {"choices": [{"delta": {"content": " World"}}]}'
            yield "[DONE]"

        response = OpenAIChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        # Verify reasoning accumulated as thinking block
        assert "content" in response.data
        content = response.data["content"]
        assert content[0]["type"] == "thinking"
        assert content[0]["content"] == "Thinking step by step"
        assert content[1]["type"] == "text"
        assert content[1]["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_openai_non_streaming(self):
        """Test non-streaming OpenAI response (all content in one event)."""
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"content": "Complete response"}}]}'
            yield "[DONE]"

        response = OpenAIChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        assert "content" in response.data
        assert response.data["content"][0]["type"] == "text"
        assert response.data["content"][0]["content"] == "Complete response"


class TestGenericChatBotResponseAnthropic:
    """Tests for GenericChatBotResponse with Anthropic translation configuration."""

    @pytest.mark.asyncio
    async def test_anthropic_streaming_response(self):
        """Test streaming Anthropic response with reasoning and text blocks."""
        async def mock_stream():
            yield 'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "Thinking step by step..."}}'
            yield 'data: {"type": "content_block_start", "index": 1, "content_block": {"type": "text"}}'
            yield 'data: {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hello"}}'
            yield 'data: {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": " World"}}'
            yield 'data: {"type": "content_block_stop", "index": 1}'
            yield "[DONE]"

        response = AnthropicChatBotResponse(mock_stream(), AnthropicChatBot.RESPONSE_TRANSLATIONS)
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Anthropic response uses content array format
        assert "content" in response.data
        content = response.data["content"]
        assert len(content) >= 1
        # First content item is thinking block
        assert content[0]["type"] == "thinking"
        assert content[0]["content"] == "Thinking step by step..."
        # Second content item is text block
        assert content[1]["type"] == "text"
        assert content[1]["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_anthropic_message_start(self):
        """Test Anthropic message_start event."""
        async def mock_stream():
            yield 'data: {"type": "message_start", "message": {"role": "assistant", "content": []}}'
            yield 'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}'
            yield 'data: {"type": "content_block_start", "index": 1, "content_block": {"type": "text"}}'
            yield 'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Initial reasoning"}}'
            yield 'data: {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hello"}}'
            yield 'data: {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": " World"}}'
            yield "[DONE]"

        response = AnthropicChatBotResponse(mock_stream(), AnthropicChatBot.RESPONSE_TRANSLATIONS)
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Anthropic response uses content array format
        assert "content" in response.data
        content = response.data["content"]
        assert content[0]["type"] == "thinking"
        assert "Initial reasoning" in content[0]["content"]
        assert content[1]["type"] == "text"
        assert content[1]["content"] == "Hello World"
        assert "role" in response.data
        assert response.data["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_anthropic_compatible_thinking(self):
        """Test Anthropic-compatible format with thinking/thinking_delta."""
        async def mock_stream():
            yield 'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}'
            yield 'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Thinking"}}'
            yield 'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": " Process"}}'
            yield 'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": ":"}}'
            yield 'data: {"type": "content_block_start", "index": 1, "content_block": {"type": "text"}}'
            yield 'data: {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hello"}}'
            yield "[DONE]"

        response = AnthropicChatBotResponse(mock_stream(), AnthropicChatBot.RESPONSE_TRANSLATIONS)
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Anthropic response uses content array format
        assert "content" in response.data
        content = response.data["content"]
        assert content[0]["type"] == "thinking"
        assert content[0]["content"] == "Thinking Process:"
        assert content[1]["type"] == "text"
        assert content[1]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_anthropic_non_streaming(self):
        """Test non-streaming Anthropic response (all content in one event)."""
        async def mock_stream():
            yield 'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}'
            yield 'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Complete response"}}'
            yield "[DONE]"

        response = AnthropicChatBotResponse(mock_stream(), AnthropicChatBot.RESPONSE_TRANSLATIONS)
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Non-streaming yields all content at once
        assert len(accumulated) >= 1
        assert "content" in response.data
        assert response.data["content"][0]["type"] == "text"
        assert response.data["content"][0]["content"] == "Complete response"


class TestAnthropicChatBotResponseRole:
    """Tests for AnthropicChatBotResponse role defaulting."""

    @pytest.mark.asyncio
    async def test_anthropic_message_start_no_role(self):
        """Test Anthropic message_start with missing role field (broken backend)."""
        async def mock_stream():
            yield 'data: {"type": "message_start", "message": {"content": []}}'
            yield 'data: {"type": "content_block_start", "content_block": {"type": "text", "text": ""}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " World"}}'
            yield "[DONE]"

        response = AnthropicChatBotResponse(mock_stream(), AnthropicChatBot.RESPONSE_TRANSLATIONS)
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        assert response.data["role"] == "assistant"
        assert "content" in response.data
        assert response.data["content"][0]["type"] == "text"
        assert response.data["content"][0]["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_anthropic_message_start_with_role(self):
        """Test Anthropic message_start with role field present (bot, not assistant)."""
        async def mock_stream():
            yield 'data: {"type": "message_start", "message": {"role": "bot", "content": []}}'
            yield 'data: {"type": "content_block_start", "content_block": {"type": "text", "text": ""}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " World"}}'
            yield "[DONE]"

        response = AnthropicChatBotResponse(mock_stream(), AnthropicChatBot.RESPONSE_TRANSLATIONS)
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Role extracted from message_start.message.role
        assert response.data["role"] == "bot"
        assert "content" in response.data
        assert response.data["content"][0]["type"] == "text"
        assert response.data["content"][0]["content"] == "Hello World"


class TestChatBotResponseProperties:
    """Tests for ChatBotResponse common properties."""

    @pytest.mark.asyncio
    async def test_reasoning_content_isolation(self):
        """Test that reasoning content doesn't mix with text content."""
        async def mock_stream():
            # Each field in separate event (API behavior)
            yield 'data: {"choices": [{"delta": {"reasoning": "Reasoning"}}]}'
            yield 'data: {"choices": [{"delta": {"content": "Text"}}]}'
            yield "[DONE]"

        response = OpenAIChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        assert "content" in response.data
        content = response.data["content"]
        # Reasoning should be a thinking block, not mixed with text
        assert content[0]["type"] == "thinking"
        assert content[0]["content"] == "Reasoning"
        assert content[1]["type"] == "text"
        assert content[1]["content"] == "Text"

    @pytest.mark.asyncio
    async def test_response_data_access(self):
        """Test dict-like access to response data."""
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"role": "assistant", "content": "Hello"}}]}'
            yield "[DONE]"

        response = OpenAIChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        # Test __getitem__
        assert response["role"] == "assistant"

        # Test __contains__
        assert "role" in response
        assert "text" not in response

        # Test data property
        assert "content" in response.data
        assert response.data["content"][0]["type"] == "text"
        assert response.data["content"][0]["content"] == "Hello"
