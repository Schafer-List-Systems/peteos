"""Tests for ChatBotResponse classes."""

import pytest
from peteos.chatbot import (
    ChatBotResponse,
    GenericChatBotResponse,
    AnthropicChatBotResponse,
)
from peteos.chatbot import OpenAIChatBot, AnthropicChatBot


class TestGenericChatBotResponseOpenAI:
    """Tests for GenericChatBotResponse with OpenAI translation configuration."""

    @pytest.mark.asyncio
    async def test_openai_streaming_response(self):
        """Test streaming OpenAI response with incremental content."""
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
            yield 'data: {"choices": [{"delta": {"content": " World"}}]}'
            yield "[DONE]"

        response = GenericChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        assert len(accumulated) == 2
        assert accumulated[0] == ("text", "Hello")
        assert accumulated[1] == ("text", " World")

        assert response.data["text"] == "Hello World"
        assert response.data.get("reasoning", "") == ""

    @pytest.mark.asyncio
    async def test_openai_with_thinking_content(self):
        """Test OpenAI response with thinking content."""
        async def mock_stream():
            # Reasoning first (separate event from content per API behavior)
            yield 'data: {"choices": [{"delta": {"thinking": "Let me think..."}}]}'
            yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
            yield 'data: {"choices": [{"delta": {"content": " World"}}]}'
            yield "[DONE]"

        response = GenericChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Each SSE yields (key, delta_chunk) - delta not accumulated
        assert len(accumulated) == 3
        assert accumulated[0] == ("reasoning", "Let me think...")
        assert accumulated[1] == ("text", "Hello")
        assert accumulated[2] == ("text", " World")

        # Verify both fields extracted
        assert response.data["reasoning"] == "Let me think..."
        assert response.data["text"] == "Hello World"

    @pytest.mark.asyncio
    async def test_openai_with_reasoning(self):
        """Test OpenAI response with reasoning (Qwen-style)."""
        async def mock_stream():
            # Reasoning streamed incrementally (one field per event)
            yield 'data: {"choices": [{"delta": {"reasoning": "Thinking"}}]}'
            yield 'data: {"choices": [{"delta": {"reasoning": " step by step"}}]}'
            yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
            yield 'data: {"choices": [{"delta": {"content": " World"}}]}'
            yield "[DONE]"

        response = GenericChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Each SSE yields (key, delta_chunk) - delta not accumulated
        assert len(accumulated) == 4
        assert accumulated[0] == ("reasoning", "Thinking")
        assert accumulated[1] == ("reasoning", " step by step")
        assert accumulated[2] == ("text", "Hello")
        assert accumulated[3] == ("text", " World")

        # Verify both reasoning and text content extracted
        assert response.data["reasoning"] == "Thinking step by step"
        assert response.data["text"] == "Hello World"

    @pytest.mark.asyncio
    async def test_openai_non_streaming(self):
        """Test non-streaming OpenAI response (all content in one event)."""
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"content": "Complete response"}}]}'
            yield "[DONE]"

        response = GenericChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        # Collect all chunks
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Non-streaming yields all content at once
        assert len(accumulated) == 1
        assert accumulated[0] == ("text", "Complete response")
        assert response.data["text"] == "Complete response"


class TestGenericChatBotResponseAnthropic:
    """Tests for GenericChatBotResponse with Anthropic translation configuration."""

    @pytest.mark.asyncio
    async def test_anthropic_streaming_response(self):
        """Test streaming Anthropic response with reasoning."""
        async def mock_stream():
            yield 'data: {"type": "content_block_start", "content_block": {"type": "text", "reasoning": "Thinking step by step..."}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " World"}}'
            yield 'data: {"type": "content_block_stop", "content_block": {"type": "text"}}'
            yield "[DONE]"

        response = GenericChatBotResponse(mock_stream(), AnthropicChatBot.RESPONSE_TRANSLATIONS)
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Check that all expected fields are present (order may vary due to dict ordering)
        assert len(accumulated) >= 3
        accumulated_keys = [key for key, _ in accumulated]

        # Verify reasoning and text fields were extracted
        assert "reasoning" in accumulated_keys
        assert "text" in accumulated_keys

        assert response.data["reasoning"] == "Thinking step by step..."
        assert response.data["text"] == "Hello World"

    @pytest.mark.asyncio
    async def test_anthropic_message_start(self):
        """Test Anthropic message_start event."""
        async def mock_stream():
            # message_start has nested content array, not inline text/thinking
            yield 'data: {"type": "message_start", "message": {"role": "assistant", "content": []}}'
            yield 'data: {"type": "content_block_start", "content_block": {"type": "thinking", "thinking": ""}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "Initial reasoning"}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " World"}}'
            yield "[DONE]"

        response = GenericChatBotResponse(mock_stream(), AnthropicChatBot.RESPONSE_TRANSLATIONS)
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Verify reasoning and text extracted from content_block events
        assert "Initial reasoning" in response.data["reasoning"]
        assert response.data["text"] == "Hello World"

    @pytest.mark.asyncio
    async def test_anthropic_compatible_thinking(self):
        """Test Anthropic-compatible format with thinking/thinking_delta."""
        async def mock_stream():
            yield 'data: {"type": "content_block_start", "content_block": {"type": "thinking", "thinking": ""}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "Thinking"}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": " Process"}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": ":"}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}'
            yield "[DONE]"

        response = GenericChatBotResponse(mock_stream(), AnthropicChatBot.RESPONSE_TRANSLATIONS)
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Verify reasoning content is extracted from thinking_delta
        assert response.data["reasoning"] == "Thinking Process:"
        assert response.data["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_anthropic_non_streaming(self):
        """Test non-streaming Anthropic response (all content in one event)."""
        async def mock_stream():
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Complete response"}}'
            yield "[DONE]"

        response = GenericChatBotResponse(mock_stream(), AnthropicChatBot.RESPONSE_TRANSLATIONS)
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Non-streaming yields all content at once
        assert len(accumulated) == 1
        assert accumulated[0] == ("text", "Complete response")
        assert response.data["text"] == "Complete response"


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
        assert response.data["text"] == "Hello World"

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
        assert response.data["text"] == "Hello World"


class TestChatBotResponseProperties:
    """Tests for ChatBotResponse common properties."""

    @pytest.mark.asyncio
    async def test_reasoning_content_isolation(self):
        """Test that reasoning content doesn't mix with text content."""
        async def mock_stream():
            # Each field in separate event (API behavior)
            yield 'data: {"choices": [{"delta": {"thinking": "Reasoning"}}]}'
            yield 'data: {"choices": [{"delta": {"content": "Text"}}]}'
            yield "[DONE]"

        response = GenericChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        assert response.data["reasoning"] == "Reasoning"
        assert response.data["text"] == "Text"
        assert "Reasoning" not in response.data["text"]
        assert "Text" not in response.data["reasoning"]

    @pytest.mark.asyncio
    async def test_response_data_access(self):
        """Test dict-like access to response data."""
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
            yield "[DONE]"

        response = GenericChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        # Test __getitem__
        assert response["text"] == "Hello"

        # Test __contains__
        assert "text" in response
        assert "reasoning" not in response

        # Test data property
        assert response.data["text"] == "Hello"
