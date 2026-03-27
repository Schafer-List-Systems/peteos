"""Tests for ChatBotResponse classes."""

import pytest
from peteos.chatbotresponse import OpenAIChatBotResponse, AnthropicChatBotResponse


class TestOpenAIChatBotResponse:
    """Tests for OpenAIChatBotResponse."""

    @pytest.mark.asyncio
    async def test_openai_streaming_response(self):
        """Test streaming OpenAI response with incremental content."""
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
            yield 'data: {"choices": [{"delta": {"content": " World"}}]}'
            yield "[DONE]"

        response = OpenAIChatBotResponse(mock_stream())
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Verify streaming accumulation
        assert len(accumulated) == 2
        assert accumulated[0] == "Hello"
        assert accumulated[1] == "Hello World"

        # Verify final content
        assert response.text_content == "Hello World"
        assert response.thinking_content == ""

    @pytest.mark.asyncio
    async def test_openai_with_thinking_content(self):
        """Test OpenAI response with thinking content."""
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"thinking": "Let me think...", "content": ""}}]}'
            yield 'data: {"choices": [{"delta": {"thinking": "", "content": "Hello"}}]}'
            yield 'data: {"choices": [{"delta": {"thinking": "", "content": " World"}}]}'
            yield "[DONE]"

        response = OpenAIChatBotResponse(mock_stream())
        async for _ in response:
            pass

        # Verify both thinking and text content
        assert response.thinking_content == "Let me think..."
        assert response.text_content == "Hello World"

    @pytest.mark.asyncio
    async def test_openai_with_reasoning(self):
        """Test OpenAI response with reasoning (Qwen-style)."""
        async def mock_stream():
            # Reasoning streamed incrementally like content
            yield 'data: {"choices": [{"delta": {"reasoning": "Thinking", "content": ""}}]}'
            yield 'data: {"choices": [{"delta": {"reasoning": " step by step", "content": ""}}]}'
            yield 'data: {"choices": [{"delta": {"reasoning": "", "content": "Hello"}}]}'
            yield 'data: {"choices": [{"delta": {"reasoning": "", "content": " World"}}]}'
            yield "[DONE]"

        response = OpenAIChatBotResponse(mock_stream())
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Verify streaming accumulation - reasoning yields empty text chunks
        assert len(accumulated) == 4
        assert accumulated[0] == ""  # First reasoning chunk
        assert accumulated[1] == ""  # Second reasoning chunk
        assert accumulated[2] == "Hello"
        assert accumulated[3] == "Hello World"

        # Verify both reasoning and text content extracted
        assert response.thinking_content == "Thinking step by step"
        assert response.text_content == "Hello World"

    @pytest.mark.asyncio
    async def test_openai_non_streaming(self):
        """Test non-streaming OpenAI response (all content in one event)."""
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"content": "Complete response"}}]}'
            yield "[DONE]"

        response = OpenAIChatBotResponse(mock_stream())
        # Collect all chunks
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Non-streaming yields all content at once
        assert len(accumulated) == 1
        assert accumulated[0] == "Complete response"
        assert response.text_content == "Complete response"


class TestAnthropicChatBotResponse:
    """Tests for AnthropicChatBotResponse."""

    @pytest.mark.asyncio
    async def test_anthropic_streaming_response(self):
        """Test streaming Anthropic response with reasoning."""
        async def mock_stream():
            # Content block start with reasoning
            yield 'data: {"type": "content_block_start", "content_block": {"type": "text", "reasoning": "Thinking step by step..."}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " World"}}'
            yield 'data: {"type": "content_block_stop", "content_block": {"type": "text"}}'
            yield "[DONE]"

        response = AnthropicChatBotResponse(mock_stream())
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Verify streaming accumulation - yields after every SSE event
        assert len(accumulated) >= 3
        # Event 1: reasoning event adds to thinking but no text
        assert accumulated[0] == ""
        # Event 2: "Hello" text
        assert accumulated[1] == "Hello"
        # Event 3: " World" text
        assert "Hello World" in accumulated[2] or accumulated[2] == "Hello World"

        # Verify reasoning content is extracted
        assert response.thinking_content == "Thinking step by step..."
        assert response.text_content == "Hello World"

    @pytest.mark.asyncio
    async def test_anthropic_message_start(self):
        """Test Anthropic message_start event."""
        async def mock_stream():
            yield 'data: {"type": "message_start", "message": {"content": [{"type": "text", "text": "Hello"}], "reasoning": "Initial reasoning"}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " World"}}'
            yield "[DONE]"

        response = AnthropicChatBotResponse(mock_stream())
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Verify message_start is processed
        assert accumulated[0] == "Hello"
        assert accumulated[1] == "Hello World"
        assert response.thinking_content == "Initial reasoning"
        assert response.text_content == "Hello World"

    @pytest.mark.asyncio
    async def test_anthropic_compatible_thinking(self):
        """Test Anthropic-compatible format with thinking/thinking_delta."""
        async def mock_stream():
            yield 'data: {"type": "content_block_start", "content_block": {"type": "thinking", "thinking": ""}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "Thinking"}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": " Process"}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": ":", "text": ""}}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}'
            yield "[DONE]"

        response = AnthropicChatBotResponse(mock_stream())
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Verify thinking content is extracted from thinking_delta
        assert response.thinking_content == "Thinking Process:"
        assert "Hello" in response.text_content

    @pytest.mark.asyncio
    async def test_anthropic_non_streaming(self):
        """Test non-streaming Anthropic response (all content in one event)."""
        async def mock_stream():
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Complete response"}}'
            yield "[DONE]"

        response = AnthropicChatBotResponse(mock_stream())
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        # Non-streaming yields all content at once
        assert len(accumulated) == 1
        assert accumulated[0] == "Complete response"
        assert response.text_content == "Complete response"


class TestChatBotResponseProperties:
    """Tests for ChatBotResponse common properties."""

    @pytest.mark.asyncio
    async def test_thinking_content_isolation(self):
        """Test that thinking content doesn't mix with text content."""
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"thinking": "Reasoning", "content": "Text"}}]}'
            yield "[DONE]"

        response = OpenAIChatBotResponse(mock_stream())
        async for _ in response:
            pass

        assert response.thinking_content == "Reasoning"
        assert response.text_content == "Text"
        assert "Reasoning" not in response.text_content
        assert "Text" not in response.thinking_content
