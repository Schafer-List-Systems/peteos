"""Unit tests for Anthropic ChatBot implementation."""

import json
import pytest
from peteos.chatbot import AnthropicChatBotResponse
from peteos.chatbot import AnthropicChatBot, ChatHistory, Message, ContentPart
from peteos.chatbot.chatbotconfig import ChatBotConfig
from peteos.chatbot.httpclient import HTTPClient


class TestAnthropicChatBotResponse:
    """Tests for AnthropicChatBot response accumulation."""

    @pytest.mark.asyncio
    async def test_streaming_text_only(self):
        """Anthropic streaming text produces content array with text item."""
        async def mock_stream():
            yield 'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}'
            yield 'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello World"}}'
            yield "[DONE]"

        response = AnthropicChatBotResponse(mock_stream(), AnthropicChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        assert "role" in response.data
        assert "content" in response.data
        assert len(response.data["content"]) >= 1
        assert response.data["content"][0]["type"] == "text"
        assert response.data["content"][0]["content"] == "Hello World"
        assert "text" not in response.data

    @pytest.mark.asyncio
    async def test_streaming_reasoning_and_text(self):
        """Anthropic streaming with reasoning (thinking) and text blocks."""
        async def mock_stream():
            yield 'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "Thinking..."}}'
            yield 'data: {"type": "content_block_start", "index": 1, "content_block": {"type": "text"}}'
            yield 'data: {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hello"}}'
            yield "[DONE]"

        response = AnthropicChatBotResponse(mock_stream(), AnthropicChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        assert "content" in response.data
        assert len(response.data["content"]) >= 2
        assert response.data["content"][0]["type"] == "thinking"
        assert response.data["content"][0]["content"] == "Thinking..."
        assert response.data["content"][1]["type"] == "text"
        assert response.data["content"][1]["content"] == "Hello"
        # text field should NOT be set (unified format only uses content)
        assert "text" not in response.data

    @pytest.mark.asyncio
    async def test_streaming_tool_calls(self):
        """Anthropic streaming with tool call produces content array with tool_use item."""
        import json as _json
        partial_json_1 = _json.dumps({"query": "1"})
        # Build SSE data with properly escaped nested JSON
        start_event = _json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "tc_1", "name": "calculator"}})
        delta1_event = _json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": partial_json_1}})
        delta2_event = _json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "1"}})
        async def mock_stream():
            yield f'data: {start_event}'
            yield f'data: {delta1_event}'
            yield f'data: {delta2_event}'
            yield "[DONE]"

        response = AnthropicChatBotResponse(mock_stream(), AnthropicChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        assert "content" in response.data
        assert len(response.data["content"]) >= 1
        assert response.data["content"][0]["type"] == "tool_use"
        assert response.data["content"][0]["id"] == "tc_1"
        assert response.data["content"][0]["name"] == "calculator"
        assert "query" in response.data["content"][0]["arguments"]

    @pytest.mark.asyncio
    async def test_streaming_with_message_start(self):
        """Anthropic streaming with message_start event preserves role."""
        async def mock_stream():
            yield 'data: {"type": "message_start", "message": {"role": "assistant"}}'
            yield 'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}'
            yield 'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}'
            yield "[DONE]"

        response = AnthropicChatBotResponse(mock_stream(), AnthropicChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        assert response.data["role"] == "assistant"
        assert response.data["content"][0]["type"] == "text"
        assert response.data["content"][0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_non_streaming_text_content(self):
        """Anthropic non-streaming from_json produces content array."""
        mock_data = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Test response"}
            ],
            "stop_reason": "end_turn"
        }
        response = AnthropicChatBotResponse.from_json(mock_data, AnthropicChatBot.RESPONSE_TRANSLATIONS)

        assert response.data["role"] == "assistant"
        assert "content" in response.data
        assert response.data["content"][0]["type"] == "text"
        assert response.data["content"][0]["content"] == "Test response"
        assert response.data["stop_reason"] == "end_turn"
        assert "text" not in response.data

    @pytest.mark.asyncio
    async def test_non_streaming_thinking_content(self):
        """Anthropic non-streaming with thinking content."""
        mock_data = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "Here is the answer"}
            ],
            "stop_reason": "end_turn"
        }
        response = AnthropicChatBotResponse.from_json(mock_data, AnthropicChatBot.RESPONSE_TRANSLATIONS)

        assert "content" in response.data
        assert response.data["content"][0]["type"] == "thinking"
        assert response.data["content"][0]["content"] == "Let me think..."
        assert response.data["content"][1]["type"] == "text"
        assert response.data["content"][1]["content"] == "Here is the answer"

    @pytest.mark.asyncio
    async def test_non_streaming_tool_use(self):
        """Anthropic non-streaming with tool use converts input dict to arguments JSON."""
        mock_data = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tc_1",
                    "name": "calculator",
                    "input": {"query": "1+1"}
                }
            ],
            "stop_reason": "tool_use"
        }
        response = AnthropicChatBotResponse.from_json(mock_data, AnthropicChatBot.RESPONSE_TRANSLATIONS)

        assert response.data["content"][0]["type"] == "tool_use"
        assert response.data["content"][0]["id"] == "tc_1"
        assert response.data["content"][0]["name"] == "calculator"
        assert response.data["content"][0]["arguments"] == json.dumps({"query": "1+1"})

    @pytest.mark.asyncio
    async def test_role_default_to_assistant(self):
        """Anthropic response defaults role to assistant when missing."""
        async def mock_stream():
            yield 'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}'
            yield 'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}'
            yield "[DONE]"

        response = AnthropicChatBotResponse(mock_stream(), AnthropicChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        assert response.data["role"] == "assistant"


class TestAnthropicRequestTranslation:
    """Tests for AnthropicChatBot request building."""

    @pytest.mark.asyncio
    async def test_build_body_with_text_message(self):
        """Text content parts should be sent as Anthropic content array."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = AnthropicChatBot(http_client, config)
        chat_history = ChatHistory()
        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Hello")],
        ))

        body = chatbot._build_body(chat_history)
        messages = body.get("messages", [])
        assert len(messages) >= 1
        user_msg = messages[0]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][0]["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_build_body_with_reasoning_message(self):
        """Reasoning content parts should be sent as thinking in Anthropic."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = AnthropicChatBot(http_client, config)
        chat_history = ChatHistory()
        chat_history.append_message(Message(
            role="assistant",
            content=[
                ContentPart(part_type="reasoning", reasoning="Let me think..."),
                ContentPart(part_type="text", text="Here is the answer"),
            ]
        ))

        body = chatbot._build_body(chat_history)
        messages = body.get("messages", [])
        assistant_msg = messages[0]
        assert assistant_msg["role"] == "assistant"
        assert isinstance(assistant_msg["content"], list)
        thinking = [c for c in assistant_msg["content"] if c.get("type") == "thinking"]
        text = [c for c in assistant_msg["content"] if c.get("type") == "text"]
        assert len(thinking) == 1
        assert thinking[0]["thinking"] == "Let me think..."
        assert len(text) == 1
        assert text[0]["text"] == "Here is the answer"