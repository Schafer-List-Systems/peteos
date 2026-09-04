"""Unit tests for OpenAI ChatBot implementation."""

import pytest

from peteos.utils import json
from peteos.chatbot import OpenAIChatBotResponse
from peteos.chatbot.chatbotconfig import ChatBotConfig
from peteos.chatbot.httpclient import HTTPClient
from peteos.chatbot.openaichatbot import OpenAIChatBot


class TestOpenAIChatBotResponse:
    """Tests for OpenAIChatBot response accumulation."""

    @pytest.mark.asyncio
    async def test_streaming_text_only(self):
        """OpenAI streaming text produces content array with text item."""
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"role": "assistant", "content": "Hello"}}]}'
            yield 'data: {"choices": [{"delta": {"content": " World"}}]}'
            yield "[DONE]"

        response = OpenAIChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        assert "role" in response.data
        assert response.data["role"] == "assistant"
        assert "content" in response.data
        assert len(response.data["content"]) >= 1
        assert response.data["content"][0]["type"] == "text"
        assert response.data["content"][0]["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_streaming_reasoning_plus_text(self):
        """OpenAI streaming with reasoning and text content."""
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"reasoning": "Let me think..."}}]}'
            yield 'data: {"choices": [{"delta": {"role": "assistant", "content": "Hello"}}]}'
            yield "[DONE]"

        response = OpenAIChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        assert response.data["role"] == "assistant"
        assert "content" in response.data
        assert len(response.data["content"]) == 2
        # Reasoning → thinking block (index 0)
        assert response.data["content"][0]["type"] == "thinking"
        assert response.data["content"][0]["content"] == "Let me think..."
        # Text (index 1)
        assert response.data["content"][1]["type"] == "text"
        assert response.data["content"][1]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_streaming_tool_calls(self):
        """OpenAI streaming with tool call produces content array with tool_use item."""
        first_delta = {
            "choices": [{
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 0,
                        "id": "tc_1",
                        "function": {
                            "name": "calculator",
                            "arguments": json.dumps({"query": "1+1"})
                        }
                    }]
                }
            }]
        }
        second_delta = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": "2"}
                    }]
                }
            }]
        }
        finish = {"choices": [{"delta": {"finish_reason": "tool_calls"}}]}
        async def mock_stream():
            yield f'data: {json.dumps(first_delta)}'
            yield f'data: {json.dumps(second_delta)}'
            yield f'data: {json.dumps(finish)}'
            yield "[DONE]"

        response = OpenAIChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        assert response.data["role"] == "assistant"
        assert "content" in response.data
        assert len(response.data["content"]) >= 1
        assert response.data["content"][0]["type"] == "tool_use"
        assert response.data["content"][0]["call_id"] == "tc_1"
        assert response.data["content"][0]["name"] == "calculator"
        assert response.data["content"][0]["arguments"] == '{"query": "1+1"}2'

    @pytest.mark.asyncio
    async def test_non_streaming_text_content(self):
        """OpenAI non-streaming from_json produces content array."""
        mock_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Test response"
                }
            }]
        }
        response = OpenAIChatBotResponse.from_json(mock_data, OpenAIChatBot.RESPONSE_TRANSLATIONS)

        assert response.data["role"] == "assistant"
        assert "content" in response.data
        assert response.data["content"][0]["type"] == "text"
        assert response.data["content"][0]["content"] == "Test response"
        # text field should NOT be set (unified format only uses content)
        assert "text" not in response.data

    @pytest.mark.asyncio
    async def test_non_streaming_content_as_list(self):
        """OpenAI non-streaming with content as array (vision format)."""
        mock_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hello world"},
                    ]
                }
            }]
        }
        response = OpenAIChatBotResponse.from_json(mock_data, OpenAIChatBot.RESPONSE_TRANSLATIONS)

        assert response.data["role"] == "assistant"
        assert "content" in response.data
        assert response.data["content"][0]["type"] == "text"
        assert response.data["content"][0]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_non_streaming_tool_calls(self):
        """OpenAI non-streaming with tool calls."""
        mock_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"query": "1+1"}'
                            }
                        }
                    ]
                }
            }]
        }
        response = OpenAIChatBotResponse.from_json(mock_data, OpenAIChatBot.RESPONSE_TRANSLATIONS)

        assert response.data["role"] == "assistant"
        assert "content" in response.data
        assert len(response.data["content"]) >= 1
        assert response.data["content"][0]["type"] == "tool_use"
        assert response.data["content"][0]["call_id"] == "tc_1"
        assert response.data["content"][0]["name"] == "calculator"
        assert response.data["content"][0]["arguments"] == '{"query": "1+1"}'

    @pytest.mark.asyncio
    async def test_non_streaming_reasoning(self):
        """OpenAI non-streaming with reasoning content."""
        mock_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Here is the answer",
                    "reasoning": "Let me think through this..."
                }
            }]
        }
        response = OpenAIChatBotResponse.from_json(mock_data, OpenAIChatBot.RESPONSE_TRANSLATIONS)

        assert response.data["role"] == "assistant"
        assert "content" in response.data
        assert len(response.data["content"]) == 2
        # Reasoning → thinking block (index 0)
        assert response.data["content"][0]["type"] == "thinking"
        assert response.data["content"][0]["content"] == "Let me think through this..."
        # Text (index 1)
        assert response.data["content"][1]["type"] == "text"
        assert response.data["content"][1]["content"] == "Here is the answer"

    @pytest.mark.asyncio
    async def test_stop_reason_preserved(self):
        """OpenAI finish_reason maps to unified stop_reason."""
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"role": "assistant", "content": "Hello"}}]}'
            yield 'data: {"choices": [{"delta": {"finish_reason": "stop"}}]}'
            yield "[DONE]"

        response = OpenAIChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        assert response.data.get("stop_reason") == "end_turn"


class TestOpenAIRequestTranslation:
    """Tests for OpenAIChatBot request building."""

    @pytest.mark.asyncio
    async def test_build_body_with_text_message(self):
        """Text content parts should be sent as OpenAI content string."""
        from peteos.chatbot import Message, ContentPart

        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = OpenAIChatBot(http_client, config)
        from peteos.conversation import Context
        context = Context.create()
        context.append(Message.create(
            role="user",
            content_parts=[ContentPart.create_text("Hello")],
        ))

        body = chatbot._build_body(context)
        messages = body.get("messages", [])
        assert len(messages) >= 1
        user_msg = messages[0]
        assert user_msg["role"] == "user"
        assert "content" in user_msg
        assert user_msg["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_build_body_with_reasoning_message(self):
        """Reasoning content parts should be sent as reasoning in OpenAI."""
        from peteos.chatbot import Message, ContentPart

        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = OpenAIChatBot(http_client, config)
        from peteos.conversation import Context
        context = Context.create()
        context.append(Message.create(
            role="assistant",
            content_parts=[
                ContentPart.create_thinking("Let me think..."),
                ContentPart.create_text("Here is the answer"),
            ]
        ))

        body = chatbot._build_body(context)
        messages = body.get("messages", [])
        assistant_msg = messages[0]
        assert assistant_msg["role"] == "assistant"
        # Should have reasoning and content fields
        assert "reasoning" in assistant_msg
        assert assistant_msg["reasoning"] == "Let me think..."
        assert assistant_msg["content"] == "Here is the answer"