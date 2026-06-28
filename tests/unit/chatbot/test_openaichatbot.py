"""Unit tests for OpenAIChatBot implementation."""

import json as _json

import pytest

from peteos.chatbot.openaichatbot import OpenAIChatBot, OpenAIChatBotResponse
from peteos.chatbot.chatbotconfig import ChatBotConfig
from peteos.chatbot.httpclient import HTTPClient
from peteos.conversation.context import Context
from peteos.conversation.message import Message, ContentPart


def _make_context(messages: list[Message]) -> Context:
    """Build a Context from Message instances via the conversation package API."""
    ctx = Context({})
    for msg in messages:
        ctx.append(msg)
    return ctx


class TestOpenAIChatBotResponse:
    """Tests for OpenAIChatBotResponse accumulation."""

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

        assert response.data["role"] == "assistant"
        assert "content" in response.data
        assert len(response.data["content"]) >= 1
        assert response.data["content"][0]["type"] == "text"
        assert response.data["content"][0]["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_streaming_reasoning_plus_text(self):
        """OpenAI streaming with reasoning and text produces thinking + text blocks."""
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
        assert response.data["content"][0]["type"] == "thinking"
        assert response.data["content"][0]["content"] == "Let me think..."
        assert response.data["content"][1]["type"] == "text"
        assert response.data["content"][1]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_streaming_tool_calls(self):
        """OpenAI streaming with tool call produces content array with tool_use."""
        first_delta = {
            "choices": [{
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 0,
                        "id": "tc_1",
                        "function": {"name": "calculator", "arguments": _json.dumps({"query": "1+1"})},
                    }],
                }
            }]
        }
        second_delta = {
            "choices": [{
                "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "2"}}]}
            }]
        }
        finish = {"choices": [{"delta": {"finish_reason": "tool_calls"}}]}
        async def mock_stream():
            yield f'data: {_json.dumps(first_delta)}'
            yield f'data: {_json.dumps(second_delta)}'
            yield f'data: {_json.dumps(finish)}'
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
                "message": {"role": "assistant", "content": "Test response"}
            }]
        }
        response = OpenAIChatBotResponse.from_json(mock_data, OpenAIChatBot.RESPONSE_TRANSLATIONS)

        assert response.data["role"] == "assistant"
        assert "content" in response.data
        assert response.data["content"][0]["type"] == "text"
        assert response.data["content"][0]["content"] == "Test response"
        assert "text" not in response.data

    @pytest.mark.asyncio
    async def test_non_streaming_content_as_list(self):
        """OpenAI non-streaming with content as array."""
        mock_data = {
            "choices": [{
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]}
            }]
        }
        response = OpenAIChatBotResponse.from_json(mock_data, OpenAIChatBot.RESPONSE_TRANSLATIONS)

        assert response.data["role"] == "assistant"
        assert response.data["content"][0]["type"] == "text"
        assert response.data["content"][0]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_non_streaming_tool_calls(self):
        """OpenAI non-streaming with tool calls."""
        mock_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "tc_1",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": '{"query": "1+1"}'},
                    }],
                }
            }]
        }
        response = OpenAIChatBotResponse.from_json(mock_data, OpenAIChatBot.RESPONSE_TRANSLATIONS)

        assert response.data["role"] == "assistant"
        assert "content" in response.data
        assert response.data["content"][0]["type"] == "tool_use"
        assert response.data["content"][0]["call_id"] == "tc_1"
        assert response.data["content"][0]["name"] == "calculator"
        assert response.data["content"][0]["arguments"] == '{"query": "1+1"}'

    @pytest.mark.asyncio
    async def test_non_streaming_reasoning(self):
        """OpenAI non-streaming with reasoning converts to thinking block."""
        mock_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Here is the answer",
                    "reasoning": "Let me think through this...",
                }
            }]
        }
        response = OpenAIChatBotResponse.from_json(mock_data, OpenAIChatBot.RESPONSE_TRANSLATIONS)

        assert response.data["role"] == "assistant"
        assert len(response.data["content"]) == 2
        assert response.data["content"][0]["type"] == "thinking"
        assert response.data["content"][0]["content"] == "Let me think through this..."
        assert response.data["content"][1]["type"] == "text"
        assert response.data["content"][1]["content"] == "Here is the answer"

    @pytest.mark.asyncio
    async def test_stop_reason_preserved(self):
        """OpenAI finish_reason maps to stop_reason."""
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"role": "assistant", "content": "Hello"}}]}'
            yield 'data: {"choices": [{"delta": {"finish_reason": "stop"}}]}'
            yield "[DONE]"

        response = OpenAIChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        assert response.data.get("stop_reason") == "stop"

    @pytest.mark.asyncio
    async def test_reasoning_text_concatenation(self):
        """OpenAI streaming accumulates reasoning text across chunks."""
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"reasoning": "Thinking"}}]}'
            yield 'data: {"choices": [{"delta": {"reasoning": " step by step"}}]}'
            yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
            yield "[DONE]"

        response = OpenAIChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        content = response.data["content"]
        assert content[0]["type"] == "thinking"
        assert content[0]["content"] == "Thinking step by step"
        assert content[1]["type"] == "text"
        assert content[1]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_response_data_access(self):
        """Test dict-like access to response data."""
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"role": "assistant", "content": "Hello"}}]}'
            yield "[DONE]"

        response = OpenAIChatBotResponse(mock_stream(), OpenAIChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        assert response["role"] == "assistant"
        assert "role" in response
        assert "text" not in response
        assert "content" in response.data


class TestOpenAIRequestTranslation:
    """Tests for OpenAIChatBot request body building."""

    @pytest.mark.asyncio
    async def test_build_body_text_message(self):
        """Text content sent as OpenAI content string."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="gpt-4")
        chatbot = OpenAIChatBot(http_client, config)
        user_msg = Message.create("user", [ContentPart.create_text("Hello")])
        ctx = _make_context([user_msg])

        body = chatbot._build_body(ctx)

        messages = body.get("messages", [])
        assert len(messages) >= 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_build_body_thinking_message(self):
        """Thinking content translated to reasoning field."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="gpt-4")
        chatbot = OpenAIChatBot(http_client, config)
        assistant_msg = Message.create(
            "assistant",
            [
                ContentPart.create_thinking("Let me think..."),
                ContentPart.create_text("Answer"),
            ],
        )
        ctx = _make_context([assistant_msg])

        body = chatbot._build_body(ctx)

        assistant_result = body["messages"][0]
        assert assistant_result["role"] == "assistant"
        assert assistant_result["reasoning"] == "Let me think..."
        assert assistant_result["content"] == "Answer"

    @pytest.mark.asyncio
    async def test_build_body_system_in_messages(self):
        """OpenAI includes system messages in the messages array."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="gpt-4")
        chatbot = OpenAIChatBot(http_client, config)
        messages = [
            Message.create("system", [ContentPart.create_text("You are helpful")]),
            Message.create("user", [ContentPart.create_text("Hi")]),
        ]
        ctx = _make_context(messages)

        body = chatbot._build_body(ctx)

        roles = [m["role"] for m in body["messages"]]
        assert "system" in roles
        assert "user" in roles

    @pytest.mark.asyncio
    async def test_build_body_tool_choice_propagated(self):
        """tool_choice from generation_config is added to body."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="gpt-4")
        chatbot = OpenAIChatBot(http_client, config)
        ctx = _make_context([])

        body = chatbot._build_body(ctx, generation_config={"tool_choice": "auto"})

        assert body["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_build_body_generation_config_merged(self):
        """generation_config keys not already in body are added."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="gpt-4")
        chatbot = OpenAIChatBot(http_client, config)
        ctx = _make_context([])

        body = chatbot._build_body(ctx, generation_config={"temperature": 0.7})

        assert body["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_build_body_generation_config_not_overwritten(self):
        """generation_config doesn't overwrite model, stream, max_tokens."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="gpt-4")
        chatbot = OpenAIChatBot(http_client, config)
        ctx = _make_context([])

        body = chatbot._build_body(
            ctx, generation_config={"model": "should-not-override", "temperature": 0.7}
        )

        assert body["model"] == "gpt-4"
        assert body["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_translate_tool_params_to_openai(self):
        """Tool params translated to OpenAI JSON Schema format."""
        raw_params = {
            "query": "str",
            "count": {"type": "int", "required": False, "default": 10},
        }
        result = OpenAIChatBot._translate_tool_params_to_openai(raw_params)

        assert result["type"] == "object"
        assert result["properties"]["query"]["type"] == "string"
        assert "query" in result["required"]
        assert result["properties"]["count"]["type"] == "integer"
        assert result["properties"]["count"]["default"] == 10
        assert "count" not in result["required"]

    def test_list_available_models_with_models(self):
        """list_available_models returns _models when set."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="gpt-4")
        bot = OpenAIChatBot(http_client, config)
        bot._models = ["gpt-4", "gpt-3.5-turbo"]
        assert bot.list_available_models() == ["gpt-4", "gpt-3.5-turbo"]

    def test_list_available_models_fallback(self):
        """list_available_models returns [model] when _models is empty."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="gpt-3.5-turbo")
        bot = OpenAIChatBot(http_client, config)
        assert bot.list_available_models() == ["gpt-3.5-turbo"]


class TestOpenAIChatBotDefaults:
    """Tests for OpenAIChatBot endpoint and translation defaults."""

    def test_default_chat_endpoint(self):
        """OpenAI default chat endpoint is /v1/chat/completions."""
        assert OpenAIChatBot.DEFAULT_CHAT_ENDPOINT == "/v1/chat/completions"

    def test_default_models_endpoint(self):
        """OpenAI default models endpoint is /v1/models."""
        assert OpenAIChatBot.DEFAULT_MODELS_ENDPOINT == "/v1/models"

    def test_response_translations_defined(self):
        """OpenAI has RESPONSE_TRANSLATIONS defined."""
        assert OpenAIChatBot.RESPONSE_TRANSLATIONS is not None
        assert isinstance(OpenAIChatBot.RESPONSE_TRANSLATIONS, dict)

    def test_request_translations_defined(self):
        """OpenAI has REQUEST_TRANSLATIONS defined."""
        assert OpenAIChatBot.REQUEST_TRANSLATIONS is not None
        assert isinstance(OpenAIChatBot.REQUEST_TRANSLATIONS, dict)

    def test_bot_config_applies_defaults(self):
        """ChatBotConfig gets API-specific defaults filled in."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="gpt-4")
        bot = OpenAIChatBot(http_client, config)
        assert bot._config.chat_endpoint == "/v1/chat/completions"
        assert bot._config.response_translations is OpenAIChatBot.RESPONSE_TRANSLATIONS
        assert bot._config.request_translations is OpenAIChatBot.REQUEST_TRANSLATIONS
