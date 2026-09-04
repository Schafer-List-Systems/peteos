"""Unit tests for AnthropicChatBot implementation."""

import pytest

from peteos.utils import json

from peteos.chatbot.anthropicchatbot import AnthropicChatBot, AnthropicChatBotResponse
from peteos.chatbot.chatbotconfig import ChatBotConfig
from peteos.chatbot.httpclient import HTTPClient
from peteos.conversation.context import Context
from peteos.conversation.message import Message, ContentPart


def _make_context(messages: list[Message]) -> Context:
    """Build a Context from Message instances via the conversation package API."""
    ctx = Context.create()
    for msg in messages:
        ctx.append(msg)
    return ctx


class TestAnthropicChatBotResponse:
    """Tests for AnthropicChatBotResponse accumulation."""

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
        assert "text" not in response.data

    @pytest.mark.asyncio
    async def test_streaming_tool_calls(self):
        """Anthropic streaming with tool call produces content array with tool_use."""
        partial_json_1 = json.dumps({"query": "1"})
        start_event = json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "tc_1", "name": "calculator"}})
        delta1_event = json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": partial_json_1}})
        delta2_event = json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "1"}})
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
        assert response.data["content"][0]["call_id"] == "tc_1"
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
        assert "content" in response.data
        assert response.data["content"][0]["type"] == "text"
        assert response.data["content"][0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_non_streaming_text_content(self):
        """Anthropic non-streaming from_json produces content array."""
        mock_data = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Test response"}],
            "stop_reason": "end_turn",
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
                {"type": "text", "text": "Here is the answer"},
            ],
            "stop_reason": "end_turn",
        }
        response = AnthropicChatBotResponse.from_json(mock_data, AnthropicChatBot.RESPONSE_TRANSLATIONS)

        assert response.data["content"][0]["type"] == "thinking"
        assert response.data["content"][0]["content"] == "Let me think..."
        assert response.data["content"][1]["type"] == "text"
        assert response.data["content"][1]["content"] == "Here is the answer"

    @pytest.mark.asyncio
    async def test_non_streaming_tool_use(self):
        """Anthropic non-streaming with tool use converts input dict to arguments JSON."""
        mock_data = {
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "tc_1",
                "name": "calculator",
                "input": {"query": "1+1"},
            }],
            "stop_reason": "tool_use",
        }
        response = AnthropicChatBotResponse.from_json(mock_data, AnthropicChatBot.RESPONSE_TRANSLATIONS)

        assert response.data["content"][0]["type"] == "tool_use"
        assert response.data["content"][0]["call_id"] == "tc_1"
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

    @pytest.mark.asyncio
    async def test_role_preserved_from_message_start(self):
        """Anthropic response uses role from message_start, not default."""
        async def mock_stream():
            yield 'data: {"type": "message_start", "message": {"role": "bot"}}'
            yield 'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}'
            yield 'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}'
            yield "[DONE]"

        response = AnthropicChatBotResponse(mock_stream(), AnthropicChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        assert response.data["role"] == "bot"

    @pytest.mark.asyncio
    async def test_stop_reason_accumulated(self):
        """Anthropic streaming accumulates stop_reason."""
        async def mock_stream():
            yield 'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}'
            yield 'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}'
            yield 'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}'
            yield "[DONE]"

        response = AnthropicChatBotResponse(mock_stream(), AnthropicChatBot.RESPONSE_TRANSLATIONS)
        async for _ in response:
            pass

        assert response.data.get("stop_reason") == "end_turn"


class TestAnthropicRequestTranslation:
    """Tests for AnthropicChatBot request body building."""

    @pytest.mark.asyncio
    async def test_build_body_text_message(self):
        """Text content is sent as Anthropic content array."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="claude-3")
        chatbot = AnthropicChatBot(http_client, config)
        user_msg = Message.create("user", [ContentPart.create_text("Hello")])
        ctx = _make_context([user_msg])

        body = chatbot._build_body(ctx)

        assert body["model"] == "claude-3"
        assert body["stream"] is False
        assert isinstance(body["max_tokens"], int)
        assert body["max_tokens"] > 0
        messages = body.get("messages", [])
        assert len(messages) >= 1
        assert messages[0]["role"] == "user"
        assert isinstance(messages[0]["content"], list)
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][0]["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_build_body_with_thinking(self):
        """Thinking content is sent as thinking in Anthropic."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="claude-3")
        chatbot = AnthropicChatBot(http_client, config)
        assistant_msg = Message.create(
            "assistant",
            [
                ContentPart.create_thinking("Let me think..."),
                ContentPart.create_text("Answer"),
            ],
        )
        ctx = _make_context([assistant_msg])

        body = chatbot._build_body(ctx)
        messages = body.get("messages", [])
        assistant_result = messages[0]
        assert assistant_result["role"] == "assistant"
        thinking = [c for c in assistant_result["content"] if c.get("type") == "thinking"]
        text = [c for c in assistant_result["content"] if c.get("type") == "text"]
        assert len(thinking) == 1
        assert thinking[0]["thinking"] == "Let me think..."
        assert len(text) == 1
        assert text[0]["text"] == "Answer"

    @pytest.mark.asyncio
    async def test_build_body_with_system_prompt(self):
        """System messages are placed in the 'system' field."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="claude-3")
        chatbot = AnthropicChatBot(http_client, config)
        messages = [
            Message.create("system", [ContentPart.create_text("You are helpful")]),
            Message.create("user", [ContentPart.create_text("Hi")]),
        ]
        ctx = _make_context(messages)

        body = chatbot._build_body(ctx)

        assert body["system"] == "You are helpful"

    @pytest.mark.asyncio
    async def test_build_body_tool_choice_propagated(self):
        """tool_choice from generation_config is added to body."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="claude-3")
        chatbot = AnthropicChatBot(http_client, config)
        ctx = _make_context([])

        body = chatbot._build_body(ctx, generation_config={"tool_choice": {"type": "auto"}})

        assert body["tool_choice"] == {"type": "auto"}

    @pytest.mark.asyncio
    async def test_build_body_streaming_override(self):
        """streaming parameter overrides config default."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="claude-3")
        chatbot = AnthropicChatBot(http_client, config)
        ctx = _make_context([])

        body = chatbot._build_body(ctx, streaming=False)

        assert body["stream"] is False

    @pytest.mark.asyncio
    async def test_translate_tool_params_to_anthropic(self):
        """Tool params are translated to Anthropic input_schema format."""
        raw_params = {
            "query": {"type": "str", "required": True},
            "count": {"type": "int", "required": False, "default": 10},
        }
        result = AnthropicChatBot._translate_tool_params_to_anthropic(raw_params)

        assert result["type"] == "object"
        assert "query" in result["properties"]
        assert result["properties"]["query"]["type"] == "string"
        assert "query" in result["required"]
        assert result["properties"]["count"]["type"] == "integer"
        assert result["properties"]["count"]["default"] == 10
        assert "count" not in result["required"]

    def test_list_available_models_with_models(self):
        """list_available_models returns _models when set."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="claude-3")
        chatbot = AnthropicChatBot(http_client, config)
        chatbot._models = ["claude-3-opus", "claude-3-sonnet"]
        assert chatbot.list_available_models() == ["claude-3-opus", "claude-3-sonnet"]

    def test_list_available_models_fallback(self):
        """list_available_models returns [model] when _models is empty."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="claude-3-haiku")
        chatbot = AnthropicChatBot(http_client, config)
        assert chatbot.list_available_models() == ["claude-3-haiku"]


class TestAnthropicChatBotDefaults:
    """Tests for AnthropicChatBot endpoint and translation defaults."""

    def test_default_chat_endpoint(self):
        """Anthropic default chat endpoint is /v1/messages."""
        assert AnthropicChatBot.DEFAULT_CHAT_ENDPOINT == "/v1/messages"

    def test_default_models_endpoint(self):
        """Anthropic default models endpoint is /v1/models."""
        assert AnthropicChatBot.DEFAULT_MODELS_ENDPOINT == "/v1/models"

    def test_response_translations_defined(self):
        """Anthropic has RESPONSE_TRANSLATIONS defined."""
        assert AnthropicChatBot.RESPONSE_TRANSLATIONS is not None
        assert isinstance(AnthropicChatBot.RESPONSE_TRANSLATIONS, dict)

    def test_request_translations_defined(self):
        """Anthropic has REQUEST_TRANSLATIONS defined."""
        assert AnthropicChatBot.REQUEST_TRANSLATIONS is not None
        assert isinstance(AnthropicChatBot.REQUEST_TRANSLATIONS, dict)

    def test_bot_config_applies_defaults(self):
        """ChatBotConfig gets API-specific defaults filled in."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="claude-3")
        bot = AnthropicChatBot(http_client, config)
        assert bot._config.chat_endpoint == "/v1/messages"
        assert bot._config.response_translations is AnthropicChatBot.RESPONSE_TRANSLATIONS
        assert bot._config.request_translations is AnthropicChatBot.REQUEST_TRANSLATIONS
