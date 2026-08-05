"""Unit tests for Gemini ChatBot implementation."""

import json

import pytest

from peteos.chatbot.geminichatbot import GeminiChatBot, GeminiChatBotResponse
from peteos.chatbot.chatbotconfig import ChatBotConfig
from peteos.chatbot.httpclient import HTTPClient


class TestGeminiChatBotResponse:
    """Tests for GeminiChatBot response accumulation."""

    @pytest.mark.asyncio
    async def test_streaming_text(self):
        """Gemini streaming text produces content array with text item."""
        # Gemini streaming sends full-state events as a JSON array
        events = [{
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "Hello"}]},
                "finishReason": "STOP",
            }]
        }]
        async def mock_stream():
            yield json.dumps(events)
            yield "[DONE]"

        response = GeminiChatBotResponse(mock_stream(), {})
        async for _ in response:
            pass

        assert response.data["role"] == "model"
        assert "content" in response.data
        assert len(response.data["content"]) >= 1
        assert response.data["content"][0]["type"] == "text"
        assert response.data["content"][0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_streaming_multiple_text_parts(self):
        """Gemini streaming with multiple text parts."""
        events = [{
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"text": "Part one"},
                    {"text": "Part two"},
                ]},
                "finishReason": "STOP",
            }]
        }]
        async def mock_stream():
            yield json.dumps(events)

        response = GeminiChatBotResponse(mock_stream(), {})
        async for _ in response:
            pass

        assert len(response.data["content"]) == 2
        assert response.data["content"][0]["content"] == "Part one"
        assert response.data["content"][1]["content"] == "Part two"

    @pytest.mark.asyncio
    async def test_streaming_tool_call(self):
        """Gemini streaming with function call produces tool_use item."""
        events = [{
            "candidates": [{
                "content": {"role": "model", "parts": [{
                    "functionCall": {
                        "name": "get_weather",
                        "args": {"city": "Zurich"},
                    }
                }]},
                "finishReason": "STOP",
            }]
        }]
        async def mock_stream():
            yield json.dumps(events)

        response = GeminiChatBotResponse(mock_stream(), {})
        async for _ in response:
            pass

        assert "content" in response.data
        assert response.data["content"][0]["type"] == "tool_use"
        assert response.data["content"][0]["name"] == "get_weather"
        assert response.data["content"][0]["arguments"] == json.dumps({"city": "Zurich"})

    @pytest.mark.asyncio
    async def test_streaming_tool_call_with_thought_signature(self):
        """Gemini streaming function call preserves thought_signature."""
        events = [{
            "candidates": [{
                "content": {"role": "model", "parts": [{
                    "functionCall": {"name": "get_weather", "args": {"city": "Zurich"}},
                    "thought_signature": "abc123",
                }]},
                "finishReason": "STOP",
            }]
        }]
        async def mock_stream():
            yield json.dumps(events)

        response = GeminiChatBotResponse(mock_stream(), {})
        async for _ in response:
            pass

        assert "content" in response.data
        assert response.data["content"][0]["type"] == "tool_use"
        assert response.data["content"][0]["thought_signature"] == "abc123"

    @pytest.mark.asyncio
    async def test_non_streaming_text(self):
        """Gemini non-streaming from_json produces content array."""
        mock_data = {
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "Test response"}]},
                "finishReason": "STOP",
            }]
        }
        response = GeminiChatBotResponse.from_json(mock_data)

        assert response.data["role"] == "model"
        assert "content" in response.data
        assert response.data["content"][0]["type"] == "text"
        assert response.data["content"][0]["content"] == "Test response"

    @pytest.mark.asyncio
    async def test_non_streaming_tool_call(self):
        """Gemini non-streaming with function call."""
        mock_data = {
            "candidates": [{
                "content": {"role": "model", "parts": [{
                    "functionCall": {
                        "name": "calculator",
                        "args": {"expression": "1+1"},
                    }
                }]},
                "finishReason": "STOP",
            }]
        }
        response = GeminiChatBotResponse.from_json(mock_data)

        assert "content" in response.data
        assert response.data["content"][0]["type"] == "tool_use"
        assert response.data["content"][0]["name"] == "calculator"
        assert response.data["content"][0]["arguments"] == json.dumps({"expression": "1+1"})

    @pytest.mark.asyncio
    async def test_stop_reason_from_finish_reason(self):
        """Gemini finishReason maps to unified stop_reason."""
        mock_data = {
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "Hello"}]},
                "finishReason": "MAX_TOKENS",
            }]
        }
        response = GeminiChatBotResponse.from_json(mock_data)

        assert response.data["stop_reason"] == "max_tokens"

    @pytest.mark.asyncio
    async def test_safety_stop_reason(self):
        """SAFETY finishReason maps to content_filter stop_reason."""
        mock_data = {
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "Hello"}]},
                "finishReason": "SAFETY",
            }]
        }
        response = GeminiChatBotResponse.from_json(mock_data)

        assert response.data["stop_reason"] == "content_filter"

    @pytest.mark.asyncio
    async def test_empty_response(self):
        """Gemini response with no candidates produces empty data."""
        mock_data = {"candidates": []}
        response = GeminiChatBotResponse.from_json(mock_data)

        assert response.data == {}

    @pytest.mark.asyncio
    async def test_error_in_response(self):
        """Gemini error is preserved in response data."""
        mock_data = {
            "error": {
                "message": "API key not valid",
                "code": 400,
            }
        }
        response = GeminiChatBotResponse.from_json(mock_data)

        assert "error" in response.data
        assert "API key not valid" in response.data["error"]


class TestGeminiChatBotRequest:
    """Tests for GeminiChatBot request building."""

    @pytest.mark.asyncio
    async def test_build_body_text_message(self):
        """Text content parts should be sent as Gemini text parts."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = GeminiChatBot(http_client, config)
        from peteos.conversation import Context
        from peteos.conversation.message import Message, ContentPart
        context = Context.create()
        context.append(Message.create(
            role="user",
            content_parts=[ContentPart.create_text("Hello")],
        ))

        body = chatbot._build_body(context)
        contents = body.get("contents", [])
        assert len(contents) >= 1
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"][0] == {"text": "Hello"}

    @pytest.mark.asyncio
    async def test_build_body_assistant_text(self):
        """Assistant text should be sent as Gemini model role."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = GeminiChatBot(http_client, config)
        from peteos.conversation import Context
        from peteos.conversation.message import Message, ContentPart
        context = Context.create()
        context.append(Message.create(
            role="assistant",
            content_parts=[ContentPart.create_text("Answer")],
        ))

        body = chatbot._build_body(context)
        contents = body.get("contents", [])
        assistant = contents[-1]
        assert assistant["role"] == "model"
        assert assistant["parts"][0] == {"text": "Answer"}

    @pytest.mark.asyncio
    async def test_build_body_thinking_content(self):
        """Thinking content should be sent as text in Gemini."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = GeminiChatBot(http_client, config)
        from peteos.conversation import Context
        from peteos.conversation.message import Message, ContentPart
        context = Context.create()
        context.append(Message.create(
            role="assistant",
            content_parts=[ContentPart.create_thinking("Let me think...")],
        ))

        body = chatbot._build_body(context)
        contents = body.get("contents", [])
        # Thinking maps to text in Gemini
        assert contents[-1]["parts"][0] == {"text": "Let me think..."}

    @pytest.mark.asyncio
    async def test_build_body_function_call(self):
        """Tool use from assistant should be sent as functionCall part."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = GeminiChatBot(http_client, config)
        from peteos.conversation import Context
        from peteos.conversation.message import Message, ContentPart
        context = Context.create()
        context.append(Message.create(
            role="assistant",
            content_parts=[ContentPart.create_tool_use(
                call_id="tc_1",
                name="get_weather",
                arguments=json.dumps({"city": "Zurich"}),
            )],
        ))

        body = chatbot._build_body(context)
        contents = body.get("contents", [])
        parts = contents[-1]["parts"]
        assert "functionCall" in parts[0]
        assert parts[0]["functionCall"]["name"] == "get_weather"
        assert parts[0]["functionCall"]["args"] == {"city": "Zurich"}

    @pytest.mark.asyncio
    async def test_build_body_function_call_with_thought_signature(self):
        """functionCall should echo back thought_signature if present."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = GeminiChatBot(http_client, config)
        from peteos.conversation import Context
        from peteos.conversation.message import Message, ContentPart
        context = Context.create()
        # Simulate a tool_use part with thought_signature (from Gemini response)
        # ContentPart stores extra fields in _json_dict
        part_dict = {
            "index": 0,
            "type": "tool_use",
            "call_id": "tc_1",
            "name": "get_weather",
            "arguments": json.dumps({"city": "Zurich"}),
            "thought_signature": "abc123",
        }
        context.append(Message.create(
            role="assistant",
            content_parts=[ContentPart(part_dict)],
        ))

        body = chatbot._build_body(context)
        parts = body["contents"][-1]["parts"]
        fc = parts[0]["functionCall"]
        assert fc["name"] == "get_weather"
        assert parts[0]["thought_signature"] == "abc123"

    @pytest.mark.asyncio
    async def test_build_body_tool_result(self):
        """Tool result should be sent as functionResponse part."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = GeminiChatBot(http_client, config)
        from peteos.conversation import Context
        from peteos.conversation.message import Message, ContentPart
        context = Context.create()
        # Tool result needs a preceding model message with the call_id lookup
        context.append(Message.create(
            role="assistant",
            content_parts=[ContentPart.create_tool_use(
                call_id="tc_1",
                name="get_weather",
                arguments=json.dumps({"city": "Zurich"}),
            )],
        ))
        context.append(Message.create(
            role="tool_result",
            content_parts=[ContentPart.create_tool_result(
                call_id="tc_1",
                content='{"temp": 20}',
            )],
        ))

        body = chatbot._build_body(context)
        contents = body.get("contents", [])
        # Tool result should be in the last user message as functionResponse
        last_msg = contents[-1]
        assert last_msg["role"] == "user"
        assert any("functionResponse" in p for p in last_msg["parts"])
        func_resp = next(p["functionResponse"] for p in last_msg["parts"] if "functionResponse" in p)
        assert func_resp["name"] == "get_weather"
        assert func_resp["response"]["content"] == '{"temp": 20}'

    @pytest.mark.asyncio
    async def test_build_body_tools_definition(self):
        """Tool definitions should be sent in the tools array."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = GeminiChatBot(http_client, config)
        from peteos.conversation import Context
        from peteos.conversation.message import Message, ContentPart
        context = Context.create()
        context.append(Message.create(
            role="tool",
            content_parts=[ContentPart.create_tool(
                name="get_weather",
                description="Get weather",
                parameters={
                    "city": {"type": "str", "required": True},
                },
            )],
        ))

        body = chatbot._build_body(context)
        tools = body.get("tools", [])
        assert len(tools) >= 1
        func_decl = tools[0]["functionDeclarations"][0]
        assert func_decl["name"] == "get_weather"
        assert func_decl["description"] == "Get weather"
        assert "city" in func_decl["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_build_body_system_instruction(self):
        """System messages should be sent as system_instruction."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = GeminiChatBot(http_client, config)
        from peteos.conversation import Context
        from peteos.conversation.message import Message, ContentPart
        context = Context.create()
        context.append(Message.create(
            role="system",
            content_parts=[ContentPart.create_text("You are a helpful assistant")],
        ))

        body = chatbot._build_body(context)
        assert "system_instruction" in body
        assert body["system_instruction"]["role"] == "system"
        assert body["system_instruction"]["parts"][0]["text"] == "You are a helpful assistant"

    @pytest.mark.asyncio
    async def test_build_body_with_generation_config(self):
        """Generation config should be forwarded to the body."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = GeminiChatBot(http_client, config)
        from peteos.conversation import Context
        from peteos.conversation.message import Message, ContentPart
        context = Context.create()
        context.append(Message.create(
            role="user",
            content_parts=[ContentPart.create_text("Hello")],
        ))

        body = chatbot._build_body(context, generation_config={"temperature": 0.5})
        gen_config = body.get("generationConfig", {})
        assert gen_config["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_build_body_generation_config_passthrough(self):
        """generationConfig keys should pass through to body."""
        http_client = HTTPClient(timeout=5.0)
        config = ChatBotConfig(name="test", url="http://test:8000", model="test-model")
        chatbot = GeminiChatBot(http_client, config)
        from peteos.conversation import Context
        from peteos.conversation.message import Message, ContentPart
        context = Context.create()
        context.append(Message.create(
            role="user",
            content_parts=[ContentPart.create_text("Hello")],
        ))

        # temperature and max_tokens pass through generationConfig
        body = chatbot._build_body(context, generation_config={"temperature": 0.5, "max_tokens": 100})
        gen_config = body.get("generationConfig", {})
        assert gen_config["temperature"] == 0.5
        assert gen_config["maxOutputTokens"] == 100
