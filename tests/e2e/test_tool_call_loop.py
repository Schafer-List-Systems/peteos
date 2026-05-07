"""End-to-end tests for tool call translation to uniform format.

Verifies:
1. OpenAI and Anthropic produce uniform `content` array for tool calls
2. Index fields are correctly propagated
3. Tool call types are set correctly
"""

import pytest
from peteos.chatbot.openaichatbot import OpenAIChatBot, OpenAIChatBotResponse
from peteos.chatbot.anthropicchatbot import AnthropicChatBot, AnthropicChatBotResponse
from peteos.chatbot.chatbotconfig import ChatBotConfig
from peteos.utils.delta_merge import merge_delta_into_target, translate_delta_event
from peteos.chatbot.message import Message
from peteos.chatbot.contentpart import ContentPart
from peteos.chatbot.chathistory import ChatHistory


class MockHTTPClient:
    """Mock HTTP client for testing."""

    def __init__(self, mock_responses=None):
        self.mock_responses = mock_responses or []
        self.request_history = []

    async def post(self, url, body):
        self.request_history.append(("POST", url, body))
        return self.mock_responses.pop(0) if self.mock_responses else {}

    def stream_post(self, url, body):
        self.request_history.append(("STREAM_POST", url, body))


def _make_config(api_type, model, base_url):
    return ChatBotConfig(
        name=f"test-{api_type}",
        url=base_url,
        api_type=api_type,
        model=model,
    )


class TestOpenAIToolCallTranslation:
    """Test OpenAI tool call translation."""

    def test_openai_translation_produces_content_array(self):
        """Verify OpenAI translations produce `content` array (not `tool_calls`)."""
        http_client = MockHTTPClient()
        chatbot = OpenAIChatBot(
            http_client=http_client,
            config=_make_config("openai", "gpt-4", "http://test-backend:8000")
        )

        event = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc",
                        "function": {
                            "name": "calculate",
                            "arguments": '{"expr":"2+2"}'
                        }
                    }]
                }
            }]
        }

        translated = translate_delta_event(event, chatbot.RESPONSE_TRANSLATIONS)

        # Verify uniform format with `content` array
        assert "content" in translated
        assert len(translated["content"]) == 1
        assert translated["content"][0]["name"] == "calculate"
        assert translated["content"][0]["arguments"] == '{"expr":"2+2"}'

    def test_openai_chatbot_response_converts_type(self):
        """Verify OpenAIChatBotResponse sets type="tool_use" for tool call items."""
        http_client = MockHTTPClient()
        chatbot = OpenAIChatBot(
            http_client=http_client,
            config=_make_config("openai", "gpt-4", "http://test-backend:8000")
        )

        # First tool call event has all fields including id, type, name
        event = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": ""
                        }
                    }]
                }
            }]
        }

        response = OpenAIChatBotResponse.__new__(OpenAIChatBotResponse)
        response._data = {}
        response._translations = chatbot.RESPONSE_TRANSLATIONS

        processed = response._process_event(event)

        # Verify type is set to "tool_use" (not "function")
        assert "content" in processed
        assert processed["content"][0]["type"] == "tool_use"
        assert processed["content"][0]["name"] == "calculate"
        assert processed["content"][0]["id"] == "call_abc"

    def test_openai_accumulate_tool_call_via_response(self):
        """Verify OpenAI tool call accumulates correctly through response.

        Real SSE flow:
        1. First event: tool_calls with id, type, name, empty arguments
        2. Subsequent events: only index + argument fragments
        """
        http_client = MockHTTPClient()
        chatbot = OpenAIChatBot(
            http_client=http_client,
            config=_make_config("openai", "gpt-4", "http://test-backend:8000")
        )

        # Simulate actual SSE events from OpenAI backend
        events = [
            # First: establishes tool call with id, type, name
            {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "calculate", "arguments": ""}
                        }]
                    }
                }]
            },
            # Second: argument fragment
            {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": '{"expr":"2+2"}'}
                        }]
                    }
                }]
            },
        ]

        response = OpenAIChatBotResponse.__new__(OpenAIChatBotResponse)
        response._data = {}
        response._translations = chatbot.RESPONSE_TRANSLATIONS

        for event in events:
            translated = response._process_event(event)
            merge_delta_into_target(response._data, translated)

        # Verify complete tool call accumulation
        assert response._data["content"][0]["name"] == "calculate"
        assert response._data["content"][0]["arguments"] == '{"expr":"2+2"}'
        assert response._data["content"][0]["type"] == "tool_use"
        assert response._data["content"][0]["id"] == "call_abc"


class TestAnthropicToolCallTranslation:
    """Test Anthropic tool call translation."""

    def test_anthropic_translation_with_index_propagation(self):
        """Verify Anthropic translations correctly propagate index field."""
        http_client = MockHTTPClient()
        chatbot = AnthropicChatBot(
            http_client=http_client,
            config=_make_config("anthropic", "claude-3-opus", "http://test-backend:8000")
        )

        event = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_xxx",
                "name": "calculate"
            }
        }

        translated = translate_delta_event(event, chatbot.RESPONSE_TRANSLATIONS)

        assert "content" in translated
        assert len(translated["content"]) == 1
        assert translated["content"][0]["index"] == 0
        assert translated["content"][0]["type"] == "tool_use"
        assert translated["content"][0]["name"] == "calculate"
        assert translated["content"][0]["id"] == "toolu_xxx"

    def test_anthropic_accumulate_tool_arguments(self):
        """Verify Anthropic tool arguments accumulate correctly."""
        http_client = MockHTTPClient()
        chatbot = AnthropicChatBot(
            http_client=http_client,
            config=_make_config("anthropic", "claude-3-opus", "http://test-backend:8000")
        )

        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_xxx",
                    "name": "calculate"
                }
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"expr":"2+2"}'}
            },
        ]

        response = AnthropicChatBotResponse.__new__(AnthropicChatBotResponse)
        response._data = {}
        response._translations = chatbot.RESPONSE_TRANSLATIONS

        for event in events:
            translated = response._process_event(event)
            merge_delta_into_target(response._data, translated)

        assert response._data["content"][0]["name"] == "calculate"
        assert response._data["content"][0]["arguments"] == '{"expr":"2+2"}'
        assert response._data["content"][0]["id"] == "toolu_xxx"

    def test_anthropic_message_start_sets_role(self):
        """Verify Anthropic message_start sets role to assistant."""
        http_client = MockHTTPClient()
        chatbot = AnthropicChatBot(
            http_client=http_client,
            config=_make_config("anthropic", "claude-3-opus", "http://test-backend:8000")
        )

        event = {
            "type": "message_start",
            "message": {"role": "assistant"}
        }

        translated = translate_delta_event(event, chatbot.RESPONSE_TRANSLATIONS)
        assert translated.get("role") == "assistant"


class TestUniformProtocolAlignment:
    """Test that OpenAI and Anthropic produce identical uniform format."""

    def test_both_produce_content_array_structure(self):
        """Verify OpenAI and Anthropic both produce `content` array."""
        openai_client = MockHTTPClient()
        openai_chatbot = OpenAIChatBot(
            http_client=openai_client,
            config=_make_config("openai", "gpt-4", "http://test-backend:8000")
        )

        anthropic_client = MockHTTPClient()
        anthropic_chatbot = AnthropicChatBot(
            http_client=anthropic_client,
            config=_make_config("anthropic", "claude-3-opus", "http://test-backend:8000")
        )

        # OpenAI first tool call event (has id, type, name)
        openai_event = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "calculate"}
                    }]
                }
            }]
        }

        # Anthropic content_block_start event
        anthropic_event = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_xxx",
                "name": "calculate"
            }
        }

        openai_translated = translate_delta_event(openai_event, openai_chatbot.RESPONSE_TRANSLATIONS)
        # Set type="tool_use" for OpenAI tool call items (OpenAI API doesn't include type in tool_calls)
        OpenAIChatBotResponse._set_tool_call_types(openai_translated)
        anthropic_translated = translate_delta_event(anthropic_event, anthropic_chatbot.RESPONSE_TRANSLATIONS)

        openai_result = {}
        merge_delta_into_target(openai_result, openai_translated)

        anthropic_result = {}
        merge_delta_into_target(anthropic_result, anthropic_translated)

        # Verify both produce `content` array (not `tool_calls`)
        assert "content" in openai_result
        assert "content" in anthropic_result
        assert "tool_calls" not in openai_result
        assert "tool_calls" not in anthropic_result

        # Verify both have tool_use type
        assert openai_result["content"][0]["type"] == "tool_use"
        assert anthropic_result["content"][0]["type"] == "tool_use"

        # Verify both have required fields for tool execution
        assert openai_result["content"][0]["name"] == "calculate"
        assert anthropic_result["content"][0]["name"] == "calculate"


class TestRequestBuilding:
    """Test that requests are built correctly with tool definitions."""

    def test_openai_builds_tool_parameters_correctly(self):
        """Verify OpenAI builds tool parameters in correct format.

        OpenAI expects: {"type": "function", "function": {"name": ..., "parameters": {...}}}
        """
        http_client = MockHTTPClient()
        chatbot = OpenAIChatBot(
            http_client=http_client,
            config=_make_config("openai", "gpt-4", "http://test-backend:8000")
        )

        messages = [
            Message(
                role="tool",
                content=[ContentPart(
                    part_type="tool",
                    name="calculate",
                    description="Calculate expression",
                    parameters={"expression": {"type": "str"}}
                )]
            ),
            Message(
                role="user",
                content=[ContentPart(part_type="text", text="Hello")]
            )
        ]
        chat_history = ChatHistory(messages=messages)

        body = chatbot._build_body(chat_history, streaming=False)

        assert "tools" in body
        assert len(body["tools"]) == 1
        tool_def = body["tools"][0]

        # OpenAI format: {"type": "function", "function": {...}}
        assert "type" in tool_def
        assert tool_def["type"] == "function"
        assert "function" in tool_def
        assert tool_def["function"]["name"] == "calculate"
        assert "parameters" in tool_def["function"]
        assert tool_def["function"]["parameters"]["type"] == "object"
        assert "expression" in tool_def["function"]["parameters"]["properties"]

    def test_anthropic_builds_tool_parameters_correctly(self):
        """Verify Anthropic builds tool parameters in correct format."""
        http_client = MockHTTPClient()
        chatbot = AnthropicChatBot(
            http_client=http_client,
            config=_make_config("anthropic", "claude-3-opus", "http://test-backend:8000")
        )

        messages = [
            Message(
                role="tool",
                content=[ContentPart(
                    part_type="tool",
                    name="calculate",
                    description="Calculate expression",
                    parameters={"expression": {"type": "str"}}
                )]
            ),
            Message(
                role="user",
                content=[ContentPart(part_type="text", text="Hello")]
            )
        ]
        chat_history = ChatHistory(messages=messages)

        body = chatbot._build_body(chat_history, streaming=False)

        assert "tools" in body
        assert len(body["tools"]) == 1
        tool_def = body["tools"][0]
        assert tool_def["name"] == "calculate"
        assert "input_schema" in tool_def
        assert tool_def["input_schema"]["type"] == "object"
        assert "expression" in tool_def["input_schema"]["properties"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
