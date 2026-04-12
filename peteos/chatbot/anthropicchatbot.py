"""Anthropic-compatible ChatBot implementation."""

import json
from typing import Any, Dict, AsyncGenerator

from peteos.logger import get_logger
from .chatbot import GenericChatBot
from .chatbotresponse import ChatBotResponse, GenericChatBotResponse
from .chathistory import ChatHistory
from peteos.utils.delta_merge import merge_delta_into_target as _merge_delta_into_target
from .httpclient import HTTPClient
from .message import Message

_logger = get_logger(__name__)


class AnthropicChatBot(GenericChatBot):
    """ChatBot implementation for Anthropic-compatible API."""

    # Default translation configuration for Anthropic API
    # Translates Anthropic SSE events to uniform delta format
    #
    # Anthropic SSE Structure:
    # - message_start: {"type":"message_start","message":{"role":"assistant"}}
    # - content_block_start: {"type":"content_block_start","content_block":{...},"index":0}
    # - content_block_delta: {"type":"content_block_delta","delta":{...},"index":0}
    # - message_delta: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}
    # - content_block_stop: {"type":"content_block_stop","index":0}
    #
    # Key difference from OpenAI: Anthropic places 'index' at top-level of events,
    # not inside arrays. The index indicates which content array item to update.
    #
    # Translation strategy: Field-by-field translation with index propagation
    # - Top-level 'index' from event is used as parent_index during translation
    # - Array syntax in target keys (e.g., "content[0].type") creates proper nested structure
    # - Type discriminators filter content_block_start/content_block_delta events
    #
    # Uniform format produced:
    # {
    #   "role": "assistant",
    #   "content": [
    #     {"index": 0, "type": "thinking", "content": "..."},
    #     {"index": 1, "type": "tool_use", "id": "...", "name": "...", "arguments": "..."}
    #   ],
    #   "stop_reason": "tool_use"
    # }
    RESPONSE_TRANSLATIONS = {
        # Top-level index - used for content array positioning
        "index": "content[0].index",
        # Message start - extract role
        "message_start.message.role": "role",
        # Message delta - extract stop_reason
        "message_delta.delta.stop_reason": "stop_reason",
        # Thinking blocks
        # Type discriminator: matches content_block_start when delta.type exists
        "content_block_start.content_block.type": "content[0].type",
        "content_block_start.content_block.thinking": "content[0].content",
        "content_block_delta.delta.thinking": "content[0].content",
        # Text blocks
        "content_block_start.content_block.type": "content[0].type",
        "content_block_start.content_block.text": "content[0].content",
        "content_block_delta.delta.text": "content[0].content",
        # Tool use blocks - field-by-field with array syntax
        "content_block_start.content_block.type": "content[0].type",
        "content_block_start.content_block.id": "content[0].id",
        "content_block_start.content_block.name": "content[0].name",
        "content_block_delta.delta.partial_json": "content[0].arguments",
    }

    REQUEST_TRANSLATIONS = {
        "text": "content",
        "reasoning": "reasoning",
        "tool_calls": "tool_calls",
    }

    @staticmethod
    def _translate_tool_params_to_anthropic(raw_params: dict) -> dict:
        """
        Translate internal tool parameters to Anthropic format.

        Internal format: {param_name: {type, required, default}, ...}
        Anthropic format: {"type": "object", "properties": {...}, "required": [...]}

        Args:
            raw_params: Tool parameters from internal format

        Returns:
            Anthropic-compatible input schema
        """
        required_params = []
        properties = {}

        type_map = {
            "int": "integer",
            "str": "string",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
            "any": "string",
        }

        for param_name, param_info in raw_params.items():
            if isinstance(param_info, str):
                json_type = type_map.get(param_info, "string")
                prop = {"type": json_type}
                required_params.append(param_name)
            else:
                py_type = param_info.get("type", "string")
                json_type = type_map.get(py_type, "string")

                prop = {"type": json_type}

                if "required" in param_info:
                    if param_info["required"]:
                        required_params.append(param_name)
                else:
                    required_params.append(param_name)

                if "default" in param_info:
                    prop["default"] = param_info["default"]

            properties[param_name] = prop

        return {
            "type": "object",
            "properties": properties,
            "required": required_params,
        }

    def __init__(self, http_client: HTTPClient, model: str, base_url: str, chat_endpoint: str = "/v1/messages", max_tokens: int = 4096):
        """
        Initialize AnthropicChatBot.

        Args:
            http_client: HTTP client for making API requests.
            model: The Anthropic model identifier (e.g., "claude-3-opus-20240229").
            base_url: The Anthropic API base URL.
            chat_endpoint: API-specific chat endpoint (default: "/v1/messages").
            max_tokens: Maximum tokens to generate (default: 4096).
        """
        super().__init__(
            http_client=http_client,
            model=model,
            base_url=base_url,
            chat_endpoint=chat_endpoint,
            models_endpoint="/v1/models",
            response_translations=self.RESPONSE_TRANSLATIONS,
            request_translations=self.REQUEST_TRANSLATIONS,
        )
        self._max_tokens = max_tokens

    async def send_message(
        self,
        chat_history: ChatHistory,
        streaming: bool = True,
        **kwargs
    ) -> ChatBotResponse:
        """Send a chat history to Anthropic-compatible API."""
        body = self._build_body(chat_history, streaming)
        body.update(kwargs)

        if streaming:
            stream = self._http_client.stream_post(f"{self._base_url}{self._chat_endpoint}", body)
            return AnthropicChatBotResponse(stream, self._translations)
        else:
            response_data = await self._http_client.post(f"{self._base_url}{self._chat_endpoint}", body)
            return AnthropicChatBotResponse.from_json(response_data, self._translations)

    def _build_body(self, chat_history: ChatHistory, streaming: bool) -> Dict[str, Any]:
        """Build Anthropic-specific request body.

        Anthropic format:
        - system goes to separate 'system' field
        - messages[] only contains user/assistant
        - tools use 'input_schema' instead of 'parameters'
        - tool_choice in Anthropic format
        """
        body = {}
        body["model"] = self._model
        body["stream"] = streaming
        body["max_tokens"] = self._max_tokens

        messages = []
        system_parts = []
        tools = []

        for msg in chat_history.messages:
            role = msg.role

            if role == "system":
                # Collect system parts
                system_parts.extend(msg.content)
            elif role == "tool":
                # Extract tool definitions and translate parameters to Anthropic format
                for part in msg.content:
                    if part.type == "tool":
                        # part.data: {name, description, parameters}
                        # Anthropic expects "input_schema" instead of "parameters"
                        tool_def = dict(part.data)
                        if "parameters" in tool_def:
                            tool_def["input_schema"] = self._translate_tool_params_to_anthropic(
                                part.data.get("parameters", {})
                            )
                        del tool_def["parameters"]
                        tools.append(tool_def)
            elif role in ("user", "assistant"):
                # Conversation messages
                content = []
                for part in msg.content:
                    if part.type == "tool_calls":
                        # Expand uniform tool_calls format to individual content items
                        for tool_call in part.data.get("tool_calls", []):
                            # tool_call: {'type': 'tool_use', 'id': '...', 'name': '...', 'arguments': '...'}
                            content_item = dict(tool_call)
                            # Convert arguments to input for Anthropic
                            if "arguments" in content_item:
                                content_item["input"] = json.loads(content_item["arguments"])
                                del content_item["arguments"]
                            content.append(content_item)
                    elif part.type == "reasoning":
                        # Convert reasoning to thinking for Anthropic format
                        content_item = {
                            "type": "thinking",
                            "thinking": part.data.get("reasoning", "")
                        }
                        content.append(content_item)
                    else:
                        content.append(part.to_dict())
                messages.append({
                    "role": role,
                    "content": content
                })

        body["messages"] = messages

        if system_parts:
            # Convert system parts to string
            system_text = " ".join(
                part.text for part in system_parts
                if part.type == "text" and part.text
            )
            body["system"] = system_text

        if tools:
            body["tools"] = tools

        # Copy generation config (excluding tool_choice for now)
        for key, value in chat_history.generation_config.items():
            if key not in body and key != "tool_choice":
                body[key] = value

        # Add tool_choice if present
        tool_choice = chat_history.generation_config.get("tool_choice")
        if tool_choice:
            body["tool_choice"] = tool_choice

        return body


class AnthropicChatBotResponse(GenericChatBotResponse):
    """ChatBotResponse for Anthropic-compatible API.

    Uses generic delta translation and merging from parent class.
    Anthropic's index field is top-level metadata (not in arrays), so it's
    simply ignored during translation.
    """

    def _accumulate_event(self, event: Dict[str, Any]) -> None:
        """
        Accumulate translated event into response dict.

        For Anthropic responses, also extract text from content array for
        backwards compatibility.
        """
        super()._accumulate_event(event)

        # Extract text from content array for backwards compatibility
        # Text blocks in Anthropic have type='text', thinking blocks have type='thinking'
        if "content" in self._data and isinstance(self._data["content"], list):
            # Concatenate all text content to 'text' field
            text_parts = [
                item.get("content", "")
                for item in self._data["content"]
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            if text_parts:
                self._data["text"] = "".join(text_parts)
            # Also concatenate thinking content to 'reasoning' field
            reasoning_parts = [
                item.get("content", "")
                for item in self._data["content"]
                if isinstance(item, dict) and item.get("type") == "thinking"
            ]
            if reasoning_parts:
                self._data["reasoning"] = "".join(reasoning_parts)

    def _process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an Anthropic event, defaulting role to 'assistant' if missing from message_start.

        The Anthropic API includes role in message_start. This fallback handles
        non-compliant backends that omit the role field entirely.
        """
        result = self._translate_event(event)

        # Default role to 'assistant' if message_start is missing it
        if event.get("type") == "message_start" and "role" not in event.get("message", {}):
            if "role" not in result:
                result["role"] = "assistant"

        return result
