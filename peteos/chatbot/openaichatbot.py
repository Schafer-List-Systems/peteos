"""OpenAI-compatible ChatBot implementation."""

import json
from dataclasses import asdict
from typing import Dict, Any, List, Optional, AsyncGenerator

from peteos.logger import get_logger
from .chatbot import GenericChatBot
from .chatbotconfig import ChatBotConfig
from .chatbotresponse import ChatBotResponse, GenericChatBotResponse
from .httpclient import HTTPClient
from .chathistory import ChatHistory
from .message import Message
from peteos.utils.delta_merge import merge_delta_into_target

_logger = get_logger(__name__)


class OpenAIChatBot(GenericChatBot):
    """ChatBot implementation for OpenAI-compatible API."""

    DEFAULT_CHAT_ENDPOINT = "/v1/chat/completions"
    DEFAULT_MODELS_ENDPOINT = "/v1/models"

    # Default translation configuration for OpenAI API
    # Translates OpenAI SSE events to uniform delta format
    # All index fields are preserved for merge_delta_into_target to use
    #
    # Default translation configuration for OpenAI API
    # Translates OpenAI SSE events to uniform delta format
    # All index fields are preserved for merge_delta_into_target to use
    #
    # Key difference from Anthropic: OpenAI content is plain text, not interleaved blocks
    # OpenAI returns text directly in 'content' field (not in an array with type)
    # Tool calls use the content array format with index fields
    RESPONSE_TRANSLATIONS = {
        # Streaming mode (delta events)
        "choices[*].delta.role": "role",
        "choices[*].delta.reasoning": "reasoning",
        # OpenAI content is plain text - map to 'text' for backwards compatibility
        "choices[*].delta.content": "text",
        "choices[*].delta.finish_reason": "stop_reason",
        # Tool calls: translate to uniform content array format
        # OpenAI returns one tool call per event, use [0] for extraction
        "choices[*].delta.tool_calls[0].index": "content[0].index",
        # OpenAI tool_calls doesn't have a type field - we set it in _process_event
        # Just extract the fields that exist: id, function.name, function.arguments
        "choices[*].delta.tool_calls[0].id": "content[0].id",
        "choices[*].delta.tool_calls[0].function.name": "content[0].name",
        "choices[*].delta.tool_calls[0].function.arguments": "content[0].arguments",
    }

    REQUEST_TRANSLATIONS = {
        "text": "content",
        "reasoning": "reasoning",
        "tool_calls": "tool_calls",
    }

    @staticmethod
    def _translate_tool_params_to_openai(raw_params: dict) -> dict:
        """
        Translate internal tool parameters to OpenAI JSON Schema format.

        Internal format: {param_name: {type, required, default}, ...}
        OpenAI format: {"type": "object", "properties": {...}, "required": [...]}

        Args:
            raw_params: Tool parameters from internal format

        Returns:
            OpenAI-compatible parameters schema
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
            # Handle both dict and string param_info
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

    def __init__(self, http_client: HTTPClient, config: ChatBotConfig):
        # Fill in API-specific defaults if not present in config
        bot_cfg = asdict(config)
        if bot_cfg.get("chat_endpoint") is None:
            bot_cfg["chat_endpoint"] = self.DEFAULT_CHAT_ENDPOINT
        if bot_cfg.get("response_translations") is None:
            bot_cfg["response_translations"] = self.RESPONSE_TRANSLATIONS
        if bot_cfg.get("request_translations") is None:
            bot_cfg["request_translations"] = self.REQUEST_TRANSLATIONS
        super().__init__(http_client, ChatBotConfig.from_dict(bot_cfg))

    async def send_message(
        self,
        chat_history: ChatHistory,
        streaming: bool | None = None,
    ) -> ChatBotResponse:
        """Send a chat history to the LLM and receive a response."""
        streaming_mode = self._config.streaming if streaming is None else streaming
        body = self._build_body(chat_history, streaming)

        if streaming_mode:
            stream = self._http_client.stream_post(f"{self._config.url}{self._config.chat_endpoint}", body)
            return OpenAIChatBotResponse(stream, self._config.response_translations or {})
        else:
            response_data = await self._http_client.post(f"{self._config.url}{self._config.chat_endpoint}", body)
            return OpenAIChatBotResponse.from_json(response_data, self._config.response_translations or {})

    def _build_body(self, chat_history: ChatHistory, streaming: bool | None = None) -> Dict[str, Any]:
        """Build OpenAI-specific request body.

        OpenAI format:
        - system messages included in messages[] array with role="system"
        - tools from tool messages
        - tool_choice from generation_config
        """
        body = {}
        body["model"] = self._config.model
        body["stream"] = self._config.streaming if streaming is None else streaming

        messages = []
        tools = []

        for msg in chat_history.messages:
            role = msg.get_role()

            if role == "tool":
                # Extract tool definitions and translate parameters to OpenAI format
                for part in msg.content:
                    if part.type == "tool":
                        # part.data: {name, description, parameters}
                        # Translate parameters to JSON Schema
                        params = self._translate_tool_params_to_openai(
                            part.data.get("parameters", {})
                        )
                        tool_def = {
                            "type": "function",
                            "function": {
                                "name": part.data.get("name"),
                                "description": part.data.get("description", ""),
                                "parameters": params
                            }
                        }
                        tools.append(tool_def)
            elif role == "tool_result":
                # Tool result messages - build tool_result dict for API
                for part in msg.content:
                    if part.type == "tool_result":
                        tool_name = part.data.get("name", "unknown")
                        tool_content = part.data.get("content", "")
                        msg_dict = {
                            "role": "tool",
                            "name": tool_name,
                            "content": tool_content
                        }
                        messages.append(msg_dict)
                        break
            elif role in ("user", "assistant", "system"):
                # Conversation messages
                msg_dict = {"role": role}
                tool_calls = []

                serialized = msg.serialize_content()
                for item in serialized:
                    if item.get("type") == "tool_calls":
                        for tool_call in item.get("tool_calls", []):
                            tool_calls.append({
                                "type": "function",
                                "id": tool_call.get("id"),
                                "function": {
                                    "name": tool_call.get("name"),
                                    "arguments": tool_call.get("arguments", "")
                                }
                            })
                    else:
                        # Apply key translation to flatten to OpenAI format
                        for key, value in item.items():
                            if key in (self._config.request_translations or {}):
                                api_key = self._config.request_translations[key]
                            else:
                                api_key = key
                            if api_key != "type":
                                msg_dict[api_key] = value

                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls

                messages.append(msg_dict)

        body["messages"] = messages

        if tools:
            body["tools"] = tools

        # Copy generation config (includes tool_choice)
        for key, value in chat_history.generation_config.items():
            if key not in body:
                body[key] = value

        _logger.debug("OpenAI request body: %s", json.dumps(body, indent=2))
        return body


class OpenAIChatBotResponse(GenericChatBotResponse):
    """ChatBotResponse for OpenAI-compatible API.

    Translates OpenAI SSE events to uniform delta format.
    Special handling: OpenAI tool_calls are converted to uniform content array
    with type="tool_use" for backwards compatibility with existing code.

    For non-streaming, overrides from_json to extract fields from the
    complete response format (choices array with message role/content).
    """

    @classmethod
    def from_json(cls, data: Dict[str, Any], translations: Dict[str, str]) -> "OpenAIChatBotResponse":
        """Create a response from a complete OpenAI non-streaming response.

        Args:
            data: Complete OpenAI API response.
            translations: Path translation configuration (unused for non-streaming).

        Returns:
            OpenAIChatBotResponse with extracted fields.
        """
        async def events():
            yield "data: {}"
            yield "[DONE]"
        response = cls(events(), translations)
        response._data = {}

        choices = data.get("choices", [])
        if choices and isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            # Extract role
            if "role" in message:
                response._data["role"] = message["role"]
            # Extract content
            content = message.get("content", "")
            if isinstance(content, str) and content:
                response._data["content"] = [{"index": 0, "type": "text", "content": content}]
                response._data["text"] = content
            elif isinstance(content, list):
                response._data["content"] = content
                text_parts = [
                    item.get("content", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                if text_parts:
                    response._data["text"] = "".join(text_parts)
            # Extract tool_calls
            if "tool_calls" in message and message["tool_calls"]:
                tool_calls = []
                for tc in message["tool_calls"]:
                    if isinstance(tc, dict):
                        tool_calls.append({
                            "type": "tool_use",
                            "id": tc.get("id"),
                            "name": tc.get("function", {}).get("name"),
                            "arguments": tc.get("function", {}).get("arguments", ""),
                        })
                if tool_calls:
                    response._data["tool_calls"] = tool_calls
            # Extract reasoning
            if "reasoning" in message and message["reasoning"]:
                response._data["reasoning"] = message["reasoning"]

        # Extract stop_reason from choices[0].finish_reason
        if choices and isinstance(choices, list) and choices:
            finish_reason = choices[0].get("finish_reason")
            if finish_reason:
                response._data["stop_reason"] = finish_reason

        return response

    @staticmethod
    def _set_tool_call_types(translated: Dict[str, Any]) -> None:
        """
        Set type="tool_use" for tool call items in translated event.

        OpenAI tool_calls don't have a type field (unlike Anthropic).
        The Uniform Delta Protocol expects type="tool_use" for all content array items.

        This is a static method so tests can call it directly on translated events.
        """
        if "content" in translated and isinstance(translated["content"], list):
            for item in translated["content"]:
                if isinstance(item, dict):
                    # Tool call items have either name+arguments (streaming) or name+id (initial event)
                    # or just name (minimal case)
                    if ("name" in item and ("arguments" in item or "id" in item or len(item) == 1)) or \
                       ("name" in item and "function" in item):
                        # This is a tool call item - set type to tool_use
                        item["type"] = "tool_use"

    def _process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an OpenAI event, setting type="tool_use" for tool call items.

        OpenAI tool_calls don't have a type field (unlike Anthropic).
        The Uniform Delta Protocol expects type="tool_use" for all content array items.
        This override sets type="tool_use" after translation.
        """
        translated = self._translate_event(event)
        self._set_tool_call_types(translated)
        return translated

    def _accumulate_event(self, event: Dict[str, Any]) -> None:
        """
        Accumulate translated event into response dict using delta merge.

        For OpenAI responses:
        - Content is plain text (not interleaved blocks like Anthropic)
        - The 'content' array contains text blocks
        - For backwards compatibility, also set 'text' field by concatenating text blocks

        Args:
            event: Translated event with target keys (and preserved index fields).
        """
        merge_delta_into_target(self._data, event)

        # For OpenAI responses: extract text from content array for backwards compatibility
        if "content" in self._data and isinstance(self._data["content"], list):
            # Concatenate all text content to 'text' field
            text_parts = [
                item.get("content", "")
                for item in self._data["content"]
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            if text_parts:
                self._data["text"] = "".join(text_parts)
