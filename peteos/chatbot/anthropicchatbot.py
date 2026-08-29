"""Anthropic-compatible ChatBot implementation."""

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from peteos.utils import get_logger
from .chatbot import ChatBot
from .chatbotconfig import ChatBotConfig
from .chatbotresponse import ChatBotResponse, GenericChatBotResponse
from peteos.conversation.context import Context
from peteos.utils.delta_merge import merge_delta_into_target as _merge_delta_into_target
from peteos.conversation.message import Message, ContentPart
from .httpclient import HTTPClient
from .response_types import normalize_stop_reason

_logger = get_logger(__name__)


async def augmented_yield():
    """Yields nothing - AnthropicChatBotResponse.from_json populates _data directly."""
    return
    yield


class AnthropicChatBot(ChatBot):
    """ChatBot implementation for Anthropic-compatible API."""

    DEFAULT_CHAT_ENDPOINT = "/v1/messages"
    DEFAULT_MODELS_ENDPOINT = "/v1/models"

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
        "content_block_start.content_block.id": "content[0].call_id",
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
            "any": "any_of",
        }

        for param_name, param_info in raw_params.items():
            if isinstance(param_info, str):
                py_type = param_info
                required_params.append(param_name)
                prop = {"type": type_map.get(py_type, "string")}
                properties[param_name] = prop
                continue

            py_type = param_info.get("type", "string")
            if param_info.get("required", True):
                required_params.append(param_name)

            json_type = type_map.get(py_type, "string")

            if json_type == "any_of":
                prop = {"anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "array"},
                    {"type": "object"},
                ]}
            else:
                prop = {"type": json_type}

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
        super().__init__(ChatBotConfig.from_dict(bot_cfg))
        self._models: List[str] = []

        # Capture api_key locally and build executors — api_key is then
        # captured in the closure and removed from config.
        api_key = self._config.api_key
        secure_headers: Dict[str, str] = {
            "anthropic-version": "2023-06-01",
        }
        if api_key:
            secure_headers["x-api-key"] = api_key
        endpoint = f"{self._config.url}{self._config.chat_endpoint}"
        self._post_executor = self._build_post_executor(http_client, secure_headers, endpoint)
        self._stream_executor = self._build_stream_executor(http_client, secure_headers, endpoint)
        self._config.api_key = None  # type: ignore[assignment]

    def list_available_models(self) -> List[str]:
        """List available models from the models endpoint or return [model]."""
        if self._models:
            return self._models
        return [self._config.model]

    async def send_context(
        self,
        context: Context,
        generation_config: Optional[Dict[str, Any]] = None,
        streaming: bool | None = None,
        **kwargs
    ) -> ChatBotResponse:
        """Send a context to the LLM and receive a response."""
        streaming_mode = self._config.streaming if streaming is None else streaming
        body = self._build_body(context, generation_config, streaming)
        body.update(kwargs)

        _logger.debug("Anthropic request: model=%s, messages=%d, tools=%d", self._config.model, len(body.get("messages", [])), len(body.get("tools", [])))

        if streaming_mode:
            stream = self._stream_executor(body, self.get_headers())
            return AnthropicChatBotResponse(stream, self._config.response_translations or {})
        else:
            response_data = await self._post_executor(body, self.get_headers())
            return AnthropicChatBotResponse.from_json(response_data, self._config.response_translations or {})

    def _build_body(
        self, context: Context, generation_config: Optional[Dict[str, Any]] = None, streaming: bool | None = None
    ) -> Dict[str, Any]:
        """Build Anthropic-specific request body.

        Anthropic format:
        - system goes to separate 'system' field
        - messages[] only contains user/assistant
        - tools use 'input_schema' instead of 'parameters'
        - tool_choice in Anthropic format
        """
        body: Dict[str, Any] = {}
        body["model"] = self._config.model
        body["stream"] = self._config.streaming if streaming is None else streaming
        body["max_tokens"] = self._config.max_tokens

        if generation_config:
            for key, value in generation_config.items():
                if key not in body and key != "tool_choice":
                    body[key] = value

        messages = []
        system_parts = []
        tools = []

        for msg in context.messages:
            role = msg.role

            if role == "system":
                # Collect system parts
                system_parts.extend(part.raw_dict for part in msg.content)
            elif role == "tool":
                # Extract tool definitions and translate parameters to Anthropic format
                for part in msg.content:
                    if part.type == "tool":
                        raw = part.raw_dict
                        # Strip internal fields before sending to Anthropic
                        tool_def = {k: v for k, v in raw.items() if k not in ("type", "parameters")}
                        if "parameters" in raw:
                            tool_def["input_schema"] = self._translate_tool_params_to_anthropic(
                                raw.get("parameters", {})
                            )
                        tools.append(tool_def)
            elif role in ("user", "assistant"):
                # Conversation messages
                content = []
                for part in msg.content:
                    raw = part.raw_dict
                    if raw.get("type") == "tool_calls":
                        # Expand uniform tool_calls format to individual content items
                        for tool_call in raw.get("tool_calls", []):
                            content_item = dict(tool_call)
                            if "arguments" in content_item:
                                content_item["input"] = json.loads(content_item["arguments"])
                                del content_item["arguments"]
                            # Framework uses call_id, Anthropic uses id
                            if "call_id" in content_item:
                                content_item["id"] = content_item.pop("call_id")
                            content.append(content_item)
                    elif raw.get("type") == "thinking":
                        content.append({
                            "type": "thinking",
                            "thinking": raw["text"],
                        })
                    else:
                        # Framework uses call_id, Anthropic uses id
                        # Framework uses arguments JSON string, Anthropic uses input dict
                        if raw.get("type") == "tool_use":
                            raw = dict(raw)
                            if "call_id" in raw:
                                raw["id"] = raw.pop("call_id")
                            if "arguments" in raw:
                                raw["input"] = json.loads(raw["arguments"])
                                del raw["arguments"]
                        content.append(raw)
                messages.append({
                    "role": role,
                    "content": content
                })
            elif role == "tool_result":
                content = []
                for part in msg.content:
                    if part.type == "tool_result":
                        content_item = {
                            "type": "tool_result",
                            "tool_use_id": part.call_id,
                            "content": part.content or "",
                        }
                        content.append(content_item)
                if content:
                    messages.append({
                        "role": "user",
                        "content": content
                    })

        _logger.debug("Anthropic request: system_parts=%d, tools=%d, messages=%d", len(system_parts), len(tools), len(messages))

        body["messages"] = messages

        if system_parts:
            # Convert system parts (dicts from serialize_content) to string
            system_text = " ".join(
                item["text"] for item in system_parts
                if item.get("type") == "text" and item.get("text")
            )
            body["system"] = system_text

        if tools:
            body["tools"] = tools

        # Add tool_choice if present
        if generation_config:
            tool_choice = generation_config.get("tool_choice")
            if tool_choice:
                body["tool_choice"] = tool_choice

        return body


class AnthropicChatBotResponse(GenericChatBotResponse):
    """ChatBotResponse for Anthropic-compatible API.

    Uses generic delta translation and merging from parent class.
    Anthropic's index field is top-level metadata (not in arrays), so it's
    simply ignored during translation.
issr
    For non-streaming, overrides from_json to extract fields from the
    complete response format (role, content array, stop_reason).
    """

    def _finalize(self) -> None:
        """Normalize Anthropic stop_reason to unified values."""
        raw = self._data.get("stop_reason")
        if raw is not None:
            self._data["stop_reason"] = normalize_stop_reason(raw)

    @classmethod
    def from_json(cls, data: Dict[str, Any], translations: Dict[str, str]) -> "AnthropicChatBotResponse":
        """Create a response from a complete Anthropic non-streaming response.

        Anthropic non-streaming returns a complete message, not SSE deltas.
        Extract role, build content array (converting input->arguments for
        tool_use items), and copy stop_reason directly.

        Args:
            data: Complete Anthropic API response.
            translations: Path translation configuration (unused for non-streaming).

        Returns:
            AnthropicChatBotResponse with extracted fields.
        """
        response = cls(augmented_yield(), translations)
        response._data = {}

        if "error" in data:
            response._data["error"] = str(data["error"])
            _logger.error("Anthropic error: %s", data["error"])
            return response

        # Extract role
        if "role" in data:
            response._data["role"] = data["role"]

        # Build uniform content array from content
        if "content" in data and isinstance(data["content"], list):
            content_array = []
            for item in data["content"]:
                if not isinstance(item, dict):
                    _logger.error("Expected content item to be a dict, got %s", type(item).__name__)
                    continue
                content_item = {"type": item.get("type", "text")}
                if item.get("type") == "tool_use":
                    # Convert input dict to arguments JSON string
                    # Normalize to the ChatBot contract key "call_id"
                    if "id" in item:
                        content_item["call_id"] = item["id"]
                    if "name" in item:
                        content_item["name"] = item["name"]
                    if "input" in item:
                        content_item["arguments"] = json.dumps(item["input"])
                elif item.get("type") == "thinking":
                    content_item["content"] = item.get("thinking", "")
                else:
                    # text blocks
                    content_item["content"] = item.get("text", "")
                content_array.append(content_item)
            response._data["content"] = content_array

        # Extract stop_reason
        if "stop_reason" in data:
            response._data["stop_reason"] = normalize_stop_reason(data["stop_reason"])

        return response

    def _accumulate_event(self, event: Dict[str, Any]) -> None:
        """Accumulate translated event and default role to 'assistant' if missing."""
        _merge_delta_into_target(self._data, event)
        if "role" not in self._data and "content" in self._data:
            self._data["role"] = "assistant"

    def _build_message(self) -> Message:
        """Convert Anthropic API-specific accumulated data into a Message."""
        role: str = self._data.get("role", "")
        content_array = self._data.get("content", [])
        content_parts: list[ContentPart] = []
        has_text = False

        if isinstance(content_array, list):
            for item in content_array:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type", "")
                if item_type == "tool_use":
                    content_parts.append(ContentPart(dict(item)))
                elif item_type == "texttool_use":
                    content_parts.append(ContentPart.create_text(item.get("content", "")))
                    content_parts.append(ContentPart(dict(item)))
                elif item_type == "text":
                    content_parts.append(ContentPart.create_text(item.get("content", "")))
                    has_text = True
                elif item_type == "thinking":
                    content_parts.append(ContentPart.create_thinking(item.get("content", "")))
                else:
                    _logger.debug("Unknown Anthropic content part type: %s", item_type)

        self._has_text_part = has_text
        return Message.create(role=role, content_parts=content_parts)
