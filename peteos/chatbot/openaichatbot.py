"""OpenAI-compatible ChatBot implementation."""

import json
from dataclasses import asdict
from typing import Dict, Any, List, Optional, AsyncGenerator, AsyncIterator

from peteos.utils import get_logger
from .chatbot import ChatBot
from .chatbotconfig import ChatBotConfig
from .chatbotresponse import ChatBotResponse, GenericChatBotResponse
from .httpclient import HTTPClient
from peteos.conversation.context import Context
from peteos.utils.delta_merge import merge_delta_into_target
from peteos.conversation.message import Message, ContentPart
from .response_types import normalize_stop_reason

_logger = get_logger(__name__)

_THINKING_MARKERS = [
    ("<think>", "</think>"),
    ("<|channel>thought", "<channel|>"),
]


def _split_thinking_tokens(text: str) -> tuple[str, str]:
    """Extract thinking portion from text marked with inline thinking markers.

    Supports two marker styles:
      - <think> / </think> (GLM models)
      - <|channel>thought / <|channel|> (Gemma models)

    Args:
        text: The raw response text which may contain inline thinking markers.

    Returns:
        A tuple of (thinking_text, remaining_text). When markers are found,
        thinking_text contains the content between them, and remaining_text
        contains everything after. When no markers are found, returns
        ("", text) — the entire input becomes remaining_text.
    """
    for start_tag, end_tag in _THINKING_MARKERS:
        start = text.find(start_tag)
        if start == -1:
            continue
        end = text.find(end_tag, start)
        if end == -1:
            return ("", text[start + len(start_tag):])
        thinking = text[start + len(start_tag):end]
        remaining = text[end + len(end_tag):]
        return (thinking, remaining)
    return ("", text)


class OpenAIChatBot(ChatBot):
    """ChatBot implementation for OpenAI-compatible API."""

    DEFAULT_CHAT_ENDPOINT = "/v1/chat/completions"
    DEFAULT_MODELS_ENDPOINT = "/v1/models"

    # Default translation configuration for OpenAI API
    # Translates OpenAI SSE events to uniform delta format
    # All index fields are preserved for merge_delta_into_target to use
    #
    # Unified content array format (same as Anthropic).
    # Reasoning maps to a temp key, converted to a "thinking" block in content array at stream end.
    RESPONSE_TRANSLATIONS = {
        # Streaming mode (delta events)
        "choices[*].delta.role": "role",
        # Reasoning is kept as a separate top-level field to avoid collision
        # with content[0].content in the unified content array.
        # It is converted to a "thinking" block in content[0] at stream end.
        "choices[*].delta.reasoning": "_reasoning",
        "choices[*].delta.content": "content[0].content",
        "choices[*].delta.finish_reason": "stop_reason",
        # Tool calls: translate to uniform content array format
        "choices[*].delta.tool_calls[0].index": "content[0].index",
        "choices[*].delta.tool_calls[0].id": "content[0].call_id",
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
        secure_headers: Dict[str, str] = {}
        if api_key:
            secure_headers["Authorization"] = f"Bearer {api_key}"
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
    ) -> ChatBotResponse:
        """Send a context to the LLM and receive a response."""
        streaming_mode = self._config.streaming if streaming is None else streaming
        body = self._build_body(context, generation_config, streaming)

        _logger.debug("OpenAI request: model=%s, messages=%d, tools=%d", self._config.model, len(body.get("messages", [])), len(body.get("tools", [])))

        if streaming_mode:
            stream = self._stream_executor(body, self.get_headers())
            return OpenAIChatBotResponse(stream, self._config.response_translations or {})
        else:
            response_data = await self._post_executor(body, self.get_headers())
            return OpenAIChatBotResponse.from_json(response_data, self._config.response_translations or {})

    def _build_body(self, context: Context, generation_config: Optional[Dict[str, Any]] = None, streaming: bool | None = None) -> Dict[str, Any]:
        """Build OpenAI-specific request body.

        OpenAI format:
        - system messages included in messages[] array with role="system"
        - tools from tool messages
        - tool_choice from generation_config
        """
        body: Dict[str, Any] = {}
        body["model"] = self._config.model
        body["stream"] = self._config.streaming if streaming is None else streaming
        body["max_tokens"] = self._config.max_tokens

        if generation_config:
            for key, value in generation_config.items():
                if key not in body:
                    body[key] = value

        messages = []
        tools = []

        for msg in context.messages:
            role = msg.role

            if role == "tool":
                # Extract tool definitions and translate parameters to OpenAI format
                for part in msg.content:
                    if part.type == "tool":
                        raw = part.raw_dict
                        # Translate parameters to JSON Schema
                        params = self._translate_tool_params_to_openai(
                            raw.get("parameters", {})
                        )
                        tool_def = {
                            "type": "function",
                            "function": {
                                "name": raw.get("name"),
                                "description": raw.get("description", ""),
                                "parameters": params
                            }
                        }
                        tools.append(tool_def)
            elif role == "tool_result":
                # Tool result messages - build tool_result dict for API
                for part in msg.content:
                    if part.type == "tool_result":
                        msg_dict = {
                            "role": "tool",
                            "tool_call_id": part.call_id,
                            "content": part.content
                        }
                        messages.append(msg_dict)
            elif role in ("user", "assistant", "system"):
                # Conversation messages
                msg_dict = {"role": role}
                tool_calls = []
                content_parts: List[Dict[str, Any]] = []

                for part in msg.content:
                    raw = part.raw_dict
                    if raw.get("type") == "tool_use":
                        tool_calls.append({
                            "type": "function",
                            "id": raw["call_id"],
                            "function": {
                                "name": raw["name"],
                                "arguments": raw["arguments"]
                            }
                        })
                    elif raw.get("type") == "image":
                        # Transform Anthropic image format to OpenAI image_url format
                        source = raw.get("source", {})
                        if source and source.get("type") == "base64":
                            b64_data = source.get("data", "")
                            media_type = source.get("media_type", "image/png")
                            img_url = f"data:{media_type};base64,{b64_data}"
                            content_parts.append({"type": "image_url", "image_url": {"url": img_url}})
                        elif source and source.get("type") == "url":
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": source["url"]}
                            })
                        else:
                            # Unknown source type - include as-is
                            content_parts.append(raw)
                    elif raw.get("type") == "video":
                        # Transform video to OpenAI media format
                        source = raw.get("source", {})
                        if source and source.get("type") == "base64":
                            b64_data = source.get("data", "")
                            media_type = source.get("media_type", "video/mp4")
                            media_url = f"data:{media_type};base64,{b64_data}"
                            content_parts.append({"type": "media", "source": {"type": "base64", "media_type": media_type, "data": b64_data}})
                        elif source and source.get("type") == "url":
                            content_parts.append({"type": "media", "source": {"type": "url", "url": source["url"]}})
                        else:
                            content_parts.append(raw)
                    elif raw.get("type") == "pdf":
                        # Transform PDF to OpenAI input_file format
                        source = raw.get("source", {})
                        if source and source.get("type") == "base64":
                            b64_data = source.get("data", "")
                            media_type = source.get("media_type", "application/pdf")
                            content_parts.append({
                                "type": "input_file",
                                "file_url": f"data:{media_type};base64,{b64_data}"
                            })
                        elif source and source.get("type") == "url":
                            content_parts.append({
                                "type": "input_file",
                                "file_url": source["url"]
                            })
                        else:
                            content_parts.append(raw)
                    elif raw.get("type") == "thinking":
                        # Translate thinking to OpenAI reasoning field
                        msg_dict["reasoning"] = raw.get("text", "")
                    elif raw.get("type") == "text":
                        # Text content part - apply key translation (text→content)
                        translated = {}
                        for key, value in raw.items():
                            if key in (self._config.request_translations or {}):
                                api_key = self._config.request_translations[key]
                            else:
                                api_key = key
                            if api_key != "type":
                                translated[api_key] = value
                        content_parts.append({"type": "text", **translated})
                    else:
                        # Other content part
                        content_parts.append(raw)

                if content_parts:
                    if len(content_parts) == 1:
                        # Single text part: keep backward-compatible flat format
                        msg_dict["content"] = content_parts[0].get("content", "")
                    else:
                        # Multi-part: use content array
                        msg_dict["content"] = content_parts

                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls

                messages.append(msg_dict)

        body["messages"] = messages

        if tools:
            body["tools"] = tools

        # Copy generation config (includes tool_choice)
        if generation_config:
            for key, value in generation_config.items():
                if key not in body:
                    body[key] = value

        return body


class OpenAIChatBotResponse(GenericChatBotResponse):
    """ChatBotResponse for OpenAI-compatible API.

    Translates OpenAI SSE events to uniform delta format.
    Special handling: OpenAI tool_calls are converted to uniform content array
    with type="tool_use" for backwards compatibility with existing code.

    For non-streaming, overrides from_json to extract fields from the
    complete response format (choices array with message role/content).
    """

    def _finalize(self) -> None:
        """Normalize OpenAI stop_reason to unified values."""
        raw = self._data.get("stop_reason")
        if raw is not None:
            self._data["stop_reason"] = normalize_stop_reason(raw)

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

        if "error" in data:
            response._data["error"] = str(data["error"])
            _logger.error("OpenAI error: %s", data["error"])
            return response

        choices = data.get("choices", [])
        if choices and isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            # Extract role
            if "role" in message:
                response._data["role"] = message["role"]
            # Extract content
            content = message.get("content", "")
            if isinstance(content, str) and content:
                thinking_text, remaining_text = _split_thinking_tokens(content)
                content_parts: list[Dict[str, Any]] = []
                if thinking_text:
                    content_parts.append({"index": 0, "type": "thinking", "content": thinking_text})
                    content_parts.append({"index": 1, "type": "text", "content": remaining_text})
                else:
                    content_parts.append({"index": 0, "type": "text", "content": content})
                response._data["content"] = content_parts
            elif isinstance(content, list):
                response._data["content"] = []
                for item in content:
                    if isinstance(item, dict):
                        transformed = {"index": len(response._data["content"])}
                        # Convert OpenAI fields to unified format
                        for k, v in item.items():
                            if k == "text":
                                transformed["content"] = v
                            elif k == "type":
                                transformed["type"] = v
                            else:
                                transformed[k] = v
                        response._data["content"].append(transformed)
            # Extract tool_calls → content array with tool_use items
            if "tool_calls" in message and message["tool_calls"]:
                for tc in message["tool_calls"]:
                    if isinstance(tc, dict):
                        response._data.setdefault("content", []).append({
                            "index": len(response._data.get("content", [])),
                            "type": "tool_use",
                            "call_id": tc.get("id"),
                            "name": tc.get("function", {}).get("name"),
                            "arguments": tc.get("function", {}).get("arguments", ""),
                        })
            # Extract reasoning (as thinking content block)
            if "reasoning" in message and message["reasoning"]:
                reasoning = message["reasoning"]
                response._data.setdefault("content", []).insert(
                    0, {"index": 0, "type": "thinking", "content": reasoning}
                )

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

        When text and tool_calls collide in the same content item (hybrid),
        split them into separate items with different indices.
        """
        if "content" in translated and not isinstance(translated["content"], list):
            return
        items = translated.get("content", [])
        if not items:
            return

        new_items: list[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                new_items.append(item)
                continue

            # Detect hybrid items: has both text field AND tool-use field
            is_hybrid = (
                "content" in item
                and (
                    ("name" in item and ("arguments" in item or "id" in item))
                    or ("name" in item and "function" in item)
                )
            )
            if not is_hybrid:
                # Pure item — set type normally
                if ("name" in item and ("arguments" in item or "id" in item or len(item) == 1)) or \
                   ("name" in item and "function" in item):
                    item["type"] = "tool_use"
                new_items.append(item)
            else:
                # Hybrid item: split into text + tool_use sub-items
                idx = item.get("index", 0)
                text_item = {"index": idx, "type": "text", "content": item["content"]}
                new_items.append(text_item)
                tool_item = {k: v for k, v in item.items() if k not in ("content",)}
                tool_item["index"] = idx + 1
                tool_item["type"] = "tool_use"
                new_items.append(tool_item)

        translated["content"] = new_items

    @staticmethod
    def _set_text_types(translated: Dict[str, Any]) -> None:
        """Set type="text" on content items without a type field."""
        if "content" in translated and isinstance(translated["content"], list):
            for item in translated["content"]:
                if isinstance(item, dict) and "type" not in item:
                    item["type"] = "text"

    @staticmethod
    def _finalize_reasoning(data: Dict[str, Any]) -> None:
        """Convert accumulated _reasoning to a thinking block in content array."""
        if "_reasoning" not in data:
            return
        reasoning_text = data.pop("_reasoning")
        if not reasoning_text:
            return
        content = data.get("content", [])
        # Insert thinking block at the front (reasoning always comes first)
        content.insert(0, {"type": "thinking", "content": reasoning_text})

    def _process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process an OpenAI event, injecting type discriminators for text/tool_use."""
        translated = self._translate_event(event)
        self._set_tool_call_types(translated)
        return translated

    def _accumulate_event(self, event: Dict[str, Any]) -> None:
        """
        Accumulate translated event into response dict using delta merge.
        Type discriminators are applied after merge to avoid string concatenation.
        """
        merge_delta_into_target(self._data, event)
        # Set type="text" on content items that still lack a type (text blocks)
        if "content" in self._data and isinstance(self._data["content"], list):
            for item in self._data["content"]:
                if isinstance(item, dict) and "type" not in item:
                    item["type"] = "text"

    def _event_generator(self) -> AsyncIterator[tuple[str, Any]]:
        """Yield streamed chunks, then finalize _reasoning → thinking block."""
        gen = super()._event_generator()

        async def _stream_generator():
            async for item in gen:
                yield item
            self._finalize_reasoning(self._data)

        return _stream_generator().__aiter__()

    def _build_message(self) -> Message:
        """Convert OpenAI API-specific accumulated data into a Message."""
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
                elif item_type == "text":
                    content_parts.append(ContentPart.create_text(item.get("content", "")))
                    has_text = True
                elif item_type == "thinking":
                    content_parts.append(ContentPart.create_thinking(item.get("content", "")))
                else:
                    _logger.debug("Unknown OpenAI content part type: %s", item_type)

        self._has_text_part = has_text
        return Message.create(role=role, content_parts=content_parts)
