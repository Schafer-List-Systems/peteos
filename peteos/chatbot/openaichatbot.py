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

_logger = get_logger(__name__)


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
        "choices[*].delta.reasoning": "_reasoning",
        "choices[*].delta.content": "content[0].content",
        "choices[*].delta.finish_reason": "stop_reason",
        # Tool calls: translate to uniform content array format
        "choices[*].delta.tool_calls[0].index": "content[0].index",
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
        self._models: List[str] = []

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
            stream = self._http_client.stream_post(f"{self._config.url}{self._config.chat_endpoint}", body)
            return OpenAIChatBotResponse(stream, self._config.response_translations or {})
        else:
            response_data = await self._http_client.post(f"{self._config.url}{self._config.chat_endpoint}", body)
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
                            "name": part.name or "unknown",
                            "content": part.content or ""
                        }
                        messages.append(msg_dict)
                        break
            elif role in ("user", "assistant", "system"):
                # Conversation messages
                msg_dict = {"role": role}
                tool_calls = []
                content_parts: List[Dict[str, Any]] = []

                for part in msg.content:
                    raw = part.raw_dict
                    if raw.get("type") == "tool_calls":
                        for tool_call in raw.get("tool_calls", []):
                            tool_calls.append({
                                "type": "function",
                                "id": tool_call.get("id"),
                                "function": {
                                    "name": tool_call.get("name"),
                                    "arguments": tool_call.get("arguments", "")
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
                    else:
                        # Text or other content part - apply key translation
                        translated = {}
                        for key, value in raw.items():
                            if key in (self._config.request_translations or {}):
                                api_key = self._config.request_translations[key]
                            else:
                                api_key = key
                            if api_key != "type":
                                translated[api_key] = value
                        content_parts.append({"type": "text", **translated})

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
                            "id": tc.get("id"),
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
        """
        if "content" in translated and isinstance(translated["content"], list):
            for item in translated["content"]:
                if isinstance(item, dict):
                    if ("name" in item and ("arguments" in item or "id" in item or len(item) == 1)) or \
                       ("name" in item and "function" in item):
                        item["type"] = "tool_use"

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
