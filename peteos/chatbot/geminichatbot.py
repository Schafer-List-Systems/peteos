"""Google Gemini ChatBot implementation."""

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, AsyncGenerator, AsyncIterator

from peteos.utils import get_logger
from .chatbot import ChatBot
from .chatbotconfig import ChatBotConfig
from .chatbotresponse import ChatBotResponse, GenericChatBotResponse
from .httpclient import HTTPClient
from peteos.conversation.context import Context
from peteos.utils.delta_merge import merge_delta_into_target
from peteos.conversation.message import Message, ContentPart

_logger = get_logger(__name__)


class GeminiChatBot(ChatBot):
    """ChatBot implementation for Google Gemini API.

    Uses the Gemini REST API directly:
    - Chat endpoint: /v1beta/models/{model}:generateContent
    - Streaming: /v1beta/models/{model}:streamGenerateContent
    - Auth: x-goog-api-key header (set via api_key in config)

    Gemini-specific request/response handling is entirely contained here.
    """

    DEFAULT_CHAT_ENDPOINT = "/v1beta/models/"
    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, http_client: HTTPClient, config: ChatBotConfig):
        bot_cfg = asdict(config)
        if bot_cfg.get("chat_endpoint") is None:
            # chat_endpoint becomes the model path prefix (e.g. /v1beta/models/{model}:generateContent)
            bot_cfg["chat_endpoint"] = self.DEFAULT_CHAT_ENDPOINT
        if bot_cfg.get("max_tokens") is None:
            bot_cfg["max_tokens"] = self.DEFAULT_MAX_TOKENS
        super().__init__(ChatBotConfig.from_dict(bot_cfg))
        self._models: List[str] = []

        # Capture api_key locally and build executors — api_key is then
        # captured in the closure and removed from config.
        api_key = self._config.api_key
        secure_headers: Dict[str, str] = {}
        if api_key:
            secure_headers["x-goog-api-key"] = api_key
        base_url = f"{self._config.url}{self._config.chat_endpoint}{self._config.model}"
        self._post_executor = self._build_post_executor(http_client, secure_headers, f"{base_url}:generateContent")
        self._stream_executor = self._build_stream_executor(http_client, secure_headers, f"{base_url}:streamGenerateContent")
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
        """Send a context to the Gemini API and receive a response."""
        streaming_mode = self._config.streaming if streaming is None else streaming
        body = self._build_body(context, generation_config, streaming)

        _logger.debug("Gemini request: model=%s, contents=%d, tools=%d",
                       self._config.model, len(body.get("contents", [])), len(body.get("tools", [])))

        # Debug: dump model parts to trace thought_signature round-trip
        for ci, c in enumerate(body.get("contents", [])):
            for pi, part in enumerate(c.get("parts", [])):
                has_ts = "thought_signature" in part
                key = json.dumps({k: part[k] for k in part if k != "thought_signature"}, ensure_ascii=False, default=str)
                _logger.debug("Gemini request contents[%d] parts[%d] role=%s key=%s has_thought_signature=%s", ci, pi, c.get("role"), key[:120], has_ts)

        if streaming_mode:
            stream = self._stream_executor(body, self.get_headers())
            return GeminiChatBotResponse(stream, {})
        else:
            response_data = await self._post_executor(body, self.get_headers())
            return GeminiChatBotResponse.from_json(response_data)

    @staticmethod
    def _translate_tool_params_to_gemini(raw_params: dict) -> dict:
        """Translate internal tool parameters to Gemini JSON Schema format.

        Internal format: {param_name: {type, required, default}, ...}
        Gemini format: {"type": "object", "properties": {...}, "required": [...]}

        Args:
            raw_params: Tool parameters from internal format

        Returns:
            Gemini-compatible JSON Schema
        """
        required_params = []
        properties = {}

        type_map = {
            "int": "INTEGER",
            "str": "STRING",
            "float": "NUMBER",
            "bool": "BOOLEAN",
            "list": "ARRAY",
            "dict": "OBJECT",
            "any": "STRING",
        }

        for param_name, param_info in raw_params.items():
            if isinstance(param_info, str):
                json_type = type_map.get(param_info, "STRING")
                prop = {"type": json_type}
                required_params.append(param_name)
            else:
                py_type = param_info.get("type", "string")
                json_type = type_map.get(py_type, "STRING")

                prop = {"type": json_type}

                if "required" in param_info:
                    if param_info["required"]:
                        required_params.append(param_name)
                else:
                    required_params.append(param_name)

                if "default" in param_info:
                    prop["description"] = str(param_info["default"])

                if py_type == "list":
                    prop["items"] = {"type": "STRING"}

            properties[param_name] = prop

        return {
            "type": "OBJECT",
            "properties": properties,
            "required": required_params,
        }

    @staticmethod
    def _build_gemini_content_part(part: ContentPart) -> Dict[str, Any]:
        """Convert an internal ContentPart to Gemini's part format."""
        raw = part.raw_dict
        part_type = raw.get("type", "")

        if part_type == "text":
            return {"text": part.text}

        if part_type == "thinking":
            return {"text": part.text}

        if part_type == "image":
            source = raw.get("source", {})
            if source.get("type") == "base64":
                return {
                    "inline_data": {
                        "mime_type": source.get("media_type", "image/png"),
                        "data": source.get("data", ""),
                    }
                }
            # URL-based images: Gemini doesn't support URL-based images directly
            # Fall back to text warning
            return {"text": f"[Image from URL: {source.get('url', 'unknown')}]" }

        if part_type == "video":
            source = raw.get("source", {})
            if source.get("type") == "base64":
                return {
                    "inline_data": {
                        "mime_type": source.get("media_type", "video/mp4"),
                        "data": source.get("data", ""),
                    }
                }
            return {"text": f"[Video from URL: {source.get('url', 'unknown')}]" }

        if part_type == "audio":
            source = raw.get("source", {})
            if source.get("type") == "base64":
                return {
                    "inline_data": {
                        "mime_type": source.get("media_type", "audio/mpeg"),
                        "data": source.get("data", ""),
                    }
                }
            return {"text": f"[Audio from URL: {source.get('url', 'unknown')}]" }

        if part_type == "pdf":
            source = raw.get("source", {})
            if source.get("type") == "base64":
                return {
                    "inline_data": {
                        "mime_type": source.get("media_type", "application/pdf"),
                        "data": source.get("data", ""),
                    }
                }
            return {"text": f"[PDF from URL: {source.get('url', 'unknown')}]" }

        if part_type in ("tool_use", "tool_result", "tool"):
            # Tool definition parts are handled separately
            return {"text": ""}

        return {"text": str(raw)}

    def _build_body(self, context: Context, generation_config: Optional[Dict[str, Any]] = None,
                     streaming: bool | None = None) -> Dict[str, Any]:
        """Build Gemini-specific request body.

        Gemini format:
        - system_instruction at top level (separate from contents)
        - contents[] with role "user" and "model" (not "assistant")
        - tools with functionDeclarations
        - inline_data for images/videos/audio (base64)
        """
        body: Dict[str, Any] = {}
        body["generationConfig"] = {
            "maxOutputTokens": self._config.max_tokens,
        }

        # Gemini's generation config field names differ from OpenAI's
        _GEMINI_GEN_CONFIG_FIELDS = {
            "temperature": "temperature",
            "max_tokens": "maxOutputTokens",
            "top_p": "topP",
            "top_k": "topK",
            "stop_sequences": "stopSequences",
        }

        if generation_config:
            for key, value in generation_config.items():
                if key not in body.get("generationConfig", {}):
                    gemini_key = _GEMINI_GEN_CONFIG_FIELDS.get(key)
                    if gemini_key:
                        body["generationConfig"][gemini_key] = value
                    elif key == "response_format":
                        # Handle response_format if present
                        body["generationConfig"]["response_mime_type"] = value.get("type", "text")
                    elif key == "tool_choice":
                        # Tool choice handled separately below
                        pass

        # Extract system prompt
        system_parts = []
        contents = []
        tools = []

        # Mapping of call_id -> function name from the most recent model message.
        # Tool results look up their corresponding function by call_id.
        _fc_by_call_id: dict[str, str] = {}

        for msg in context.messages:
            role = msg.role

            if role == "system":
                system_parts.extend(part.raw_dict for part in msg.content)
            elif role == "tool":
                # Tool definitions
                for part in msg.content:
                    if part.type == "tool":
                        raw = part.raw_dict
                        params = self._translate_tool_params_to_gemini(
                            raw.get("parameters", {})
                        )
                        func_decl = {
                            "name": raw.get("name"),
                            "description": raw.get("description", ""),
                            "parameters": params,
                        }
                        tools.append({"functionDeclarations": [func_decl]})
            elif role == "user":
                # Conversation message — maps to Gemini role "user"
                parts = self._build_gemini_parts(msg.content)
                if parts:
                    contents.append({"role": "user", "parts": parts})
            elif role == "assistant":
                # Gemini has no "assistant" role — map to "model"
                # Extract call_id → name from ContentParts before serialization
                # (call_id is stored in the framework ContentPart, not in the Gemini
                # functionCall — Gemini rejects unknown keys in functionCall objects)
                _fc_by_call_id.clear()
                for part in msg.content:
                    if part.type == "tool_use" and part.call_id:
                        _fc_by_call_id[part.call_id] = part.name or ""
                parts = self._build_gemini_parts(msg.content)
                if parts:
                    contents.append({"role": "model", "parts": parts})
            elif role == "tool_result":
                # Tool result — look up function name by call_id, add to user parts
                response_parts = []
                for part in msg.content:
                    if part.type == "tool_result":
                        call_id = part.call_id or ""
                        name = _fc_by_call_id.get(call_id)
                        if name is None:
                            _logger.warning(
                                "Gemini: tool_result with call_id=%r has no matching "
                                "functionCall in the preceding model message. Skipping.",
                                call_id,
                            )
                            continue
                        response_parts.append({
                            "functionResponse": {
                                "name": name,
                                "response": {"content": part.content or ""},
                            }
                        })
                if response_parts:
                    # Merge with most recent user message to avoid consecutive user messages
                    if contents and contents[-1]["role"] == "user":
                        contents[-1]["parts"].extend(response_parts)
                    else:
                        contents.append({"role": "user", "parts": response_parts})

        if system_parts:
            system_text = " ".join(
                item.get("text", "") for item in system_parts
                if item.get("type") in ("text", "thinking") and item.get("text")
            )
            if system_text:
                body["system_instruction"] = {"role": "system", "parts": [{"text": system_text}]}

        if contents:
            body["contents"] = contents

        if tools:
            body["tools"] = tools

        # Handle tool_choice in generationConfig
        if generation_config:
            tool_choice = generation_config.get("tool_choice")
            if tool_choice:
                if isinstance(tool_choice, dict):
                    tc_type = tool_choice.get("type", "auto")
                    if tc_type == "any":
                        body["generationConfig"]["tool_config"] = {
                            "any_function_config": {}
                        }
                    elif tc_type == "specific":
                        body["generationConfig"]["tool_config"] = {
                            "function_calling_config": {
                                "mode": "ANY",
                                "allowed_function_names": [tool_choice.get("name", "")],
                            }
                        }
                    else:
                        body["generationConfig"]["tool_config"] = {
                            "function_calling_config": {
                                "mode": "AUTO" if tc_type == "auto" else tc_type.upper(),
                            }
                        }
                elif isinstance(tool_choice, str):
                    body["generationConfig"]["tool_config"] = {
                        "function_calling_config": {
                            "mode": tool_choice.upper(),
                        }
                    }

        return body

    def _build_gemini_parts(self, content_parts: list) -> list[dict]:
        """Build a list of Gemini format parts from message content parts.

        Converts tool_use/tool_calls into functionCall parts. call_id is
        extracted from the framework ContentPart for internal matching but
        NOT included in the functionCall dict (Gemini rejects unknown keys).
        """
        parts: list[dict] = []
        for part in content_parts:
            raw = part.raw_dict
            part_type = raw.get("type")
            if part_type == "tool_calls":
                for tc in raw.get("tool_calls", []):
                    fc_part: dict[str, Any] = {
                        "name": tc.get("name"),
                        "args": json.loads(tc.get("arguments", "{}")),
                    }
                    part_obj: dict[str, Any] = {"functionCall": fc_part}
                    ts = tc.get("thought_signature")
                    if ts:
                        part_obj["thought_signature"] = ts
                    parts.append(part_obj)
            elif part_type == "tool_use":
                fc_part: dict[str, Any] = {
                    "name": raw.get("name"),
                    "args": json.loads(raw.get("arguments", "{}")),
                }
                part_obj: dict[str, Any] = {"functionCall": fc_part}
                ts = raw.get("thought_signature")
                if ts:
                    part_obj["thought_signature"] = ts
                parts.append(part_obj)
            else:
                part_data = self._build_gemini_content_part(part)
                if part_data:
                    parts.append(part_data)
        return parts


class GeminiChatBotResponse(GenericChatBotResponse):
    """ChatBotResponse for Google Gemini API.

    Follows the OpenAI/Anthropic pattern: inherits GenericChatBotResponse,
    overrides _process_event and _accumulate_event for Gemini-specific handling.

    Gemini streaming sends full accumulated state per event (not deltas).
    Streaming event format:
      data: {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}

    Non-streaming returns the complete response directly.
    """

    # Gemini uses full-state events: each event contains the complete
    # accumulated response. Override _accumulate_event to replace instead of merge.
    def _accumulate_event(self, event: Dict[str, Any]) -> None:
        """Replace fields from full-state Gemini events instead of merging deltas."""
        for key, value in event.items():
            if value is not None:
                if key == "content" and isinstance(value, list):
                    # Full state: replace content array entirely
                    self._data["content"] = value
                else:
                    # Simple assignment for role, stop_reason, etc.
                    self._data[key] = value

    def _process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Translate Gemini SSE event to uniform format."""
        # Check for error in event
        if isinstance(event, dict) and "error" in event:
            self._data["error"] = str(event["error"])
            _logger.error("Gemini error: %s", event["error"])
            return {}
        # Handle list (e.g., error wrapped in array)
        if isinstance(event, list):
            for item in event:
                if isinstance(item, dict) and "error" in item:
                    self._data["error"] = str(item["error"])
                    _logger.error("Gemini error: %s", item["error"])
                    return {}
        translated = self._extract_from_response(event)
        self._set_text_types(translated)
        return translated

    def _event_generator(self) -> AsyncIterator[tuple[str, Any]]:
        """Stream generator that reads Gemini's full-state streaming events.

        Gemini streaming sends a JSON array of events:
          [{event1}, {event2}, ...]
        Each event contains the full accumulated state (not deltas).
        """
        async def _stream_generator():
            # Collect all lines from the stream
            all_lines = []
            async for line in self._stream:
                stripped = line.strip()
                if stripped == "[DONE]" or stripped == "" or line.startswith("data: "):
                    # Skip SSE artifacts; the actual JSON is in the raw stream
                    continue
                all_lines.append(line)

            # Join and parse as JSON (could be array or single object)
            combined = "".join(all_lines).strip()
            if not combined:
                return

            try:
                events = json.loads(combined)
                if isinstance(events, list):
                    event_iter = iter(events)
                elif isinstance(events, dict):
                    event_iter = iter([events])
                else:
                    _logger.warning("Unexpected Gemini stream type: %s", type(events).__name__)
                    return

                for raw_event in event_iter:
                    for key, chunk in self._process_event(raw_event).items():
                        if chunk is not None:
                            self._accumulate_event({key: chunk})
                            yield (key, chunk)
                    # Check for [DONE] embedded in event data
                    if isinstance(raw_event, dict) and "finishReason" in raw_event:
                        fr = raw_event["finishReason"]
                        if fr in ("STOP", "MAX_TOKENS"):
                            break
            except json.JSONDecodeError:
                # Check if the response is an error dict in non-stream format
                # (e.g., Gemini API returning a raw error response instead of a stream)
                try:
                    error_body = json.loads(combined)
                    if isinstance(error_body, dict) and (
                        "error" in error_body or "code" in error_body or "status" in error_body
                    ):
                        self._data["error"] = str(error_body.get("error", error_body))
                        _logger.error(
                            "Gemini returned error response (non-stream format): %s",
                            combined[:300],
                        )
                        return
                except (json.JSONDecodeError, ValueError):
                    pass
                # Truly unparseable — raise to terminate the stream
                _logger.error(
                    "Gemini stream returned unparseable response: %s", combined[:200]
                )
                raise RuntimeError(
                    f"Gemini stream returned unparseable response: {combined[:200]}"
                )

        return _stream_generator().__aiter__()

    @staticmethod
    def _extract_from_response(data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract role, content, and stop_reason from a Gemini response event."""
        result: Dict[str, Any] = {}
        # Handle case where parsed JSON is a list (e.g., error wrapped in array)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "error" in item:
                    result["error"] = str(item["error"])
                    _logger.error("Gemini error: %s", item["error"])
                    return result
            return result
        if "error" in data:
            result["error"] = str(data["error"])
            _logger.error("Gemini error: %s", data["error"])
            return result
        candidates = data.get("candidates", [])
        if not candidates:
            return result

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # Extract role
        result["role"] = content.get("role", "model")

        # Build uniform content array
        content_array = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if "text" in part:
                content_array.append({
                    "index": len(content_array),
                    "type": "text",
                    "content": part["text"],
                })
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_use: Dict[str, Any] = {
                    "index": len(content_array),
                    "type": "tool_use",
                    "name": fc.get("name", ""),
                    "arguments": json.dumps(fc.get("args", {})),
                }
                # Gemini may return thought_signature in either snake_case or camelCase
                ts = part.get("thought_signature") or part.get("thoughtSignature")
                if ts:
                    tool_use["thought_signature"] = ts
                content_array.append(tool_use)
            elif "inline_data" in part:
                _logger.warning("Gemini response contains inline_data — not directly supported")
                content_array.append({
                    "index": len(content_array),
                    "type": "text",
                    "content": f"[Media: {part['inline_data'].get('mime_type', 'unknown')}]",
                })

        if content_array:
            result["content"] = content_array

        # Extract finish reason → stop_reason
        finish_reason = candidate.get("finishReason", "")
        if finish_reason:
            if finish_reason in ("STOP", "MAX_TOKENS"):
                result["stop_reason"] = "stop"
            elif finish_reason == "SAFETY":
                result["stop_reason"] = "safety"
            else:
                result["stop_reason"] = finish_reason.lower()

        return result

    @staticmethod
    def _set_text_types(translated: Dict[str, Any]) -> None:
        """Set type='text' on content items without a type field."""
        if "content" in translated and isinstance(translated["content"], list):
            for item in translated["content"]:
                if isinstance(item, dict) and "type" not in item:
                    item["type"] = "text"

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "GeminiChatBotResponse":
        """Create a response from a complete Gemini non-streaming response."""
        response = cls(augmented_yield(), {})
        response._data = {}
        translated = cls._extract_from_response(data)
        response._data.update(translated)
        return response

    def _build_message(self) -> Message:
        """Convert Gemini API-specific accumulated data into a Message."""
        # Map Gemini's "model" role to framework's "assistant" role
        grole: str = self._data.get("role", "model")
        role: str = "assistant" if grole == "model" else grole
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
                    text = item.get("content", "")
                    if text:
                        content_parts.append(ContentPart.create_text(text))
                        has_text = True
                else:
                    _logger.debug("Unknown Gemini content part type: %s", item_type)

        msg = Message.create(role=role, content_parts=content_parts)

        # Assign deterministic call_ids derived from the message's own UUID
        for idx, part in enumerate(msg.content):
            if part.type == "tool_use":
                part.raw_dict["call_id"] = f"msg-{msg.id}-{idx}"

        self._has_text_part = has_text
        return msg


async def augmented_yield():
    """Yields nothing - GeminiChatBotResponse.from_json populates _data directly."""
    return
    yield
