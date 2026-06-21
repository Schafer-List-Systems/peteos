"""Response classes for ChatBot with streaming support."""

import json
from typing import Any, Dict, AsyncGenerator, AsyncIterator

from peteos.utils.delta_merge import merge_delta_into_target as _merge_delta_into_target
from peteos.utils.delta_merge import translate_delta_event as _translate_delta_event
from peteos.utils import get_logger

_logger = get_logger(__name__)


class ChatBotResponse:
    """Generic response wrapper that accumulates ALL response fields.

    The response is a single dict containing all accumulated values.
    Translation routes source paths to target keys in the dict.

    Usage:
        response = await chatbot.send_message(history)
        async for _ in response:
            pass

        # Access response data as dict
        role = response.data["role"]
        content = response.data["content"]  # list of {type, content/name/arguments}
        stop_reason = response.data.get("stop_reason")

        # Or with __getitem__
        role = response["role"]
        if "images" in response:
            images = response["images"]
    """

    def __init__(self, stream: AsyncGenerator[str, None]):
        """
        Initialize ChatBotResponse.

        Args:
            stream: Async generator of raw SSE lines from HTTPClient.
        """
        self._stream = stream
        self._data: Dict[str, Any] = {}

    @property
    def data(self) -> Dict[str, Any]:
        """Access raw accumulated data dict."""
        return self._data

    def __getitem__(self, key: str) -> Any:
        """Access response fields by key."""
        return self._data.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._data


class GenericChatBotResponse(ChatBotResponse):
    """ChatBotResponse with configurable path-based translations.

    Args:
        stream: Async generator of raw SSE lines from HTTPClient.
        translations: Dict mapping source path -> target field.
            Paths use JSONPath-style notation like "choices[*].delta.content".
            Target keys are used for accumulation in response.data.

    Example:
        response = GenericChatBotResponse(stream, {
            "choices[*].delta.content": "content[0].content",
            "choices[*].delta.reasoning": "content[0].reasoning",
            "choices[*].delta.tool_calls[0].id": "content[0].id"
        })
    """

    def __init__(self, stream: AsyncGenerator[str, None], translations: Dict[str, str]):
        super().__init__(stream)
        self._translations = translations

    @classmethod
    def from_json(cls, data: Dict[str, Any], translations: Dict[str, str]) -> "GenericChatBotResponse":
        """Create a response from a JSON response (for non-streaming mode).

        Args:
            data: Parsed JSON response from API.
            translations: Path translation configuration.

        Returns:
            GenericChatBotResponse with the JSON data translated.
        """
        async def events():
            yield f"data: {json.dumps(data)}"
            yield "[DONE]"
        return cls(events(), translations)

    def _accumulate_event(self, event: Dict[str, Any]) -> None:
        """
        Accumulate translated event into response dict using delta merge.

        Args:
            event: Translated event with target keys (and preserved index fields).
        """
        _merge_delta_into_target(self._data, event)

    def _event_generator(self) -> AsyncIterator[tuple[str, Any]]:
        """
        Async generator that yields (key, chunk) pairs from SSE stream.

        Yields one (key, chunk) per async iteration.
        Multiple key-value pairs from the same SSE event are yielded sequentially.

        Input SSE lines:
            data: {"choices": [{"delta": {"content": "Hello"}}]}
            data: {"type": "message_start", "message": {"content": [...], "reasoning": "R"}}
            error: {"error": "some error message"}

        Output tuples:
            ("text", "Hello")
            ("text", "..."), ("reasoning", "R")  # Same event, multiple keys
        """
        async def _stream_generator():
            async for line in self._stream:
                # Check for [DONE] first (can be "data: [DONE]" or just "[DONE]")
                if line.strip() == "[DONE]" or line == "data: [DONE]\n" or line == "data: [DONE]":
                    _logger.debug("Received [DONE] signal")
                    break
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip():
                        try:
                            raw_event = json.loads(data)
                            _logger.debug("Raw SSE event: %s", raw_event)
                            translated = self._process_event(raw_event)
                            _logger.debug("SSE event translated: %s", translated)
                            for key, chunk in translated.items():
                                if chunk is not None:
                                    self._accumulate_event({key: chunk})
                                    _logger.debug("Accumulated %s: %s", key, chunk)
                                    yield (key, chunk)
                        except json.JSONDecodeError:
                            _logger.warning("Failed to parse SSE event: %s", line.strip())
                elif line.startswith("event: "):
                    # Skip event type markers (e.g., "event: message_start")
                    # The actual data is on the next line starting with "data: "
                    # This is expected behavior for SSE streams with event type markers
                    _logger.debug("Received event marker: %s", line.strip())
                    continue
                elif line.startswith("{") or line.startswith("["):
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict) and "error" in data:
                            self._data["error"] = str(data["error"])
                            _logger.error("Chatbot error: %s", data["error"])
                            continue
                        else:
                            _logger.warning("Unknown line from chatbot: %s", line.strip())
                    except json.JSONDecodeError:
                        _logger.warning("Failed to parse line from chatbot: %s", line.strip())
                else:
                    _logger.warning("Unknown line from chatbot: %s", line.strip())

        return _stream_generator().__aiter__()

    def __aiter__(self) -> AsyncIterator[tuple[str, Any]]:
        """Async iterable that yields translated event key-value pairs."""
        return self._event_generator()

    def _process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an event and return translated fields.

        Override in subclasses to customize event processing logic.
        """
        return self._translate_event(event)

    def _translate_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate an event using configured path translations.

        This uses delta-aware translation that preserves index fields for
        proper delta merging.

        Args:
            event: Raw API response event.

        Returns:
            Translated event with target keys and preserved index fields.
        """
        return _translate_delta_event(event, self._translations)
