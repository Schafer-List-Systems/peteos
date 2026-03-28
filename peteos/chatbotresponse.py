"""Response classes for ChatBot with streaming support."""

import json
from typing import AsyncGenerator, AsyncIterator, Dict, Any, List

from peteos.utils import get_value_at_path as _get_value_at_path


class ChatBotResponse:
    """Generic response wrapper that accumulates ALL response fields.

    The response is a single dict containing all accumulated values.
    Translation routes source paths to target keys in the dict.

    Usage:
        response = await chatbot.send_message(history)
        async for _ in response:
            pass

        # Access response data as dict
        text = response.data["text"]
        reasoning = response.data["reasoning"]
        tool_calls = response.data["tool_calls"]

        # Or with __getitem__
        text = response["text"]
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
            "choices[*].delta.content": "text",
            "choices[*].delta.reasoning": "reasoning",
            "choices[*].delta.tool_calls": "tool_calls"
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
        Accumulate translated event into response dict.

        Args:
            event: Translated event with target keys.
        """
        for key, value in event.items():
            if value is not None:
                existing = self._data.get(key, "")
                if isinstance(existing, str):
                    self._data[key] = str(existing) + str(value)
                elif isinstance(existing, list):
                    self._data[key].append(value)
                elif isinstance(existing, dict):
                    self._data[key].update(value)
                else:
                    self._data[key] = value

    def _event_generator(self) -> AsyncIterator[tuple[str, Any]]:
        """
        Async generator that yields (key, chunk) pairs from SSE stream.

        Yields one (key, chunk) per async iteration.
        Multiple key-value pairs from the same SSE event are yielded sequentially.

        Input SSE lines:
            data: {"choices": [{"delta": {"content": "Hello"}}]}
            data: {"type": "message_start", "message": {"content": [...], "reasoning": "R"}}

        Output tuples:
            ("text", "Hello")
            ("text", "..."), ("reasoning", "R")  # Same event, multiple keys
        """
        async def _stream_generator():
            async for line in self._stream:
                if line.startswith("data: ") and line.strip() != "[DONE]":
                    data = line[6:]
                    if data.strip():
                        try:
                            event = json.loads(data)
                            translated = await self._translate_event(event)
                            for key, value in translated.items():
                                if value is not None:
                                    self._accumulate_event({key: value})
                                    yield (key, value)
                        except json.JSONDecodeError:
                            pass

        return _stream_generator().__aiter__()

    def __aiter__(self) -> AsyncIterator[tuple[str, Any]]:
        """Async iterable that yields translated event key-value pairs."""
        return self._event_generator()

    async def _translate_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate an event using configured path translations.

        Args:
            event: Raw API response event.

        Returns:
            Translated event with target keys.
        """
        translated: Dict[str, Any] = {}

        for path, target in self._translations.items():
            value = _get_value_at_path(event, path)
            if value is not None:
                # Handle array values from wildcard paths
                if isinstance(value, list):
                    value = "".join(str(v) for v in value if v)
                else:
                    value = str(value)
                translated[target] = value

        return translated


class OpenAIChatBotResponse(GenericChatBotResponse):
    """Response wrapper for OpenAI API using standard translations."""

    def __init__(self, stream: AsyncGenerator[str, None]):
        super().__init__(stream, {
            "choices[*].delta.content": "text",
            "choices[*].delta.reasoning": "reasoning",
            "choices[*].delta.thinking": "reasoning",
            "choices[*].message.content": "text",
            "choices[*].message.reasoning": "reasoning",
            "choices[*].message.thinking": "reasoning",
            "choices[*].delta.tool_calls": "tool_calls",
            "choices[*].message.tool_calls": "tool_calls",
        })


class AnthropicChatBotResponse(GenericChatBotResponse):
    """Response wrapper for Anthropic API using standard translations."""

    def __init__(self, stream: AsyncGenerator[str, None]):
        super().__init__(stream, {
            "content_block_delta.delta.text": "text",
            "content_block_delta.delta.reasoning": "reasoning",
            "content_block_delta.delta.thinking": "reasoning",
            "content_block_start.content_block.text": "text",
            "content_block_start.content_block.reasoning": "reasoning",
            "content_block_start.content_block.thinking": "reasoning",
            "message_start.message.content[*].text": "text",
            "message_start.message.reasoning": "reasoning",
            "message_start.message.thinking": "reasoning",
            "content[*].text": "text",
            "content[*].reasoning": "reasoning",
            "content[*].thinking": "reasoning",
        })
