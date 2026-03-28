"""Response classes for ChatBot with streaming support."""

import json
from abc import ABC, abstractmethod
from typing import AsyncIterator, AsyncGenerator, Dict, Any, List, Optional

from peteos.utils import get_value_at_path as _get_value_at_path


class ChatBotResponse(ABC):
    """Abstract base class for chatbot responses.

    Provides a unified interface for streaming responses from different
    LLM APIs. Internally handles SSE parsing and schema translation.

    Usage:
        async for chunk in response:
            print(chunk)  # yields accumulated text as it arrives

        # Access full results
        print(response.thinking_content)
        print(response.text_content)
    """

    def __init__(self, stream: AsyncGenerator[str, None]):
        """
        Initialize ChatBotResponse.

        Args:
            stream: Async generator of raw SSE lines from HTTPClient.
        """
        self._stream = stream
        self._thinking_content: str = ""
        self._text_content: str = ""

    def _accumulate_content(self, event: Dict[str, Any]) -> None:
        """
        Accumulate all fields from a translated event.

        Args:
            event: Translated event with any number of fields.
        """
        for key, value in event.items():
            if value:
                # Route to appropriate accumulator based on field name
                if "thinking" in key.lower():
                    self._thinking_content += str(value)
                else:
                    self._text_content += str(value)

    @abstractmethod
    async def _translate_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate an API-specific event to the common schema.

        Args:
            event: Raw API response event.

        Returns:
            Translated event with common schema:
            - "content": str - response text
            - "thinking": str (optional) - thinking/reasoning content
        """
        pass

    @property
    def thinking_content(self) -> str:
        """Extracted thinking/reasoning content."""
        return self._thinking_content

    @property
    def text_content(self) -> str:
        """Extracted response text content."""
        return self._text_content

    def __aiter__(self) -> "ChatBotResponse":
        """Async iterable that yields accumulated text content."""
        self._lines_iterator = self._stream.__aiter__()
        self._last_chunk = ""
        return self

    async def __anext__(self) -> str:
        """
        Get next accumulated text chunk.

        Returns:
            Accumulated text content after processing next event.

        Raises:
            StopAsyncIteration: When stream is exhausted.
        """
        try:
            line = await self._lines_iterator.__anext__()
            if line.startswith("data: "):
                data = line[6:]
                if data.strip():
                    try:
                        event = json.loads(data)
                        translated = await self._translate_event(event)
                        self._accumulate_content(translated)
                        self._last_chunk = self._text_content
                        return self._last_chunk
                    except json.JSONDecodeError:
                        return await self.__anext__()
            elif line.strip() == "[DONE]":
                raise StopAsyncIteration
            else:
                return await self.__anext__()

        except StopAsyncIteration:
            raise


class GenericChatBotResponse(ChatBotResponse):
    """ChatBotResponse with configurable path-based translations.

    Args:
        stream: Async generator of raw SSE lines from HTTPClient.
        translations: Dict mapping source path -> target field.
            Paths use JSONPath-style notation like "choices[*].delta.content".
            Valid target fields are "content" and "thinking".
            If no translation matches, the event is passed through silently.

    Example:
        response = GenericChatBotResponse(stream, {
            "choices[*].delta.content": "text_content",
            "choices[*].delta.reasoning": "thinking_content"
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

    async def _translate_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate an event using configured path translations.

        Args:
            event: Raw API response event.

        Returns:
            Translated event with configurable fields.
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
            "choices[*].delta.content": "text_content",
            "choices[*].delta.reasoning": "thinking_content",
            "choices[*].delta.thinking": "thinking_content",
        })


class AnthropicChatBotResponse(GenericChatBotResponse):
    """Response wrapper for Anthropic API using standard translations."""

    def __init__(self, stream: AsyncGenerator[str, None]):
        super().__init__(stream, {
            "content_block_delta.delta.text": "text_content",
            "content_block_delta.delta.reasoning": "thinking_content",
            "content_block_delta.delta.thinking": "thinking_content",
            "content_block_start.content_block.text": "text_content",
            "content_block_start.content_block.reasoning": "thinking_content",
            "content_block_start.content_block.thinking": "thinking_content",
            "message_start.message.content[*].text": "text_content",
            "message_start.message.reasoning": "thinking_content",
            "message_start.message.thinking": "thinking_content",
        })
