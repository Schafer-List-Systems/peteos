"""Response classes for ChatBot with streaming support."""

import json
from abc import ABC, abstractmethod
from typing import AsyncIterator, AsyncGenerator, Dict, Any, List, Optional


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
        Accumulate content from a translated event.

        Args:
            event: Event with common schema (has 'content' key).
        """
        content = event.get("content", "")
        if content:
            self._text_content += content

        thinking = event.get("thinking", "")
        if thinking:
            self._thinking_content += thinking

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


class OpenAIChatBotResponse(ChatBotResponse):
    """Response wrapper for OpenAI API.

    Handles both:
    - content -> text_content
    - thinking -> thinking_content
    - reasoning -> thinking_content (for Qwen-style models)
    """

    async def _translate_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate OpenAI-compatible event to common schema.

        OpenAI format: {"choices": [{"delta": {"content": "...", "reasoning": "..."}}]}
        Translates: content -> content, reasoning/thinking -> thinking
        """
        translated: Dict[str, Any] = {"content": "", "thinking": ""}
        choices = event.get("choices", [])
        if choices and "delta" in choices[0]:
            delta = choices[0]["delta"]
            translated["content"] = delta.get("content", "")
            # Handle both "reasoning" (Qwen) and "thinking" keys
            translated["thinking"] = delta.get("reasoning", "") or delta.get("thinking", "")
        return translated


class AnthropicChatBotResponse(ChatBotResponse):
    """Response wrapper for Anthropic API.

    Handles both:
    - text / text_delta -> content
    - reasoning / reasoning_delta -> thinking (original Anthropic)
    - thinking / thinking_delta -> thinking (Anthropic-compatible)
    """

    async def _translate_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate Anthropic event to common schema.

        Anthropic format variations:
        - {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "..."}}
        - {"type": "content_block_delta", "delta": {"type": "reasoning_delta", "reasoning": "..."}}
        - {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "..."}}
        - {"type": "message_start", "message": {"content": [...], "reasoning": {...}}}
        - {"type": "content_block_start", "content_block": {"type": "text", "reasoning": "..."}}
        - {"type": "content_block_start", "content_block": {"type": "thinking", "thinking": "..."}}

        Translates: reasoning/thinking -> thinking, text -> content
        """
        translated: Dict[str, Any] = {"content": "", "thinking": ""}
        event_type = event.get("type", "")

        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                translated["content"] = delta.get("text", "")
            elif delta.get("type") == "reasoning_delta":
                # Handle incremental reasoning streaming (original Anthropic)
                translated["thinking"] = delta.get("reasoning", "")
            elif delta.get("type") == "thinking_delta":
                # Handle incremental thinking streaming (Anthropic-compatible)
                translated["thinking"] = delta.get("thinking", "")

        elif event_type == "content_block_start":
            # Extract reasoning/thinking from content_block if present
            content_block = event.get("content_block", {})
            if "reasoning" in content_block:
                translated["thinking"] = content_block["reasoning"]
            if "thinking" in content_block:
                translated["thinking"] = content_block["thinking"]
            if "text" in content_block:
                translated["content"] = content_block["text"]

        elif event_type == "message_start":
            message = event.get("message", {})
            # Handle nested content array
            for block in message.get("content", []):
                if block.get("type") == "text":
                    translated["content"] = block.get("text", "")
            # Also check for top-level reasoning/thinking key in message
            if "reasoning" in message:
                translated["thinking"] = message["reasoning"]
            if "thinking" in message:
                translated["thinking"] = message["thinking"]

        return translated
