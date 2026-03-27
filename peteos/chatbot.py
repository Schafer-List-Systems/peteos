"""Abstract ChatBot base class and implementations."""

from abc import ABC, abstractmethod
import json
from typing import Dict, Any, List, Optional, AsyncGenerator

from peteos.httpclient import HTTPClient
from peteos.chatbotresponse import ChatBotResponse, OpenAIChatBotResponse, AnthropicChatBotResponse
from peteos.chathistory import ChatHistory
from peteos.message import Message


class ChatBot(ABC):
    """Abstract base class for chatbot implementations.

    Subclasses implement specific LLM providers (OpenAI, Anthropic, etc.)
    and translate their response schemas into a common interface.
    """

    def __init__(self, http_client: HTTPClient, model: str):
        """
        Initialize ChatBot.

        Args:
            http_client: HTTP client for making API requests.
            model: The model identifier to use.
        """
        self._http_client = http_client
        self._model = model

    @abstractmethod
    async def send_message(
        self,
        chat_history: ChatHistory,
        streaming: bool = True
    ) -> ChatBotResponse:
        """
        Send a chat history to the LLM and receive a response.

        Args:
            chat_history: The ChatHistory to send to the LLM.
            streaming: If True, returns a streaming response that yields
                      accumulated text as it arrives. If False, returns
                      the complete response at once.

        Returns:
            A ChatBotResponse that can be iterated to receive the response.
        """
        pass

    @abstractmethod
    def list_available_models(self) -> List[str]:
        """
        List all available models from the LLM provider.

        Returns:
            A list of model identifiers.
        """
        pass

    @property
    def model(self) -> str:
        """Get the current model identifier."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Set a new model identifier."""
        self._model = value

    def _build_request_body(self, chat_history: ChatHistory) -> Dict[str, Any]:
        """
        Build the request body from chat history.

        Subclasses should override this to format messages according
        to their specific API requirements.

        Args:
            chat_history: The chat history to convert.

        Returns:
            Request body dictionary.
        """
        messages = []
        for msg in chat_history.messages:
            content = msg.content.get("content", "")
            role = msg.content.get("role", "user")
            messages.append({"role": role, "content": content})

        return {"messages": messages, "model": self._model}


class OpenAIChatBot(ChatBot):
    """ChatBot implementation for OpenAI API."""

    def __init__(self, http_client: HTTPClient, model: str, base_url: str):
        """
        Initialize OpenAIChatBot.

        Args:
            http_client: HTTP client for making API requests.
            model: The OpenAI model identifier (e.g., "gpt-4").
            base_url: The OpenAI API base URL.
        """
        super().__init__(http_client, model)
        self._base_url = base_url

    async def send_message(
        self,
        chat_history: ChatHistory,
        streaming: bool = True
    ) -> ChatBotResponse:
        """
        Send a chat history to OpenAI and receive a response.

        Args:
            chat_history: The ChatHistory to send.
            streaming: If True, returns streaming response.

        Returns:
            OpenAIChatBotResponse (inherits from ChatBotResponse).
        """
        body = self._build_request_body(chat_history)
        body["stream"] = streaming if streaming else False

        if streaming:
            stream = self._http_client.stream_post(f"{self._base_url}/v1/chat/completions", body)
            return OpenAIChatBotResponse(stream)
        else:
            response_data = await self._http_client.post(f"{self._base_url}/v1/chat/completions", body)
            # For non-streaming, convert to a simple stream of one event
            async def events():
                choices = response_data.get("choices", [])
                if choices and "message" in choices[0]:
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': choices[0]['message']['content']}}]})}"
                    yield "[DONE]"
            return OpenAIChatBotResponse(events())

    def list_available_models(self) -> List[str]:
        """List OpenAI models."""
        return [self._model]  # Simplified - would fetch from API in real implementation


class AnthropicChatBot(ChatBot):
    """ChatBot implementation for Anthropic API."""

    def __init__(self, http_client: HTTPClient, model: str, base_url: str, max_tokens: int = 4096):
        """
        Initialize AnthropicChatBot.

        Args:
            http_client: HTTP client for making API requests.
            model: The Anthropic model identifier (e.g., "claude-3-opus-20240229").
            base_url: The Anthropic API base URL.
            max_tokens: Maximum tokens to generate (default: 4096).
        """
        super().__init__(http_client, model)
        self._base_url = base_url
        self._max_tokens = max_tokens

    async def send_message(
        self,
        chat_history: ChatHistory,
        streaming: bool = True,
        **kwargs
    ) -> ChatBotResponse:
        """
        Send a chat history to Anthropic and receive a response.

        Args:
            chat_history: The ChatHistory to send.
            streaming: If True, returns streaming response.
            **kwargs: Additional request parameters that will override defaults.

        Returns:
            AnthropicChatBotResponse (inherits from ChatBotResponse).
        """
        body = self._build_anthropic_body(chat_history, streaming, kwargs)

        if streaming:
            stream = self._http_client.stream_post(f"{self._base_url}/v1/messages", body)
            return AnthropicChatBotResponse(stream)
        else:
            response_data = await self._http_client.post(f"{self._base_url}/v1/messages", body)
            # For non-streaming, convert to events
            async def events():
                # Anthropic non-streaming response has content array directly
                for block in response_data.get("content", []):
                    if block.get("type") == "text":
                        yield f"data: {json.dumps({
                            "type": "content_block_start",
                            "content_block": {"type": "text", "text": block.get("text", "")}
                        })}"
                        yield f"data: {json.dumps({
                            "type": "content_block_delta",
                            "delta": {"type": "text_delta", "text": block.get("text", "")}
                        })}"
                        yield f"data: {json.dumps({
                            "type": "content_block_stop",
                            "content_block": {"type": "text"}
                        })}"
                yield "[DONE]"
            return AnthropicChatBotResponse(events())

    def _build_anthropic_body(self, chat_history: ChatHistory, streaming: bool = True, kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Build Anthropic-specific request body.

        Args:
            chat_history: The chat history to convert.
            streaming: Whether to enable streaming.
            kwargs: Additional parameters that override defaults.

        Returns:
            Anthropic-formatted request body.
        """
        messages = []
        for msg in chat_history.messages:
            content = msg.content.get("content", "")
            role = msg.content.get("role", "user")
            if role == "system":
                # Anthropic puts system messages separately
                continue
            messages.append({"role": role, "content": content})

        # Separate system message if present
        system_message = None
        for msg in chat_history.messages:
            if msg.content.get("role") == "system":
                system_message = msg.content.get("content", "")
                break

        body = {
            "model": self._model,
            "messages": messages,
            "stream": streaming,
            "max_tokens": self._max_tokens
        }

        if system_message:
            body["system"] = system_message

        # Allow kwargs to override defaults
        if kwargs:
            body.update(kwargs)

        return body

    def list_available_models(self) -> List[str]:
        """List Anthropic models."""
        return [self._model]  # Simplified - would fetch from API in real implementation
