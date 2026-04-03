"""Abstract ChatBot base class and implementations."""

from abc import ABC, abstractmethod
import json
from typing import Dict, Any, List, Optional, AsyncGenerator

from .httpclient import HTTPClient
from .chatbotresponse import ChatBotResponse, GenericChatBotResponse, AnthropicChatBotResponse
from .chathistory import ChatHistory
from .message import Message


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


class GenericChatBot(ChatBot):
    """Generic ChatBot with configurable endpoints and response translations.

    Args:
        http_client: HTTP client for making API requests.
        model: The model identifier to use.
        chat_endpoint: API endpoint for chat (e.g., "/v1/chat/completions").
        models_endpoint: API endpoint for listing models (e.g., "/v1/models").
        response_translations: Dict mapping source path -> target field.
            Paths use notation like "choices[*].delta.content" -> "text".
            If no translation matches, events are passed through unchanged.
        **defaults: Additional request body parameters (e.g., max_tokens=4096).

    Example:
        chatbot = GenericChatBot(
            http_client,
            "qwen3.5-35b",
            chat_endpoint="/v1/chat/completions",
            models_endpoint="/v1/models",
            response_translations={
                "choices[*].delta.content": "text",
                "choices[*].delta.reasoning": "reasoning"
            },
            max_tokens=4096
        )
    """

    def __init__(
        self,
        http_client: HTTPClient,
        model: str,
        base_url: str = "",
        chat_endpoint: str = "/v1/chat/completions",
        models_endpoint: str = "/v1/models",
        response_translations: Optional[Dict[str, str]] = None,
        request_translations: Optional[Dict[str, str]] = None,
        **defaults
    ):
        super().__init__(http_client, model)
        self._base_url = base_url
        self._chat_endpoint = chat_endpoint
        self._models_endpoint = models_endpoint
        self._translations = response_translations or {}
        self._request_translations = request_translations or {}
        self._defaults = defaults

    async def send_message(
        self,
        chat_history: ChatHistory,
        streaming: bool = True
    ) -> ChatBotResponse:
        """Send a chat history to the LLM and receive a response."""
        body = self._build_body(chat_history, streaming)

        if streaming:
            stream = self._http_client.stream_post(f"{self._base_url}{self._chat_endpoint}", body)
            return GenericChatBotResponse(stream, self._translations)
        else:
            response_data = await self._http_client.post(f"{self._base_url}{self._chat_endpoint}", body)
            return GenericChatBotResponse.from_json(response_data, self._translations)

    def _translate_message_fields(self, msg_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate message content keys from uniform API to API-specific format.

        Generic implementation: keys in request_translations are translated,
        keys not in the table are forwarded as-is.

        Args:
            msg_data: Message content dict with uniform keys.

        Returns:
            Message content dict with API-specific keys.
        """
        translated: Dict[str, Any] = {}
        for key, value in msg_data.items():
            if key in self._request_translations:
                translated[self._request_translations[key]] = value
            else:
                translated[key] = value
        return translated

    def _build_body(self, chat_history: ChatHistory, streaming: bool) -> Dict[str, Any]:
        """Build request body from chat history and defaults."""
        body = dict(self._defaults)
        body["model"] = self._model
        body["stream"] = streaming

        # Build messages array - translate uniform keys to API keys
        messages = []
        system_content = None

        for msg in chat_history.messages:
            # Translate message content keys
            msg_data = self._translate_message_fields(msg.content)
            role = msg_data.get("role", "user")

            if role == "system":
                # Extract system content separately
                system_content = msg_data.get("content", msg_data)
            else:
                messages.append(msg_data)

        body["messages"] = messages

        # Add system if present
        if system_content:
            # Check if system is in our defaults (Anthropic style)
            if "system" not in body:
                body["system"] = system_content

        return body

    def list_available_models(self) -> List[str]:
        """List available models from the models endpoint or return [model]."""
        # For now, return just the configured model
        # Could be extended to fetch from models_endpoint
        return [self._model]


class OpenAIChatBot(GenericChatBot):
    """ChatBot implementation for OpenAI-compatible API."""

    # Default translation configuration for OpenAI API
    RESPONSE_TRANSLATIONS = {
        "choices[*].delta.role": "role",
        "choices[*].message.content": "text",
        "choices[*].message.reasoning": "reasoning",
        "choices[*].message.thinking": "reasoning",
        "choices[*].delta.content": "text",
        "choices[*].delta.reasoning": "reasoning",
        "choices[*].delta.thinking": "reasoning",
        "choices[*].delta.tool_calls": "tool_calls",
        "choices[*].message.tool_calls": "tool_calls",
    }

    REQUEST_TRANSLATIONS = {
        "text": "content",
        "reasoning": "reasoning",
        "tool_calls": "tool_calls",
    }

    def __init__(self, http_client: HTTPClient, model: str, base_url: str):
        """
        Initialize OpenAIChatBot.

        Args:
            http_client: HTTP client for making API requests.
            model: The OpenAI model identifier (e.g., "gpt-4").
            base_url: The OpenAI API base URL.
        """
        super().__init__(
            http_client=http_client,
            model=model,
            base_url=base_url,
            chat_endpoint="/v1/chat/completions",
            models_endpoint="/v1/models",
            response_translations=self.RESPONSE_TRANSLATIONS,
            request_translations=self.REQUEST_TRANSLATIONS,
        )

class AnthropicChatBot(GenericChatBot):
    """ChatBot implementation for Anthropic-compatible API."""

    # Default translation configuration for Anthropic API
    # stream=True (streaming): content via content_block_start/content_block_delta events
    # stream=False (non-streaming): role, content array at top level
    # Real Anthropic API: {"role": "assistant", "content": [{"type": "text", "text": "..."}], ...}
    # Qwen local API:      {"type": "message", "role": "assistant", "content": [...], ...}
    RESPONSE_TRANSLATIONS = {
        # stream=True entries (streaming mode)
        "message_start.message.role": "role",              # role in message_start event
        "content_block_delta.delta.text": "text",          # text chunks
        "content_block_delta.delta.thinking": "reasoning", # reasoning chunks
        "content_block_start.content_block.text": "text",  # text block start
        "content_block_start.content_block.thinking": "reasoning",  # thinking block start
        "content_block_start.content_block.reasoning": "reasoning",  # reasoning block start
        # stream=False entries (non-streaming mode)
        "role": "role",                                    # role at top level
        "content[*].text": "text",                         # content array at top level
        "content[*].thinking": "reasoning",                # content array with thinking
    }

    REQUEST_TRANSLATIONS = {
        "text": "content",
        "reasoning": "reasoning",
        "tool_calls": "tool_calls",
    }

    def __init__(self, http_client: HTTPClient, model: str, base_url: str, max_tokens: int = 4096):
        """
        Initialize AnthropicChatBot.

        Args:
            http_client: HTTP client for making API requests.
            model: The Anthropic model identifier (e.g., "claude-3-opus-20240229").
            base_url: The Anthropic API base URL.
            max_tokens: Maximum tokens to generate (default: 4096).
        """
        super().__init__(
            http_client=http_client,
            model=model,
            base_url=base_url,
            chat_endpoint="/v1/messages",
            models_endpoint="/v1/models",
            response_translations=self.RESPONSE_TRANSLATIONS,
            request_translations=self.REQUEST_TRANSLATIONS,
        )
        self._max_tokens = max_tokens

    async def send_message(
        self,
        chat_history: ChatHistory,
        streaming: bool = True,
        **kwargs
    ) -> ChatBotResponse:
        """Send a chat history to Anthropic-compatible API."""
        body = self._build_body(chat_history, streaming)
        body.update(kwargs)

        if streaming:
            stream = self._http_client.stream_post(f"{self._base_url}{self._chat_endpoint}", body)
            return AnthropicChatBotResponse(stream, self._translations)
        else:
            response_data = await self._http_client.post(f"{self._base_url}{self._chat_endpoint}", body)
            return AnthropicChatBotResponse.from_json(response_data, self._translations)

    def _build_body(self, chat_history: ChatHistory, streaming: bool) -> Dict[str, Any]:
        """Build request body from chat history and defaults."""
        body = super()._build_body(chat_history, streaming)
        body["max_tokens"] = self._max_tokens
        return body
