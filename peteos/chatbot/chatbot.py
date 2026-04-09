"""Abstract ChatBot base class and implementations."""

from abc import ABC, abstractmethod
import json
from typing import Dict, Any, List, Optional, AsyncGenerator

from peteos.logger import get_logger
from .httpclient import HTTPClient
from .chatbotresponse import ChatBotResponse, GenericChatBotResponse
from .chathistory import ChatHistory
from .message import Message

_logger = get_logger(__name__)


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
        """Build request body from chat history and defaults.

        Override in subclasses for API-specific request format.
        """
        body = dict(chat_history.generation_config)
        body["model"] = self._model
        body["stream"] = streaming

        # Build messages array from ChatHistory
        messages = []
        system_parts = []
        tools = []

        for msg in chat_history.messages:
            role = msg.role

            if role == "system":
                # Extract system content parts
                system_parts.extend(msg.content)
            elif role == "tool":
                # Extract tool definitions
                for part in msg.content:
                    if part.type == "tool":
                        tools.append(part.data)
            elif role in ("user", "assistant"):
                # Conversation messages - build message dict from ContentPart fields
                # Use translation table to map uniform keys to API-specific keys
                msg_dict = {"role": role}
                for part in msg.content:
                    # part.data contains fields like "text", "reasoning", etc.
                    # Use translation to map to API-specific key
                    for key, value in part.data.items():
                        if key in self._request_translations:
                            api_key = self._request_translations[key]
                        else:
                            api_key = key
                        msg_dict[api_key] = value
                messages.append(msg_dict)

        body["messages"] = messages

        # Add system if present
        if system_parts:
            body["system"] = system_parts

        # Add tools if present
        if tools:
            body["tools"] = tools

        return body

    def list_available_models(self) -> List[str]:
        """List available models from the models endpoint or return [model]."""
        # For now, return just the configured model
        # Could be extended to fetch from models_endpoint
        return [self._model]
