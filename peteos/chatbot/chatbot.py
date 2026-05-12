"""Abstract ChatBot base class and implementations."""

from abc import ABC, abstractmethod
import json
from typing import Dict, Any, List, Optional, AsyncGenerator

from peteos.logger import get_logger
from .httpclient import HTTPClient
from .chatbotconfig import ChatBotConfig
from .chatbotresponse import ChatBotResponse, GenericChatBotResponse
from .chathistory import ChatHistory
from .message import Message

_logger = get_logger(__name__)


class ChatBot(ABC):
    """Abstract base class for chatbot implementations.

    Subclasses implement specific LLM providers (OpenAI, Anthropic, etc.)
    and translate their response schemas into a common interface.
    """

    def __init__(self, http_client: HTTPClient, config: ChatBotConfig):
        """
        Initialize ChatBot.

        Args:
            http_client: HTTP client for making API requests.
            config: ChatBot configuration dataclass with all defaults applied.
        """
        self._http_client = http_client
        self._config = config

    @abstractmethod
    async def send_message(
        self,
        chat_history: ChatHistory,
        streaming: bool | None = None
    ) -> ChatBotResponse:
        """
        Send a chat history to the LLM and receive a response.

        Args:
            chat_history: The ChatHistory to send to the LLM.
            streaming: If None, uses the instance default.
                       If True/False, overrides the instance default.

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
        return self._config.model

    @model.setter
    def model(self, value: str) -> None:
        """Set a new model identifier."""
        self._config.model = value


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
        config: ChatBotConfig,
    ):
        super().__init__(http_client, config)

    async def send_message(
        self,
        chat_history: ChatHistory,
        streaming: bool | None = None
    ) -> ChatBotResponse:
        """Send a chat history to the LLM and receive a response.

        Args:
            chat_history: The ChatHistory to send to the LLM.
            streaming: If None, uses the instance default.
        """
        streaming_mode = self._config.streaming if streaming is None else streaming
        body = self._build_body(chat_history, streaming)

        if streaming_mode:
            stream = self._http_client.stream_post(f"{self._config.url}{self._config.chat_endpoint}", body)
            return GenericChatBotResponse(stream, self._config.response_translations or {})
        else:
            response_data = await self._http_client.post(f"{self._config.url}{self._config.chat_endpoint}", body)
            return GenericChatBotResponse.from_json(response_data, self._config.response_translations or {})

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
        request_translations = self._config.request_translations or {}
        translated: Dict[str, Any] = {}
        for key, value in msg_data.items():
            if key in request_translations:
                translated[request_translations[key]] = value
            else:
                translated[key] = value
        return translated

    def _build_body(self, chat_history: ChatHistory, streaming: bool | None = None) -> Dict[str, Any]:
        """Build request body from chat history and defaults.

        Override in subclasses for API-specific request format.

        Args:
            chat_history: The chat history to send.
            streaming: Override the default streaming mode. None uses the
                instance default set in __init__.
        """
        body = dict(chat_history.generation_config)
        body["model"] = self._config.model
        body["stream"] = self._config.streaming if streaming is None else streaming

        # Build messages array from ChatHistory
        messages = []
        system_parts = []
        tools = []

        request_translations = self._config.request_translations or {}

        for msg in chat_history.messages:
            role = msg.get_role()

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
                for item in msg.serialize_content():
                    for key, value in item.items():
                        if key in request_translations:
                            api_key = request_translations[key]
                        else:
                            api_key = key
                        if api_key != "type":
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
