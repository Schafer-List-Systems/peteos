"""Abstract ChatBot base class and implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from peteos.utils import get_logger
from .httpclient import HTTPClient
from .chatbotconfig import ChatBotConfig
from .chatbotresponse import ChatBotResponse
from peteos.conversation.context import Context

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
    async def send_context(
        self,
        context: Context,
        generation_config: Optional[Dict[str, Any]] = None,
        streaming: bool | None = None
    ) -> ChatBotResponse:
        """
        Send a context to the LLM and receive a response.

        Args:
            context: The Context to send to the LLM.
            generation_config: Optional generation parameters (temperature,
                max_tokens, tool_choice, etc.) passed to the LLM provider.
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
