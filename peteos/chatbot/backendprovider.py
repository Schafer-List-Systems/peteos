"""Abstract backend provider protocol for API-specific model listing and chatbot creation."""

from abc import ABC, abstractmethod
from typing import List, Optional

from .httpclient import HTTPClient
from .chatbotconfig import ChatBotConfig
from .chatbot import ChatBot


class BackendProvider(ABC):
    """Protocol for API-specific backend operations.

    Each provider knows how to:
    - List models from its API (correct URL, auth, response parsing)
    - Create the appropriate ChatBot instance
    """

    @abstractmethod
    async def list_models(
        self, url: str, api_key: Optional[str] = None
    ) -> List[str]:
        """List available models from this API.

        Args:
            url: API base URL.
            api_key: Optional API key for authentication.

        Returns:
            List of model identifiers.
        """

    @abstractmethod
    def create_chatbot(
        self, http_client: HTTPClient, config: ChatBotConfig
    ) -> ChatBot:
        """Create the appropriate ChatBot instance.

        Args:
            http_client: HTTP client for making API requests.
            config: ChatBot configuration dataclass.

        Returns:
            ChatBot instance.
        """
