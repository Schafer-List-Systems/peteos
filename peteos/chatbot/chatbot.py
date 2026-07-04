"""Abstract ChatBot base class and implementations."""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from peteos.utils import get_logger
from .httpclient import HTTPClient
from .chatbotconfig import ChatBotConfig
from .chatbotresponse import ChatBotResponse
from peteos.conversation.context import Context

_logger = get_logger(__name__)

# Executor type: async func(url, body, caller_headers) -> response
PostExecutor = Callable[..., Any]


class ChatBot(ABC):
    """Abstract base class for chatbot implementations.

    Subclasses implement specific LLM providers (OpenAI, Anthropic, etc.)
    and translate their response schemas into a common interface.
    """

    def __init__(self, config: ChatBotConfig):
        """
        Initialize ChatBot.

        Args:
            config: ChatBot configuration dataclass with all defaults applied.
        """
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

    @property
    def priority(self) -> int:
        """Get the priority of this chatbot's model.

        Higher values indicate higher priority. Models are sorted by
        descending priority when listing available chatbots.
        """
        return self._config.priority

    def get_headers(self) -> Dict[str, str]:
        """Return extra HTTP headers to include with every API request.

        Override in subclasses to provide API-specific auth/version headers
        that will be merged into the request at execution time.
        """
        return {}

    def _build_post_executor(
        self,
        http_client: HTTPClient,
        secure_headers: Dict[str, str],
    ) -> PostExecutor:
        """Factory that builds a POST executor closure.

        The executor captures the supplied HTTP client and the secure
        headers (which may include API keys) in its closure. Callers
        never see the raw values — they are merged into a local dict and
        passed only to the HTTP client.

        Args:
            http_client: The HTTP client instance to use.
            secure_headers: Headers to merge on every call, including any
                API keys or other secrets.

        Returns:
            An async callable ``(url, body, caller_headers) -> response``.
        """
        async def executor(
            url: str,
            body: Dict[str, Any],
            caller_headers: Optional[Dict[str, str]],
        ) -> Any:
            safe = dict(caller_headers or {})
            safe.update(secure_headers)
            return await http_client.post(url, body, headers=safe)

        return executor

    def _build_stream_executor(
        self,
        http_client: HTTPClient,
        secure_headers: Dict[str, str],
    ) -> PostExecutor:
        """Factory that builds a streaming POST executor closure.

        Same pattern as ``_build_post_executor`` but calls
        ``http_client.stream_post`` to return an async generator.

        Args:
            http_client: The HTTP client instance to use.
            secure_headers: Headers to merge on every call, including any
                API keys or other secrets.

        Returns:
            An async callable ``(url, body, caller_headers) ->
            AsyncGenerator[str, None]``.
        """
        async def executor(
            url: str,
            body: Dict[str, Any],
            caller_headers: Optional[Dict[str, str]],
        ) -> AsyncGenerator[str, None]:  # type: ignore[return]
            safe = dict(caller_headers or {})
            safe.update(secure_headers)
            async for line in http_client.stream_post(url, body, headers=safe):
                yield line

        return executor
