"""Abstract ChatBot base class and implementations."""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from peteos.utils import get_logger
from peteos.utils import json
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
        endpoint: str,
    ) -> PostExecutor:
        """Factory that builds a POST executor with compiled source.

        The executor source code is compiled via exec() with secure_headers
        and the endpoint embedded as hardcoded values in the generated string.
        The returned callable has no closure cells containing secrets.

        WARNING: The API keys and endpoint URL are embedded as string literals
        in the compiled bytecode and are visible through ``executor.__code__.co_consts``.
        An agent with access to the returned callable can extract these secrets
        via introspection. This approach protects against closure-based leaks
        (e.g., ``__closure__[0].__cell_contents``) but is NOT a secure mechanism
        for storing secrets. If sandboxed code may access the executor, use an
        intermediary proxy vault that holds real API keys — the executor should
        only contain unprivileged proxy credentials (e.g. ``http://localhost``
        proxy URL + proxy key) that are useless outside the local network.

        Args:
            http_client: The HTTP client instance to use.
            secure_headers: Headers to merge on every call, including any
                API keys or other secrets.
            endpoint: The API endpoint URL (hardcoded into compiled source).

        Returns:
            An async callable ``(body, caller_headers) -> response``.
        """
        headers_json = json.dumps(secure_headers)

        _code = f"""
async def executor(body, caller_headers):
    if id(http_client) != {id(http_client)}:
        raise ValueError("HTTP client was replaced at runtime")
    safe = dict(caller_headers or {{}})
    safe.update({headers_json})
    return await http_client.post({endpoint!r}, body, headers=safe)
"""
        _globals: Dict[str, Any] = {"http_client": http_client}
        exec(_code, _globals)
        return _globals["executor"]

    def _build_stream_executor(
        self,
        http_client: HTTPClient,
        secure_headers: Dict[str, str],
        endpoint: str,
    ) -> PostExecutor:
        """Factory that builds a streaming POST executor with compiled source.

        Same pattern as ``_build_post_executor`` but calls
        ``http_client.stream_post`` to return an async generator.

        WARNING: The API keys and endpoint URL are embedded as string literals
        in the compiled bytecode and are visible through ``executor.__code__.co_consts``.
        An agent with access to the returned callable can extract these secrets
        via introspection. This approach protects against closure-based leaks
        but is NOT a secure mechanism for storing secrets. If sandboxed code may
        access the executor, use an intermediary proxy vault holding real API keys.

        Args:
            http_client: The HTTP client instance to use.
            secure_headers: Headers to merge on every call, including any
                API keys or other secrets.
            endpoint: The API endpoint URL (hardcoded into compiled source).

        Returns:
            An async callable ``(body, caller_headers) ->
            AsyncGenerator[str, None]``.
        """
        headers_json = json.dumps(secure_headers)

        _code = f"""
async def executor(body, caller_headers):
    if id(http_client) != {id(http_client)}:
        raise ValueError("HTTP client was replaced at runtime")
    safe = dict(caller_headers or {{}})
    safe.update({headers_json})
    async for line in http_client.stream_post({endpoint!r}, body, headers=safe):
        yield line
"""
        _globals: Dict[str, Any] = {"http_client": http_client}
        exec(_code, _globals)
        return _globals["executor"]
