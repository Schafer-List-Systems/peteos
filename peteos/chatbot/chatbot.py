"""Abstract ChatBot base class and implementations."""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from peteos.utils import get_logger
from peteos.utils import json
from .httpclient import HTTPClient
from .chatbotconfig import ChatBotConfig
from .chatbotresponse import ChatBotResponse
from peteos.conversation import ContentPart, Message, Context

_logger = get_logger(__name__)

# Executor type: async func(url, body, caller_headers) -> response
PostExecutor = Callable[..., Any]


class ContextOverflowError(Exception):
    """Raised when an HTTP error is confirmed as likely caused by context size exceeding the model's limit.

    Fired after a probe (minimal context) succeeds but the original context fails,
    giving high confidence the error is a context overflow, not a transient network
    or auth issue.
    """


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
    async def _send_context(
        self,
        context: Context,
        generation_config: Optional[Dict[str, Any]] = None,
        streaming: bool | None = None,
        hooks: Optional[dict[str, list[Callable]]] = None,
    ) -> ChatBotResponse:
        """
        Provider-internal send — override to implement actual LLM calls.

        Called by the public send_context() wrapper in this base class,
        which manages learned context limit bounds on success and overflow.
        Subclasses should not need to manage bounds themselves.

        Args:
            context: The Context to send to the LLM.
            generation_config: Optional generation parameters (temperature,
                max_tokens, tool_choice, etc.) passed to the LLM provider.
            streaming: If None, uses the instance default.
                       If True/False, overrides the instance default.
            hooks: Optional dict of hook name -> list of callables.
                Passed to the HTTP client for error interception.
                "on_http_error" hooks receive error context and may raise.

        Returns:
            A ChatBotResponse that can be iterated to receive the response.
        """
        pass

    async def send_context(
        self,
        context: Context,
        generation_config: Optional[Dict[str, Any]] = None,
        streaming: bool | None = None,
    ) -> ChatBotResponse:
        """Send a context to the LLM, learning bounds on success and overflow.

        On a successful response, tightens the learned lower bound to
        max(known_floor, context_token_count). On a confirmed context
        overflow (ContextOverflowError from the probe strategy), tightens
        the learned upper bound to the failed size. Calls the provider's
        _send_context for the actual LLM interaction.

        Args:
            context: The Context to send to the LLM.
            generation_config: Optional generation parameters.
            streaming: If None, uses the instance default.

        Returns:
            A ChatBotResponse from the provider.
        """
        # Probe hook: on HTTP error, send a minimal context to check backend
        # reachability. Retries at most once. On clear non-context errors (network
        # failures), returns immediately without probing.
        _probe_attempt = 0
        async def _on_http_error(err_ctx: dict) -> None:
            nonlocal _probe_attempt
            exc = err_ctx.get("exception")
            status_code = err_ctx["status_code"]
            attempt = err_ctx.get("attempt", 1)

            # Let the HTTP-Client handle non-context errors
            if isinstance(exc, (ConnectionError, OSError, TimeoutError)):
                return

            # On the second call, the first probe already succeeded but the
            # retry still failed — treat this as a confirmed context overflow.
            if _probe_attempt >= 1:
                failed_size = context.total_token_count() + self._config.max_tokens
                if failed_size < self._config.context_limit_ceil:
                    self._config.context_limit_ceil = failed_size
                raise ContextOverflowError(
                    f"HTTP {status_code} after probe on retry #2 — context overflow",
                )

            # Attempt 1: send a minimal probe to check backend reachability.
            _probe_attempt += 1
            _logger.debug(
                "[runner] _send_to_chatbot: HTTP %d — probing reachability (attempt %d)",
                status_code,
                _probe_attempt,
            )

            # Build a minimal probe context inheriting the parent's tool definitions.
            probe_ctx = Context.create(parent_context=context)
            probe_msg = Message.create(
                role="user",
                content_parts=[ContentPart.create_text("hello")],
            )
            probe_ctx.append(probe_msg, anchor_point="messages")

            # Send the probe with a tight output limit.
            try:
                probe_response = await self._send_context(
                    probe_ctx,
                    generation_config={"max_tokens": 10},
                )
                async for _ in probe_response:
                    pass
            except Exception as probe_err:
                _logger.debug(
                    "[runner] _send_to_chatbot: Probe failed — backend unreachable: %s",
                    str(probe_err)[:80],
                )
                raise err_ctx["exception"]

            # Probe succeeded — the backend is reachable. Let the HTTP client
            # retry the original. If that also fails, this hook fires again and
            # the second call will raise ContextOverflowError.
            _logger.debug(
                "[runner] _send_to_chatbot: Probe succeeded — allowing HTTP retry",
            )

        response = await self._send_context(
            context,
            generation_config,
            streaming,
            hooks={"on_http_error": [_on_http_error]}
        )

        # Confirm the context is handleable — update the learned lower bound.
        context_size = context.total_token_count()
        new_floor = context_size + self._config.max_tokens
        if new_floor > self._config.context_limit_floor:
            self._config.context_limit_floor = new_floor

        return response

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
        (e.g., ``__closure__[0].__cell_contents__``) but is NOT a secure mechanism
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
            An async callable ``(body, caller_headers, hooks=None) -> response``.
        """
        headers_json = json.dumps(secure_headers)

        _code = f"""
async def executor(body, caller_headers, hooks=None):
    if id(http_client) != {id(http_client)}:
        raise ValueError("HTTP client was replaced at runtime")
    safe = dict(caller_headers or {{}})
    safe.update({headers_json})
    return await http_client.post({endpoint!r}, body, headers=safe, hooks=hooks)
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
        but is NOT a secure mechanism for storing secrets. If sandboxed code
        may access the executor, use an intermediary proxy vault holding real API keys.

        Args:
            http_client: The HTTP client instance to use.
            secure_headers: Headers to merge on every call, including any
                API keys or other secrets.
            endpoint: The API endpoint URL (hardcoded into compiled source).

        Returns:
            An async callable ``(body, caller_headers, hooks=None) ->
            AsyncGenerator[str, None]``.
        """
        headers_json = json.dumps(secure_headers)

        _code = f"""
async def executor(body, caller_headers, hooks=None):
    if id(http_client) != {id(http_client)}:
        raise ValueError("HTTP client was replaced at runtime")
    safe = dict(caller_headers or {{}})
    safe.update({headers_json})
    async for line in http_client.stream_post({endpoint!r}, body, headers=safe, hooks=hooks):
        yield line
"""
        _globals: Dict[str, Any] = {"http_client": http_client}
        exec(_code, _globals)
        return _globals["executor"]
