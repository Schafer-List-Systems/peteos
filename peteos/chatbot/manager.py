"""ChatBot manager for multiple backend providers."""

import re
import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .httpclient import HTTPClient
from .backendconfig import BackendConfig
from .chatbotconfig import ChatBotConfig
from .backendprovider import BackendProvider
from .openaiprovider import OpenAIChatBotProvider
from .anthropicprovider import AnthropicChatBotProvider
from .geminiprovider import GeminiChatBotProvider
from peteos.utils import get_logger, json

_logger = get_logger(__name__)


@dataclass
class BackendInfo:
    """Information about a backend provider."""
    name: str
    url: str
    api_type: str  # "openai", "anthropic", or "gemini"
    models: Dict[str, Any]  # model_id -> ChatBot instance


class ChatBotManager:
    """Manager for multiple LLM backend providers.

    Uses class-level state so backends are globally accessible without
    passing an instance around. API-specific model listing and chatbot
    creation are delegated to BackendProvider instances.
    """

    # Class-level state
    _backends: Dict[str, BackendInfo] = {}
    _timeout: Optional[float] = None
    _providers: Dict[str, BackendProvider] = {
        "openai": OpenAIChatBotProvider(),
        "anthropic": AnthropicChatBotProvider(),
        "gemini": GeminiChatBotProvider(),
    }

    def __init__(self, timeout: Optional[float] = None) -> None:
        """Update the default timeout for backward compatibility.

        Args:
            timeout: HTTP request timeout in seconds. Pass None for no timeout.
        """
        if timeout is not None:
            ChatBotManager._timeout = timeout
            BackendConfig._default_timeout = timeout

    @classmethod
    def reset(cls) -> None:
        """Clear all backends.

        Use in tests or when reloading configuration. Does not remove
        registered providers — use ``unregister_provider`` for that.
        """
        cls._backends.clear()

    @classmethod
    async def add_backend(
        cls,
        name: str,
        url: Optional[str] = None,
        api_type: Optional[str] = None,
        **kwargs
    ) -> BackendInfo:
        """Add a new backend and auto-detect its capabilities.

        Args:
            name: Unique identifier for the backend.
            url: Base URL of the API (e.g., "http://localhost:8000").
                 Optional for mock providers.
            api_type: Optional API type override ("openai", "anthropic", "gemini",
                      or any registered provider type like "simple-mock").
                     If not provided, API type is auto-detected (requires url).
            **kwargs: Configuration options (streaming, max_tokens, api_key, etc.).

        Returns:
            BackendInfo with detected API type and discovered models.

        Raises:
            ValueError: If backend with same name already exists.
            RuntimeError: If API detection fails.
        """
        if name in cls._backends:
            raise ValueError(f"Backend '{name}' already exists")

        config = BackendConfig.from_dict({
            "name": name,
            "url": url,
            **{"api_type": api_type},
            **kwargs,
        })
        return await cls._add_backend(config)

    @classmethod
    async def _add_backend(cls, config: BackendConfig) -> BackendInfo:
        """Add a backend from a BackendConfig."""
        if config.name in cls._backends:
            raise ValueError(f"Duplicate backend name '{config.name}' in configuration")

        client = HTTPClient(
            timeout=config.timeout,
            retry_delays=config.retry_delays,
        )

        # Add placeholder before detection so methods can look up url
        cls._backends[config.name] = BackendInfo(
            name=config.name,
            url=config.url,
            api_type=config.api_type,
            models={},
        )

        # Auto-detect api_type or delegate to provider
        if config.api_type is None:
            config.api_type, models = await cls._detect_api_and_list_models(config.name, config.api_key)
        else:
            provider = cls._providers.get(config.api_type)
            if provider is None:
                raise RuntimeError(f"No provider registered for api_type: {config.api_type}")
            models = await provider.list_models(config.url, config.api_key)

        # Create ChatBot instances via provider
        provider = cls._providers[config.api_type]
        model_ids = models
        chatbots: Dict[str, Any] = {}
        for model_id in model_ids:
            chatbot_config = ChatBotConfig.from_dict({
                **vars(config),
                "model": model_id,
                "priority": (config.model_priorities or {}).get(model_id, 0),
            })
            chatbot = provider.create_chatbot(client, chatbot_config)
            chatbots[model_id] = chatbot

        cls._backends[config.name].api_type = config.api_type
        cls._backends[config.name].models = chatbots
        return cls._backends[config.name]

    @classmethod
    def remove_backend(cls, name: str) -> bool:
        """Remove a backend and all its ChatBot instances.

        Args:
            name: Backend identifier.

        Returns:
            True if backend was removed, False if not found.
        """
        if name in cls._backends:
            del cls._backends[name]
            return True
        return False

    @classmethod
    def register_provider(cls, api_type: str, provider: "BackendProvider") -> None:
        """Register a new backend provider.

        Args:
            api_type: Unique API type identifier (e.g. "simple-mock").
            provider: BackendProvider instance to handle chatbot creation.

        Raises:
            ValueError: If provider with same api_type already exists.
        """
        if api_type in cls._providers:
            raise ValueError(f"Provider already registered for api_type: {api_type}")
        cls._providers[api_type] = provider

    @classmethod
    def unregister_provider(cls, api_type: str) -> bool:
        """Remove a backend provider.

        Args:
            api_type: API type identifier of the provider to remove.

        Returns:
            True if provider was removed, False if not found.
        """
        if api_type in cls._providers:
            del cls._providers[api_type]
            return True
        return False

    @staticmethod
    def _classify_exception(exc: Exception) -> bool:
        """Return True if the exception is soft (should try next provider).

        Fatal errors are re-raised immediately; the backend is genuinely
        unavailable so other providers won't help either.

        Args:
            exc: The caught exception.

        Returns:
            True for soft errors (try next provider), False for fatal errors.
        """
        import httpx

        # Fatal: network/transport-level issues
        if isinstance(exc, (httpx.RequestError, httpx.InvalidURL,
                            socket.error, ConnectionRefusedError, OSError)):
            return False

        # Fatal: timeouts
        if isinstance(exc, httpx.TimeoutException):
            return False

        # Soft: got an HTTP response but not right for this provider
        if isinstance(exc, (httpx.HTTPStatusError, httpx.ProtocolError,
                            httpx.RemoteProtocolError)):
            return True

        # Soft: malformed response (wrong JSON shape, missing fields, etc.)
        if isinstance(exc, (ValueError, KeyError, TypeError,
                            json.JSONDecodeError)):
            return True

        # Unknown: re-raise conservatively
        return False

    @staticmethod
    async def _detect_api_and_list_models(
        backend_name: str, api_key: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """Detect API type by probing all providers and listing available models.

        Classifies each exception as fatal (re-raise immediately — the backend
        is unreachable) or soft (collect and try the next provider). If all
        providers yield soft errors, raises a RuntimeError with diagnostics
        for all of them.

        Probing order: OpenAI -> Anthropic -> Gemini (backward compat).

        Args:
            backend_name: Backend identifier used to look up URL.
            api_key: Optional API key for authentication.

        Returns:
            Tuple of (api_type, [model_ids]).

        Raises:
            RuntimeError: If no provider succeeds and all errors were soft.
            (Fatal exceptions propagate immediately.)
        """
        url = ChatBotManager._backends[backend_name].url
        soft_errors: list[str] = []

        for api_type, provider_name in [
            ("openai", "OpenAI"),
            ("anthropic", "Anthropic"),
            ("gemini", "Gemini"),
        ]:
            try:
                models = await ChatBotManager._providers[api_type].list_models(
                    url, api_key
                )
                if models:
                    return api_type, models
                soft_errors.append(f"{provider_name}: no models found")
            except Exception as exc:
                if ChatBotManager._classify_exception(exc):
                    soft_errors.append(
                        f"{provider_name}: {type(exc).__name__}: {exc}"
                    )
                else:
                    raise

        raise RuntimeError(
            f"Failed to detect API type or list models from {url}.\n"
            f"Providers tried:\n"
            + "\n".join(f"  - {e}" for e in soft_errors)
        )

    @classmethod
    def list_chatbots(cls, model_regex: str) -> List[Tuple[str, Any]]:
        """List all ChatBot instances matching a regex pattern.

        Args:
            model_regex: Regex pattern to match model IDs.

        Returns:
            List of (model_id, ChatBot) tuples, sorted by descending priority
            (higher priority first), then alphabetically by model_id for ties.
        """
        pattern = re.compile(model_regex)
        results = []

        for backend in cls._backends.values():
            for model_id, chatbot in backend.models.items():
                if pattern.search(model_id):
                    results.append((model_id, chatbot))

        return sorted(results, key=lambda x: (-x[1].priority, x[0]))

    @classmethod
    async def load_from_json(cls, json_obj: dict) -> None:
        """Load backend configuration from JSON object.

        Clears current state and recreates all backends from the JSON.
        API types can be auto-detected or explicitly specified in config.

        Config format:
            {
                "backends": [
                    {
                        "name": "my-backend",
                        "url": "http://localhost:8000",
                        "api_type": "anthropic",
                        "streaming": false
                    }
                ]
            }

        Args:
            json_obj: Dict with "backends" key containing list of backend configs.
        """
        cls._backends.clear()

        for backend_config in json_obj.get("backends", []):
            config = BackendConfig.from_dict(backend_config)

            if config.api_type is not None and config.api_type not in ("openai", "anthropic", "gemini"):
                raise ValueError(f"Invalid api_type in config: {config.api_type}")

            await cls._add_backend(config)

    @classmethod
    async def load_from_file(cls, filepath: str) -> None:
        """Load backend configuration from JSON file.

        Args:
            filepath: Path to JSON configuration file.
        """
        with open(filepath, "r") as f:
            json_obj = json.load(f)
        await cls.load_from_json(json_obj)
