"""ChatBot manager for multiple backend providers."""

import json
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .httpclient import HTTPClient
from .backendconfig import BackendConfig
from .chatbotconfig import ChatBotConfig
from .backendprovider import BackendProvider
from .openaiprovider import OpenAIChatBotProvider
from .anthropicprovider import AnthropicChatBotProvider
from .geminiprovider import GeminiChatBotProvider
from peteos.utils import get_logger

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
    _clients: Dict[str, HTTPClient] = {}
    _timeout: Optional[float] = None
    _config_dir: Optional[str] = None  # directory containing loaded peteos.json
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

    @classmethod
    def reset(cls) -> None:
        """Clear all backends and clients.

        Use in tests or when reloading configuration.
        """
        cls._backends.clear()
        cls._clients.clear()
        cls._config_dir = None

    @staticmethod
    def _resolve_config_path() -> Optional[str]:
        """Resolve the path to peteos.json using standard discovery order.

        Resolution order (first found wins):
            1. PETEOS_CONFIG environment variable (full path)
            2. $XDG_CONFIG_HOME/peteos/peteos.json (defaults to ~/.config/peteos/peteos.json)
            3. /etc/peteos/peteos.json
            4. ./peteos.json (current working directory)

        Returns:
            Absolute path to peteos.json, or None if no config file exists.
        """
        # 1. Explicit env var
        env_path = os.environ.get("PETEOS_CONFIG")
        if env_path and Path(env_path).is_file():
            return os.path.abspath(env_path)

        # 2. XDG config directory
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        if not xdg_config:
            xdg_config = str(Path.home() / ".config")
        xdg_path = Path(xdg_config) / "peteos" / "peteos.json"
        if xdg_path.is_file():
            return str(xdg_path)

        # 3. System config
        system_path = "/etc/peteos/peteos.json"
        if Path(system_path).is_file():
            return system_path

        # 4. Current working directory
        local_path = Path.cwd() / "peteos.json"
        if local_path.is_file():
            return str(local_path)

        return None

    @staticmethod
    def _load_config_file(filepath: str) -> dict:
        """Load and parse a peteos.json config file.

        Args:
            filepath: Absolute path to the JSON config file.

        Returns:
            Parsed JSON dict.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        with open(filepath, "r") as f:
            return json.load(f)

    @classmethod
    async def load_from_config(cls) -> None:
        """Load backend configuration from the first discovered peteos.json.

        Discovers the config file using standard locations (see _resolve_config_path).
        If no config file is found, logs a warning and does nothing.
        The directory containing the loaded config is stored as _config_dir
        for use by other subsystems (e.g., role loading from {config_dir}/roles/).

        Backends are loaded via the existing load_from_json method.
        """
        filepath = cls._resolve_config_path()
        if filepath is None:
            _logger.warning("No peteos.json config file found in standard locations")
            cls._config_dir = None
            return

        config_dir = str(Path(filepath).parent)
        cls._config_dir = config_dir
        json_obj = cls._load_config_file(filepath)
        await cls.load_from_json(json_obj)

    @classmethod
    async def add_backend(
        cls,
        name: str,
        url: str,
        api_type: Optional[str] = None,
        **kwargs
    ) -> BackendInfo:
        """Add a new backend and auto-detect its capabilities.

        Args:
            name: Unique identifier for the backend.
            url: Base URL of the API (e.g., "http://localhost:8000").
            api_type: Optional API type override ("openai", "anthropic", or "gemini").
                     If not provided, API type is auto-detected.
            **kwargs: Configuration options (streaming, max_tokens, api_key, etc.).

        Returns:
            BackendInfo with detected API type and discovered models.

        Raises:
            ValueError: If backend with same name already exists.
            RuntimeError: If API detection fails.
        """
        if name in cls._backends:
            raise ValueError(f"Backend '{name}' already exists")

        if api_type is not None and api_type not in ("openai", "anthropic", "gemini"):
            raise ValueError(f"Invalid api_type: {api_type}. Must be 'openai', 'anthropic', or 'gemini'")

        config = BackendConfig.from_dict({
            "name": name,
            "url": url,
            **{"api_type": api_type},
            **kwargs,
        })
        cls._clients[name] = HTTPClient(
            timeout=cls._timeout,
            retry_delays=config.retry_delays,
        )

        # Add placeholder before detection so methods can look up url
        cls._backends[name] = BackendInfo(
            name=config.name,
            url=config.url,
            api_type=config.api_type,
            models={},
        )

        # Auto-detect api_type or delegate to provider
        if config.api_type is None:
            config.api_type, models = await cls._detect_api_and_list_models(name, config.api_key)
        else:
            provider = cls._providers.get(config.api_type)
            if provider is None:
                raise RuntimeError(f"No provider registered for api_type: {config.api_type}")
            models = await provider.list_models(config.url, config.api_key)

        # Create ChatBot instances via provider
        provider = cls._providers[config.api_type]
        chatbots: Dict[str, Any] = {}
        for model_id in models:
            chatbot_config = ChatBotConfig.from_dict({
                **vars(config),
                "model": model_id,
                "priority": (config.model_priorities or {}).get(model_id, 0),
            })
            chatbot = provider.create_chatbot(cls._clients[name], chatbot_config)
            chatbots[model_id] = chatbot

        cls._backends[name].api_type = config.api_type
        cls._backends[name].models = chatbots
        return cls._backends[name]

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
            del cls._clients[name]
            return True
        return False

    @staticmethod
    async def _detect_api_and_list_models(backend_name: str, api_key: Optional[str] = None) -> Tuple[str, List[str]]:
        """Detect API type by probing all providers and listing available models.

        Tries each registered provider's list_models method and uses whichever
        succeeds. Falls back to OpenAI (data) before Anthropic (models) before
        Gemini to maintain backward compatibility with auto-detection.

        Args:
            backend_name: Backend identifier used to look up URL.
            api_key: Optional API key for authentication.

        Returns:
            Tuple of (api_type, [model_ids]).

        Raises:
            RuntimeError: If no provider succeeds.
        """
        url = ChatBotManager._backends[backend_name].url

        # Try OpenAI first (backward compat with auto-detection)
        try:
            models = await ChatBotManager._providers["openai"].list_models(url, api_key)
            if models:
                return "openai", models
        except Exception:
            pass

        # Try Anthropic
        try:
            models = await ChatBotManager._providers["anthropic"].list_models(url, api_key)
            if models:
                return "anthropic", models
        except Exception:
            pass

        # Try Gemini
        try:
            models = await ChatBotManager._providers["gemini"].list_models(url, api_key)
            if models:
                return "gemini", models
        except Exception:
            pass

        raise RuntimeError(f"Failed to detect API type or list models from {url}")

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
        cls._clients.clear()

        for backend_config in json_obj.get("backends", []):
            config = BackendConfig.from_dict(backend_config)

            if config.api_type is not None and config.api_type not in ("openai", "anthropic", "gemini"):
                raise ValueError(f"Invalid api_type in config: {config.api_type}")

            cls._clients[config.name] = HTTPClient(
                timeout=cls._timeout,
                retry_delays=config.retry_delays,
            )
            backend_info = BackendInfo(
                name=config.name,
                url=config.url,
                api_type=config.api_type,
                models={},
            )
            cls._backends[config.name] = backend_info

            if config.api_type is None:
                config.api_type, models = await cls._detect_api_and_list_models(config.name, config.api_key)
            else:
                provider = cls._providers.get(config.api_type)
                if provider is None:
                    raise RuntimeError(f"No provider registered for api_type: {config.api_type}")
                models = await provider.list_models(config.url, config.api_key)

            # Create ChatBot instances via provider
            provider = cls._providers[config.api_type]
            chatbots: Dict[str, Any] = {}
            for model_id in models:
                chatbot_config = ChatBotConfig.from_dict({
                    **vars(config),
                    "model": model_id,
                    "priority": (config.model_priorities or {}).get(model_id, 0),
                })
                chatbot = provider.create_chatbot(cls._clients[config.name], chatbot_config)
                chatbots[model_id] = chatbot

            cls._backends[config.name].api_type = config.api_type
            cls._backends[config.name].models = chatbots

    @classmethod
    async def load_from_file(cls, filepath: str) -> None:
        """Load backend configuration from JSON file.

        Args:
            filepath: Path to JSON configuration file.
        """
        with open(filepath, "r") as f:
            json_obj = json.load(f)
        await cls.load_from_json(json_obj)
