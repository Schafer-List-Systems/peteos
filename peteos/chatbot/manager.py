"""ChatBot manager for multiple backend providers."""

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from .httpclient import HTTPClient
from .openaichatbot import OpenAIChatBot
from .anthropicchatbot import AnthropicChatBot
from .backendconfig import BackendConfig
from .chatbotconfig import ChatBotConfig
from .geminichatbot import GeminiChatBot


@dataclass
class BackendInfo:
    """Information about a backend provider."""
    name: str
    url: str
    api_type: str  # "openai" or "anthropic"
    models: Dict[str, Any]  # model_id -> ChatBot instance


class ChatBotManager:
    """Manager for multiple LLM backend providers.

    Uses class-level state so backends are globally accessible without
    passing an instance around.
    """

    # Class-level state
    _backends: Dict[str, BackendInfo] = {}
    _clients: Dict[str, HTTPClient] = {}
    _timeout: Optional[float] = None

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
            api_type: Optional API type override ("openai" or "anthropic").
                     If not provided, API type is auto-detected.
            **kwargs: Configuration options (streaming, max_tokens, etc.).

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

        # Add placeholder before detection so _detect methods can look up url
        cls._backends[name] = BackendInfo(
            name=config.name,
            url=config.url,
            api_type=config.api_type,
            models={},
        )

        # Auto-detect api_type if not provided
        if config.api_type is None:
            config.api_type, models = await cls._detect_api_and_list_models(name)
        else:
            models = await cls._list_models_for_api_type(name, config.api_type)

        # Create ChatBot instances
        chatbots: Dict[str, Any] = {}
        for model_id in models:
            chatbot = cls._create_chatbot(config, model_id)
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
    async def _detect_api_and_list_models(backend_name: str) -> Tuple[str, List[str]]:
        """Detect API type and list available models.

        Probes /v1/models endpoint and detects API based on response structure.

        Args:
            backend_name: Backend identifier used to look up HTTP client.

        Returns:
            Tuple of (api_type, [model_ids]).

        Raises:
            RuntimeError: If API detection fails.
        """
        models_url = f"{ChatBotManager._backends[backend_name].url}/v1/models"
        client = ChatBotManager._clients[backend_name]

        try:
            response = await client.get(models_url)
        except Exception as e:
            raise RuntimeError(f"Failed to probe {models_url}: {e}")

        # Detect API type by response structure
        if "data" in response and isinstance(response.get("data"), list):
            # OpenAI format: {"data": [{"id": "...", ...}, ...]}
            api_type = "openai"
            model_ids = [m["id"] for m in response["data"] if "id" in m]
        elif "models" in response and isinstance(response.get("models"), list):
            # Anthropic format: {"models": [{"id": "...", ...}, ...]}
            api_type = "anthropic"
            model_ids = [m["id"] for m in response["models"] if "id" in m]
        else:
            raise RuntimeError(
                f"Could not detect API type from response: {response}"
            )

        if not model_ids:
            raise RuntimeError(f"No models found at {models_url}")

        return api_type, model_ids

    @staticmethod
    async def _list_models_for_api_type(backend_name: str, api_type: str) -> List[str]:
        """List models for a specific API type without auto-detection.

        Args:
            backend_name: Backend identifier used to look up HTTP client.
            api_type: "openai" or "anthropic".

        Returns:
            List of model IDs.

        Raises:
            RuntimeError: If model listing fails.
        """
        models_url = f"{ChatBotManager._backends[backend_name].url}/v1/models"
        client = ChatBotManager._clients[backend_name]

        try:
            response = await client.get(models_url)
        except Exception as e:
            raise RuntimeError(f"Failed to probe {models_url}: {e}")

        # Extract model IDs from response - support both OpenAI and Anthropic formats
        model_ids = []

        # Try OpenAI format first
        if "data" in response and isinstance(response.get("data"), list):
            model_ids = [m["id"] for m in response["data"] if "id" in m]
        # Then try Anthropic format
        elif "models" in response and isinstance(response.get("models"), list):
            model_ids = [m["id"] for m in response["models"] if "id" in m]

        if not model_ids:
            raise RuntimeError(f"No models found at {models_url}, response: {response}")

        return model_ids

    @staticmethod
    def _create_chatbot(config: BackendConfig, model_id: str) -> Any:
        """Create appropriate ChatBot instance for model.

        Args:
            config: Backend configuration dataclass with all defaults applied.
            model_id: Model identifier.

        Returns:
            ChatBot instance.
        """
        chatbot_config = ChatBotConfig.from_dict({
            **vars(config),
            "model": model_id,
        })
        client = ChatBotManager._clients[config.name]
        if config.api_type == "openai":
            return OpenAIChatBot(client, chatbot_config)
        elif config.api_type == "anthropic":
            return AnthropicChatBot(client, chatbot_config)
        elif config.api_type == "gemini":
            return GeminiChatBot(client, chatbot_config)
        else:
            raise ValueError(f"Unknown API type: {config.api_type}")

    @classmethod
    def list_chatbots(cls, model_regex: str) -> List[Tuple[str, Any]]:
        """List all ChatBot instances matching a regex pattern.

        Args:
            model_regex: Regex pattern to match model IDs.

        Returns:
            List of (model_id, ChatBot) tuples, sorted by model_id.
        """
        pattern = re.compile(model_regex)
        results = []

        for backend in cls._backends.values():
            for model_id, chatbot in backend.models.items():
                if pattern.search(model_id):
                    results.append((model_id, chatbot))

        return sorted(results, key=lambda x: x[0])

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
                config.api_type, models = await cls._detect_api_and_list_models(config.name)
            else:
                models = await cls._list_models_for_api_type(config.name, config.api_type)

            # Create ChatBot instances
            chatbots: Dict[str, Any] = {}
            for model_id in models:
                chatbot = cls._create_chatbot(config, model_id)
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
