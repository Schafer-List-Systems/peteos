"""ChatBot manager for multiple backend providers."""

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any

from .httpclient import HTTPClient
from .openaichatbot import OpenAIChatBot
from .anthropicchatbot import AnthropicChatBot


@dataclass
class BackendInfo:
    """Information about a backend provider."""
    name: str
    url: str
    api_type: str  # "openai" or "anthropic"
    models: Dict[str, Any]  # model_id -> ChatBot instance


class ChatBotManager:
    """Manager for multiple LLM backend providers.

    Manages backends, detects their supported APIs, discovers models,
    and provides filtered access to ChatBot instances.
    """

    def __init__(self):
        """Initialize empty manager."""
        self._backends: Dict[str, BackendInfo] = {}
        self._http_client = HTTPClient(timeout=60.0)

    async def add_backend(
        self,
        name: str,
        url: str,
        api_type: Optional[str] = None
    ) -> BackendInfo:
        """Add a new backend and auto-detect its capabilities.

        Args:
            name: Unique identifier for the backend.
            url: Base URL of the API (e.g., "http://localhost:8000").
            api_type: Optional API type override ("openai" or "anthropic").
                     If not provided, API type is auto-detected.

        Returns:
            BackendInfo with detected API type and discovered models.

        Raises:
            ValueError: If backend with same name already exists.
            RuntimeError: If API detection fails.
        """
        if name in self._backends:
            raise ValueError(f"Backend '{name}' already exists")

        if api_type is not None and api_type not in ("openai", "anthropic"):
            raise ValueError(f"Invalid api_type: {api_type}. Must be 'openai' or 'anthropic'")

        if api_type is None:
            api_type, models = await self._detect_api_and_list_models(url)
        else:
            models = await self._list_models_for_api_type(url, api_type)

        # Create ChatBot instances - let classes use their default endpoints
        chatbots: Dict[str, Any] = {}
        for model_id in models:
            chatbot = self._create_chatbot(api_type, model_id, url)
            chatbots[model_id] = chatbot

        backend_info = BackendInfo(
            name=name,
            url=url,
            api_type=api_type,
            models=chatbots
        )
        self._backends[name] = backend_info
        return backend_info

    def remove_backend(self, name: str) -> bool:
        """Remove a backend and all its ChatBot instances.

        Args:
            name: Backend identifier.

        Returns:
            True if backend was removed, False if not found.
        """
        if name in self._backends:
            del self._backends[name]
            return True
        return False

    async def _detect_api_and_list_models(self, url: str) -> Tuple[str, List[str]]:
        """Detect API type and list available models.

        Probes /v1/models endpoint and detects API based on response structure.

        Args:
            url: Base URL of the API.

        Returns:
            Tuple of (api_type, [model_ids]).

        Raises:
            RuntimeError: If API detection fails.
        """
        models_url = f"{url}/v1/models"

        try:
            response = await self._http_client.get(models_url)
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

    async def _list_models_for_api_type(self, url: str, api_type: str) -> List[str]:
        """List models for a specific API type without auto-detection.

        Args:
            url: Base URL of the API.
            api_type: "openai" or "anthropic".

        Returns:
            List of model IDs.

        Raises:
            RuntimeError: If model listing fails.
        """
        models_url = f"{url}/v1/models"

        try:
            response = await self._http_client.get(models_url)
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

    def _create_chatbot(self, api_type: str, model_id: str, base_url: str) -> Any:
        """Create appropriate ChatBot instance for model.

        Args:
            api_type: "openai" or "anthropic".
            model_id: Model identifier.
            base_url: Base URL of the API.

        Returns:
            ChatBot instance.
        """
        if api_type == "openai":
            return OpenAIChatBot(
                http_client=self._http_client,
                model=model_id,
                base_url=base_url
            )
        elif api_type == "anthropic":
            return AnthropicChatBot(
                http_client=self._http_client,
                model=model_id,
                base_url=base_url
            )
        else:
            raise ValueError(f"Unknown API type: {api_type}")

    def list_chatbots(self, model_regex: str) -> List[Tuple[str, Any]]:
        """List all ChatBot instances matching a regex pattern.

        Args:
            model_regex: Regex pattern to match model IDs.

        Returns:
            List of (model_id, ChatBot) tuples, sorted by model_id.
        """
        pattern = re.compile(model_regex)
        results = []

        for backend in self._backends.values():
            for model_id, chatbot in backend.models.items():
                if pattern.search(model_id):
                    results.append((model_id, chatbot))

        return sorted(results, key=lambda x: x[0])

    async def load_from_json(self, json_obj: dict) -> None:
        """Load backend configuration from JSON object.

        Clears current state and recreates all backends from the JSON.
        API types can be auto-detected or explicitly specified in config.

        Args:
            json_obj: Dict with "backends" key containing list of backend configs.
                     Each backend has "name" and "url", optionally "api_type".
        """
        self._backends.clear()

        for backend_config in json_obj.get("backends", []):
            name = backend_config["name"]
            url = backend_config["url"]
            api_type = backend_config.get("api_type")

            if api_type is not None and api_type not in ("openai", "anthropic"):
                raise ValueError(f"Invalid api_type in config: {api_type}")

            if api_type is None:
                api_type, models = await self._detect_api_and_list_models(url)
            else:
                models = await self._list_models_for_api_type(url, api_type)

            # Create ChatBot instances - let classes use their default endpoints
            chatbots: Dict[str, Any] = {}
            for model_id in models:
                chatbot = self._create_chatbot(api_type, model_id, url)
                chatbots[model_id] = chatbot

            self._backends[name] = BackendInfo(
                name=name,
                url=url,
                api_type=api_type,
                models=chatbots
            )

    async def load_from_file(self, filepath: str) -> None:
        """Load backend configuration from JSON file.

        Args:
            filepath: Path to JSON configuration file.
        """
        with open(filepath, "r") as f:
            json_obj = json.load(f)
        await self.load_from_json(json_obj)
