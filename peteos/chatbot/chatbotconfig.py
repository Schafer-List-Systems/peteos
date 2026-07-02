"""ChatBot configuration dataclass for per-chatbot settings."""

from dataclasses import dataclass, fields, MISSING
from typing import Any, Dict, Optional

from .backendconfig import BackendConfig


@dataclass
class ChatBotConfig(BackendConfig):
    """Configuration for a single ChatBot instance.

    Inherits backend config (url, api_type, api_key, etc.) and adds
    per-chatbot options like translations and model selection.

    All defaults are defined here so there is a single source of truth.
    The config object is built once from the JSON and passed down through
    the entire constructor chain.

    Attributes:
        name: Unique backend identifier.
        url: API base URL.
        api_type: "openai", "anthropic", or "gemini".
        chat_endpoint: Custom chat endpoint (default: API-specific).
        models_endpoint: Custom models endpoint (default: API-specific).
        streaming: Use streaming mode by default.
        max_tokens: Maximum tokens to generate (set per-API-type in chatbot __init__).
        api_key: API key for authentication
        model: Model identifier for this ChatBot instance.
        response_translations: Per-chatbot SSE event translations.
        request_translations: Per-chatbot request key translations.
    """

    model: Optional[str] = None
    response_translations: Optional[Dict[str, str]] = None
    request_translations: Optional[Dict[str, str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatBotConfig":
        """Create a ChatBotConfig from a JSON dict, applying defaults.

        All default values are read from the dataclass fields so there is
        only one source of truth. Fields present in the dict override the
        defaults. Only dataclass field names from the JSON are accepted.

        Args:
            data: Raw config dict from JSON.

        Returns:
            ChatBotConfig with defaults applied.
        """
        field_names = {f.name for f in fields(cls)}
        defaults = {
            f.name: f.default
            for f in fields(cls)
            if f.default is not MISSING
        }
        return cls(**{**defaults, **{k: v for k, v in data.items() if k in field_names}})