"""Backend configuration dataclass for chatbot backends."""

from dataclasses import dataclass, fields, MISSING
from typing import Any, Dict, List, Optional


@dataclass
class BackendConfig:
    """Configuration for a single backend.

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
    """

    name: str
    url: str
    api_type: Optional[str] = None
    chat_endpoint: Optional[str] = None
    models_endpoint: Optional[str] = None
    streaming: bool = False
    max_tokens: int = 4096
    retry_delays: Optional[list[float]] = None
    api_key: Optional[str] = None
    model_priorities: Optional[Dict[str, int]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackendConfig":
        """Create a BackendConfig from a JSON dict, applying defaults.

        All default values are read from the dataclass fields so there is
        only one source of truth. Fields present in the dict override the
        defaults. Only dataclass field names from the JSON are accepted.

        Args:
            data: Raw config dict from JSON. Must include 'name' and 'url'.

        Returns:
            BackendConfig with defaults applied.
        """
        field_names = {f.name for f in fields(cls)}
        defaults = {
            f.name: f.default
            for f in fields(cls)
            if f.default is not MISSING
        }
        return cls(**{**defaults, **{k: v for k, v in data.items() if k in field_names}})