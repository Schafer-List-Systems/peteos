"""Backend configuration dataclass for chatbot backends."""

from dataclasses import dataclass, fields, MISSING
from typing import Any, Dict, Optional


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
        max_tokens: Maximum tokens to generate.
    """

    name: str
    url: Optional[str] = None
    api_type: Optional[str] = None
    chat_endpoint: Optional[str] = None
    models_endpoint: Optional[str] = None
    streaming: bool = False
    max_tokens: int = 4096
    retry_delays: Optional[list[float]] = None
    api_key: Optional[str] = None
    model_priorities: Optional[Dict[str, int]] = None
    timeout: Optional[float] = None

    # Fallback timeout applied when the config value is None.
    # Set by ChatBotManager.__init__ to keep the default in sync.
    _default_timeout: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackendConfig":
        """Create a BackendConfig from a JSON dict, applying defaults.

        All default values are read from the dataclass fields so there is
        only one source of truth. Fields present in the dict override the
        defaults. Only dataclass field names from the JSON are accepted.

        For fields whose value is None in the JSON, the class-level default
        is used instead (e.g., timeout falls back to ``_default_timeout``).

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
        # Fill in None values with class-level defaults
        for field_name in field_names:
            if defaults.get(field_name) is None and hasattr(cls, f"_default_{field_name}"):
                defaults[field_name] = getattr(cls, f"_default_{field_name}")
        merged = {**defaults, **{k: v for k, v in data.items() if k in field_names}}

        # Normalize URL: strip trailing slash so concatenation with endpoint paths is safe.
        url = merged.get("url")
        if url is not None:
            merged["url"] = url.rstrip("/")

        # Normalize endpoint paths: ensure they start with "/" so concatenation with
        # the base URL produces a valid absolute path (e.g. "url/v1/chat/completions").
        for endpoint_field in ("chat_endpoint", "models_endpoint"):
            ep = merged.get(endpoint_field)
            if ep is not None and not ep.startswith("/"):
                merged[endpoint_field] = "/" + ep

        # Ensure model_priorities values are ints (JSON may parse them as strings).
        priorities = merged.get("model_priorities")
        if priorities is not None:
            merged["model_priorities"] = {
                k: int(v) for k, v in priorities.items()
            }

        return cls(**merged)
