"""Unified LLM API response types."""

from enum import StrEnum


class StopReason(StrEnum):
    """Unified stop/finish reason across Anthropic, OpenAI, and Gemini APIs.

    Normalized values: end_turn, max_tokens, tool_use, content_filter.
    """
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    TOOL_USE = "tool_use"
    CONTENT_FILTER = "content_filter"


_STOP_REASON_MAP: dict[str, str] = {
    # Anthropic
    "end_turn": "end_turn",
    "max_tokens": "max_tokens",
    "tool_use": "tool_use",
    # OpenAI
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "content_filter",
    # Gemini
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "content_filter",
}


def normalize_stop_reason(raw: str | None) -> str | None:
    """Map an API-specific stop/finish reason to the unified value."""
    if raw is None:
        return None
    return _STOP_REASON_MAP.get(raw, raw)
