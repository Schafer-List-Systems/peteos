"""Output schema extraction and LLM formatting for OAP."""

from __future__ import annotations

from typing import Any


def extract_schema_info(output_schema: type | None) -> dict[str, Any] | None:
    """Extract schema information from an output schema type."""
    if output_schema is None:
        return None
    return {"type": "object", "properties": {}}


def format_schema_prompt(schema_info: dict[str, Any] | None) -> str:
    """Format schema info as a human-readable prompt for the LLM."""
    return ""


def parse_output(result: str, output_schema: type) -> Any:
    """Parse an LLM's text response into the output schema type."""
    return result
