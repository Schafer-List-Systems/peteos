"""Output schema extraction and LLM formatting for OAP."""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, get_args, get_origin, get_type_hints

logger = logging.getLogger(__name__)

# Type-to-JSON-schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def extract_schema_info(output_schema: type | None) -> dict[str, Any] | None:
    """Extract schema information from an output schema type.

    Returns a dict with schema fields suitable for the LLM system prompt,
    or None if no schema was specified.
    """
    if output_schema is None:
        return None

    info: dict[str, Any] = {"type": "object", "properties": {}}

    if hasattr(output_schema, "__annotations__"):
        try:
            hints = get_type_hints(output_schema)
        except Exception:
            hints = getattr(output_schema, "__annotations__", {})

        for field_name, field_type in hints.items():
            json_type = _resolve_type(field_type)
            info["properties"][field_name] = {
                "type": json_type,
                "description": _extract_field_description(output_schema, field_name, field_type),
            }

    # Build required list from non-optional fields (no default, no Optional)
    required = _extract_required_fields(output_schema)
    if required:
        info["required"] = required

    return info


def format_schema_prompt(schema_info: dict[str, Any] | None) -> str:
    """Format schema info as a human-readable prompt for the LLM system prompt.

    Returns an empty string if no schema info provided.
    """
    if schema_info is None:
        return ""

    props = schema_info.get("properties", {})
    required = schema_info.get("required", [])
    lines = [
        "Return the result as a JSON object with the following fields:",
        "",
    ]

    for field_name, field_info in props.items():
        req = " (required)" if field_name in required else " (optional)"
        desc = field_info.get("description", "")
        lines.append(f"- `{field_name}` ({field_info['type']}){req}: {desc}")

    lines.append("")
    lines.append("Format your response as a valid JSON object matching this schema.")
    return "\n".join(lines)


def _resolve_type(field_type: type) -> str:
    """Resolve a Python type to a JSON schema type string."""
    origin = get_origin(field_type)
    if origin is not None:
        if origin is list:
            args = get_args(field_type)
            if args:
                inner = _resolve_type(args[0])
                return f"array<{inner}>"
            return "array"
        if origin is dict:
            return "object"
        if origin is type(None) or origin is type(None):
            return "null"
        return _TYPE_MAP.get(origin, "string")

    return _TYPE_MAP.get(field_type, "string")


def _extract_field_description(
    output_schema: type,
    field_name: str,
    field_type: type,
) -> str:
    """Extract a description for a field from its type annotations."""
    # Check for Pydantic Field or similar with description
    field_meta = getattr(output_schema, "__dict__", {}).get(field_name)
    if field_meta is not None:
        # Pydantic v2 FieldInfo
        if hasattr(field_meta, "description"):
            return field_meta.description  # type: ignore[union-attr]
        if isinstance(field_meta, dict) and "description" in field_meta:
            return field_meta["description"]

    # Check for docstring-style comments in annotations
    hints = getattr(output_schema, "__annotations__", {})
    hint_value = hints.get(field_name)
    if hint_value is not None and isinstance(hint_value, str):
        # PEP 563 style: annotations="str | None # field description"
        if "#" in hint_value:
            return hint_value.split("#", 1)[1].strip()

    # Check for __doc__ on the field (NamedTuple)
    if hasattr(output_schema, "_field_comments"):
        return output_schema._field_comments.get(field_name, "")  # type: ignore[union-attr]

    return ""


def _extract_required_fields(output_schema: type) -> list[str]:
    """Extract list of required field names from an output schema type."""
    required: list[str] = []
    if not hasattr(output_schema, "__annotations__"):
        return required

    try:
        hints = get_type_hints(output_schema)
    except Exception:
        hints = getattr(output_schema, "__annotations__", {})

    if not hasattr(output_schema, "__init__"):
        return required

    sig = inspect.signature(output_schema.__init__)
    defaults = {}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "args", "kwargs"):
            continue
        if param.default is not inspect.Parameter.empty:
            defaults[param_name] = param.default

    for field_name in hints:
        if field_name not in defaults:
            required.append(field_name)

    return required


def parse_output(result: str, output_schema: type) -> Any:
    """Parse an LLM's text response into the output schema type.

    Args:
        result: The raw text response from the LLM.
        output_schema: The target type to parse into.

    Returns:
        An instance of the output schema type, or the raw string if parsing fails.
    """
    if output_schema is None:
        return result

    if not hasattr(output_schema, "__annotations__"):
        return result

    # Try to parse as JSON
    try:
        json_data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        # Not JSON - return as-is for the LLM to reformat
        return result

    if not isinstance(json_data, dict):
        return result

    hints = getattr(output_schema, "__annotations__", {})
    obj_kwargs: dict[str, Any] = {}

    for field_name, field_type in hints.items():
        if field_name in json_data:
            obj_kwargs[field_name] = _cast_value(json_data[field_name], field_type)

    try:
        return output_schema(**obj_kwargs)  # type: ignore[operator]
    except TypeError:
        return result


def _cast_value(value: Any, target_type: type) -> Any:
    """Cast a value to the target type."""
    if value is None:
        origin = get_origin(target_type)
        if origin is list:
            return []
        if target_type in (int, float, str):
            return target_type()  # 0 for int/float, "" for str
        return value

    origin = get_origin(target_type)

    if origin is list:
        args = get_args(target_type)
        if args:
            return [_cast_value(v, args[0]) for v in value]
        return value

    if origin is dict:
        return value

    # Direct type cast
    if target_type is bool:
        return bool(value)
    if target_type in (int, float, str):
        return target_type(value)

    return value
