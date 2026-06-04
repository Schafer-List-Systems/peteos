"""Output schema extraction and LLM formatting for OAP."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import Any, get_args, get_origin

import dataclasses


def get_schema_description(schema: type | None) -> tuple[str, str] | None:
    """Extract the JSON schema dict and docstring for a dataclass output schema.

    Handles dataclasses and list[SomeDataclass].

    Args:
        schema: The output schema to describe.

    Returns:
        A tuple of (json_schema_str, docstring) or None if the schema
        is not a dataclass or list[SomeDataclass].
    """
    if schema is None:
        return None
    origin = get_origin(schema)
    args = get_args(schema)
    if origin is list and args and is_dataclass(args[0]):
        inner = args[0]
        desc = get_schema_description(inner)
        if desc is None:
            return None
        inner_schema, inner_doc = desc
        return f"[{inner_schema}, ...]", inner_doc
    if not is_dataclass(schema):
        return None
    fields = dataclasses.fields(schema)
    json_schema = "{" + ", ".join(f"'{f.name}': {f.type.__name__}" for f in fields) + "}"
    docstring = (schema.__doc__ or "").strip()
    return json_schema, docstring


def validate_produced_data(data: Any, schema: type | None) -> str | None:
    """Validate data against the current output schema.

    Returns an error message string if validation fails, None if valid.

    Args:
        data: The parsed data to validate.
        schema: The expected output schema type.

    Returns:
        An error message string or None.
    """
    if schema is None:
        return None
    origin = get_origin(schema)
    args = get_args(schema)
    if origin is list and args and is_dataclass(args[0]):
        if not isinstance(data, list):
            return "Error: expected list, got " + type(data).__name__
        inner_type = args[0]
        for item in data:
            if isinstance(item, dict):
                for field in dataclasses.fields(inner_type):
                    if field.name not in item:
                        return f"Error: missing required field '{field.name}' in list item"
                return None
        return None
    if schema is str:
        if not isinstance(data, str):
            return "Error: expected str, got " + type(data).__name__
        return None
    if schema is int:
        if not isinstance(data, int) or isinstance(data, bool):
            return "Error: expected int, got " + type(data).__name__
        return None
    if schema is float:
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            return "Error: expected float, got " + type(data).__name__
        return None
    if schema is bool:
        if not isinstance(data, bool):
            return "Error: expected bool, got " + type(data).__name__
        return None
    if schema is list:
        if not isinstance(data, list):
            return "Error: expected list, got " + type(data).__name__
        return None
    if schema is dict:
        if not isinstance(data, dict):
            return "Error: expected dict, got " + type(data).__name__
        return None
    if schema is str | int | float | bool | list | dict | type(None):
        return None
    return None


def cast_produced_data(data: Any, schema: type | None) -> Any:
    """Cast parsed data to the expected schema type.

    Handles dataclasses and list[SomeDataclass].

    Args:
        data: The parsed data to cast.
        schema: The expected output schema type.

    Returns:
        The data cast to the appropriate type.
    """
    if schema is None:
        return data
    origin = get_origin(schema)
    args = get_args(schema)
    if origin is list and args and is_dataclass(args[0]):
        inner_type = args[0]
        if isinstance(data, list):
            data = [
                inner_type(**item) if isinstance(item, dict) else item
                for item in data
            ]
    elif is_dataclass(schema) and isinstance(data, dict):
        data = schema(**data)
    return data