"""Output schema parsing, validation, and casting for OAP."""

import json
import types

from dataclasses import dataclass, is_dataclass
from typing import Any, get_args, get_origin

import dataclasses


def _resolve_type(hint: Any, globalns: dict | None = None) -> type:
    """Resolve a type hint — already a class, or a string annotation.

    ``f.type`` from ``dataclasses.fields()`` may be a class or a string
    (when ``from __future__ import annotations`` is used in the caller's
    module).  This handles both.

    Args:
        hint: The type hint (class or string).
        globalns: Optional namespace for evaluating string annotations.
    """
    if isinstance(hint, type):
        return hint
    return eval(hint, globalns or {}, {})  # noqa: S307


def parse_data(raw: str, schema: type | None) -> Any:
    """Full pipeline: parse JSON → validate → cast to typed schema.

    Returns the parsed data cast to the schema type (dataclass instances,
    list[dataclass], etc.) or raises ValueError on any failure.

    Args:
        raw: The raw JSON string from the agent.
        schema: The expected output schema type.

    Returns:
        The typed data (dataclass instance, list[dataclass], or raw value).

    Raises:
        ValueError: If parsing, validation, or casting fails.
    """
    if schema is None:
        return raw

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e

    error = validate_data(parsed, schema)
    if error:
        raise ValueError(error)

    return _recursive_cast(parsed, schema)


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


def validate_data(data: Any, schema: type | None) -> str | None:
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


def _recursive_cast(data: Any, schema: type) -> Any:
    """Recursively cast data to a schema, handling nested dataclasses.

    Walks the parsed data tree and instantiates dataclass objects at every
    level.  Supports scalar types, dicts, lists, and arbitrary nesting depth.

    10 progressively complex examples:

    1. Simple scalar dataclass:
        schema = TaskStatus(decision: str)
        data   = {"decision": "ready"}
        → TaskStatus(decision="ready")

    2. List of dataclasses:
        schema = list[EdgeEvaluation(edge_id: str, met: bool)]
        data   = [{"edge_id": "A", "met": True}, {"edge_id": "B", "met": False}]
        → [EdgeEvaluation(edge_id="A", met=True), EdgeEvaluation(edge_id="B", met=False)]

    3. Nested dataclass (dataclass with a list field):
        class Inner(D): name: str
        class Outer(D): items: list[Inner]
        data   = {"items": [{"name": "a"}, {"name": "b"}]}
        → Outer(items=[Inner(name="a"), Inner(name="b")])

    4. List of nested dataclasses:
        schema = list[Outer]
        data   = [{"items": [{"name": "x"}]}, {"items": []}]
        → [Outer(items=[Inner(name="x")]), Outer(items=[])]

    5. Deep nesting (3 levels):
        class L3(D): value: int
        class L2(D): child: L3
        class L1(D): children: list[L2]
        data   = {"children": [{"child": {"value": 42}}]}
        → L1(children=[L2(child=L3(value=42))])

    6. Mixed scalar fields in dataclass:
        class Metric(D):
            name: str
            count: int
            active: bool
            score: float
        data   = {"name": "cpu", "count": 100, "active": true, "score": 0.95}
        → Metric(name="cpu", count=100, active=True, score=0.95)

    7. Nested dataclass with mixed fields:
        class Report(D):
            title: str
            metrics: list[Metric]
            total: int
        data   = {"title": "stats", "metrics": [{"name": "x", "count": 1, "active": true, "score": 0.5}], "total": 1}
        → Report(title="stats", metrics=[Metric(name="x", count=1, active=True, score=0.5)], total=1)

    8. Multiple nested lists:
        class Cell(D): val: int
        class Row(D): cells: list[Cell]
        class Grid(D): rows: list[Row]
        data   = {"rows": [{"cells": [{"val": 1}, {"val": 2}]}]}
        → Grid(rows=[Row(cells=[Cell(val=1), Cell(val=2)])])

    9. Recursive/self-referential structure:
        class TreeNode(D):
            value: str
            children: list['TreeNode']
        data   = {"value": "root", "children": [{"value": "a", "children": []}]}
        → TreeNode(value="root", children=[TreeNode(value="a", children=[])])

    10. Maximum complexity (deep nesting + mixed types + lists):
        class Metric(D): name: str; value: float
        class Node(D): label: str; metrics: list[Metric]; parent: 'Node | None'
        data   = {"label": "tree", "metrics": [{"name": "depth", "value": 3.0}], "parent": {"label": "root", "metrics": [], "parent": null}}
        → Node(label="tree", metrics=[Metric(name="depth", value=3.0)], parent=Node(label="root", metrics=[], parent=None))
    """
    # Compute origin/args once — reused by list and union branches below
    origin = get_origin(schema)
    args = get_args(schema)

    # Dataclass: map each dict field through its declared type
    if is_dataclass(schema) and isinstance(data, dict):
        # Build namespace from the schema's module so string refs
        # (like "Inner" in Outer.items: list[Inner]) resolve correctly.
        # Add the schema itself for self-referential types (TreeNode → TreeNode).
        import sys
        ns = dict(sys.modules.get(schema.__module__, object()).__dict__)
        ns[schema.__name__] = schema
        return schema(
            **{f.name: _recursive_cast(data.get(f.name), _resolve_type(f.type, ns)) for f in dataclasses.fields(schema)}
        )

    # Generic list of dataclasses: cast each element
    if origin is list and args and is_dataclass(args[0]) and isinstance(data, list):
        return [_recursive_cast(item, args[0]) for item in data]

    # Union type (e.g. Node | None): extract non-None type and cast
    is_union = origin is types.UnionType or (args and type(None) in args)
    if is_union:
        non_none = [t for t in args if t is not type(None)]
        if data is None:
            return None
        if isinstance(data, dict) and non_none:
            for t in non_none:
                if is_dataclass(t):
                    return _recursive_cast(data, t)
            # Non-dataclass union member (e.g. str | int) — pass through
            return data

    # Scalar or unrecognized type: pass through unchanged
    return data
