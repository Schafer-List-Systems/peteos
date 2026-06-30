"""Output schema parsing, validation, and casting for OAP."""

from enum import Enum

import json
import sys
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
    if isinstance(hint, str):
        return eval(hint, globalns or {}, {})  # noqa: S307
    return hint  # generic alias (list[...], dict[...], etc.) — already a valid type


def parse_data(raw: str, schema: type | None) -> Any:
    """Full pipeline: parse JSON and cast to typed schema.

    Returns the parsed data cast to the schema type (dataclass instances,
    list[dataclass], etc.) or raises ValueError on any failure.

    When JSON parsing fails and the schema is a scalar type (str, int,
    float, bool, Enum), treats the raw string as the actual value and
    casts it directly. This handles LLMs that return bare strings
    (e.g. ``"string_argument"``) instead of JSON-encoded strings
    (e.g. ``"\"string_argument\""``).

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
    except json.JSONDecodeError:
        # Fallback for scalar schemas: interpret raw string directly.
        # Complex types (dataclass, list) fundamentally need JSON syntax,
        # so we only handle scalars and Enums here.
        origin = get_origin(schema)
        args = get_args(schema)
        is_enum = origin is None and isinstance(schema, type) and issubclass(schema, Enum)
        is_scalar = schema in (str, int, float, bool)
        if is_enum or is_scalar:
            return _recursive_cast(raw, schema)
        raise

    return _recursive_cast(parsed, schema)


def _type_name(schema: type, globalns: dict | None = None) -> str:
    """Return a human-readable name for a schema type."""
    if globalns is None:
        globalns = {}
    resolved = _resolve_type(schema, globalns)
    if isinstance(resolved, type) and issubclass(resolved, Enum):
        return resolved.__name__
    return resolved.__name__


def get_schema_description(schema: type | None) -> tuple[str, str] | None:
    """Extract the JSON schema dict and docstring for an output schema.

    Handles dataclasses, enums, lists, unions, and scalar primitives.

    Args:
        schema: The output schema to describe.

    Returns:
        A tuple of (json_schema_str, docstring) or None if the schema
        is not supported.
    """
    if schema is None:
        return None
    origin = get_origin(schema)
    args = get_args(schema)

    # Dataclass: describe fields
    if is_dataclass(schema):
        fields = dataclasses.fields(schema)
        globalns = dict(sys.modules[schema.__module__].__dict__)
        type_names = []
        for f in fields:
            resolved = _resolve_type(f.type, globalns)
            type_names.append(f"'{f.name}': {_type_name(resolved)}")
        json_schema = "{" + ", ".join(type_names) + "}"
        docstring = (schema.__doc__ or "").strip()
        return json_schema, docstring

    # Enum: describe as a string with allowed values
    if origin is None and isinstance(schema, type) and issubclass(schema, Enum):
        allowed = ", ".join(f'"{m.value}"' for m in schema)
        docstring = (schema.__doc__ or "").strip()
        return (
            f'{schema.__name__} is a placeholder for one of the following strings: {allowed}.',
            docstring if docstring else ''
        )

    # List with inner type
    if origin is list and args:
        inner = get_schema_description(args[0])
        if inner is not None:
            inner_schema, inner_doc = inner
            return f"[{inner_schema}, ...]", inner_doc

    # Union type
    is_union = origin is types.UnionType or (args and type(None) in args)
    if is_union:
        non_none = [t for t in args if t is not type(None)]
        parts = []
        for t in non_none:
            desc = get_schema_description(t)
            if desc is not None:
                parts.append(desc[0])
            else:
                parts.append(_type_name(t))
        return " | ".join(parts), ""

    # Scalar primitives
    if schema is str:
        return '"string"', "string type"
    if schema is int:
        return "123", "integer type"
    if schema is float:
        return "1.0", "float type"
    if schema is bool:
        return "true", "boolean type"
    if schema is list:
        return "[1, 2]", "list type"
    if schema is dict:
        return '{"key": "value"}', "dict type"
    if schema is str | int | float | bool | list | dict | type(None):
        parts = []
        for t in [str, int, float, bool, list, dict, type(None)]:
            if t in schema:
                desc = get_schema_description(t)
                if desc:
                    parts.append(desc[0])
        return " | ".join(parts), "one of the above types"

    # No matching schema type
    return None

def _recursive_cast(data: Any, schema: type) -> Any:
    """Recursively cast data to a schema, handling nested dataclasses.
    Resolves string annotations (e.g. 'str') to actual types when needed.

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

    # Dataclass: validate shape and cast each field
    if is_dataclass(schema) and isinstance(data, dict):
        declared = [f.name for f in dataclasses.fields(schema)]
        for name in declared:
            if name not in data:
                raise ValueError(f"missing required field '{name}' in schema {schema.__name__}")
        for key in data:
            if key not in declared:
                raise ValueError(
                    f"unexpected key '{key}' in schema {schema.__name__}. "
                    f"Expected fields: {', '.join(sorted(declared))}"
                )
        ns = dict(sys.modules.get(schema.__module__, object()).__dict__)
        ns[schema.__name__] = schema
        return schema(
            **{f.name: _recursive_cast(data[f.name], _resolve_type(f.type, ns)) for f in dataclasses.fields(schema)}
        )

    # List with inner type: validate container and cast each element
    if origin is list:
        if not isinstance(data, list):
            raise ValueError(f"expected list, got {type(data).__name__}")
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
            return data
        return data

    # Enum: match by value
    if origin is None and isinstance(schema, type) and issubclass(schema, Enum):
        try:
            return schema(data)
        except (ValueError, TypeError):
            raise ValueError(
                f"Expected {schema.__name__}, which is one of the following strings: "
                f"{', '.join(repr(m.value) for m in schema)}. "
                f"Got instead {type(data).__name__}: {data!r}."
            )

    # Scalar: strict type checks
    if schema is int:
        if not isinstance(data, int) or isinstance(data, bool):
            raise ValueError(f"expected int. Got instead: {type(data).__name__} {data!r}")
        return data
    if schema is float:
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            raise ValueError(f"expected float. Got instead: {type(data).__name__} {data!r}")
        return data
    if schema is bool:
        if not isinstance(data, bool):
            raise ValueError(f"expected bool. Got instead: {type(data).__name__} {data!r}")
        return data
    if schema is str:
        if not isinstance(data, str):
            raise ValueError(f"expected str. Got instead: {type(data).__name__} {data!r}")
        return data
    if schema is list:
        if not isinstance(data, list):
            raise ValueError(f"expected list. Got instead: {type(data).__name__} {data!r}")
        return data
    if schema is dict:
        if not isinstance(data, dict):
            raise ValueError(f"expected dict. Got instead: {type(data).__name__} {data!r}")
        return data
    if schema is str | int | float | bool | list | dict | type(None):
        return data

    # No matching schema type
    raise ValueError(f"no matching schema type for schema {schema}")
