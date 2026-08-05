"""Output schema parsing, validation, and casting for OAP."""

from enum import Enum

import json
import sys
import types

from dataclasses import dataclass, is_dataclass
from typing import Any, get_args, get_origin

import dataclasses


def relaxed_parse_data(raw: Any, schema: type | None) -> Any:
    """Try strict parsing first, then unwrap single-key dicts as a fallback.

    Args:
        raw: The raw value — a JSON string or an already-parsed object.
        schema: The expected output schema type.

    Returns:
        The typed data cast to the schema type.
    """
    try:
        return parse_data(raw, schema)
    except ValueError:
        if schema is not None and isinstance(raw, dict):
            if len(raw) == 1:
                return parse_data(next(iter(raw.values())), schema)
            if len(raw) == 2 and "type" in raw:
                other = next(k for k in raw if k != "type")
                return parse_data(raw[other], schema)
        raise


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


def parse_data(raw: Any, schema: type | None) -> Any:
    """Full pipeline: parse JSON and cast to typed schema.

    Accepts either a raw JSON string or an already-parsed Python value
    (dict, list, scalar). For strings, attempts ``json.loads()`` first;
    on failure, treats the string directly as the value (scalars only).

    For already-parsed values, skips JSON parsing and casts directly.

    Args:
        raw: The raw value — a JSON string or an already-parsed object.
        schema: The expected output schema type.

    Returns:
        The typed data (dataclass instance, list[dataclass], or raw value).

    Raises:
        ValueError: If parsing, validation, or casting fails.
    """
    if schema is None:
        return raw

    # Already a parsed value — skip JSON decoding
    if not isinstance(raw, str):
        return _recursive_cast(raw, schema)

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Fallback for scalar schemas: interpret raw string directly.
        # Complex types (dataclass, list) fundamentally need JSON syntax,
        # so we only handle scalars and Enums here.
        return _recursive_cast(raw, schema)

    return _recursive_cast(parsed, schema)


def _type_name(schema: type, globalns: dict | None = None) -> str:
    """Return a human-readable name for a schema type."""
    if globalns is None:
        globalns = {}
    resolved = _resolve_type(schema, globalns)
    origin = get_origin(schema)
    args = get_args(schema)
    if isinstance(resolved, type) and issubclass(resolved, Enum):
        allowed = ", ".join(f'"{m.value}"' for m in resolved)
        return f"one of: {allowed}"
    if is_dataclass(resolved):
        fields = dataclasses.fields(resolved)
        inner_globalns = dict(sys.modules[resolved.__module__].__dict__)
        type_names = []
        for f in fields:
            resolved_field = _resolve_type(f.type, inner_globalns)
            type_names.append(f"'{f.name}': {_type_name(resolved_field)}")
        return "{" + ", ".join(type_names) + "}"
    # Generic types: dict, list, etc.
    if origin is dict and len(args) == 2:
        k, v = args
        return "{" + _type_name(k) + ": " + _type_name(v) + "}"
    if origin is list and args:
        return "[" + ", ".join(_type_name(a) for a in args) + ", ...]"
    return resolved.__name__


def _format_docs(docs: list[tuple[str, str]]) -> str:
    """Format a list of (type_name, docstring) tuples into a readable block.

    Each non-empty docstring gets a '# TypeName' header for clarity.
    """
    if not docs:
        return ""
    lines = []
    for type_name, doc in docs:
        if doc:
            lines.append(f"# {type_name}")
            lines.append(doc)
            lines.append("")
    return "\n".join(lines).strip()


def get_schema_description(schema: type | None, _seen: set | None = None) -> tuple[str, list[tuple[str, str]]]:
    """Extract the JSON schema type expression and structured docstrings.

    Handles dataclasses, enums, lists, unions, and scalar primitives.

    The first element is a concise type expression with dataclass names
    (e.g. "Person{'name': str}", "list[Person{'name': str}]").
    The second element is a list of (type_name, docstring) tuples for
    every type that has a user-provided docstring.

    Args:
        schema: The output schema to describe.
        _seen: Internal set to prevent infinite recursion on self-referential types.

    Returns:
        A tuple of (type_expression, [(type_name, docstring), ...]).
    """
    if _seen is None:
        _seen = set()

    if schema is None or schema is Any:
        return "any", [("any", "accept any value")]

    # Resolve string annotations so _seen works on the actual type
    if isinstance(schema, str):
        try:
            resolved_schema = _resolve_type(schema)
        except NameError:
            return schema, []
    else:
        resolved_schema = schema

    schema_id = id(resolved_schema)
    if schema_id in _seen:
        return resolved_schema.__name__ if hasattr(resolved_schema, "__name__") else str(resolved_schema), []
    _seen.add(schema_id)

    origin = get_origin(resolved_schema)
    args = get_args(resolved_schema)

    # Dataclass: describe as ClassName{field: type, ...}
    if is_dataclass(schema):
        fields = dataclasses.fields(schema)
        globalns = dict(sys.modules[schema.__module__].__dict__)
        globalns[schema.__name__] = schema
        type_names = []
        doc_entries: list[tuple[str, str]] = []
        class_doc = (schema.__doc__ or "").strip()
        # Skip auto-generated dataclass docstrings (pattern: "ClassName(field: type, ...)")
        if class_doc and not (class_doc.startswith(schema.__name__ + "(") and class_doc.endswith(")")):
            doc_entries.append((schema.__name__, class_doc))
        for f in fields:
            try:
                resolved = _resolve_type(f.type, globalns)
            except NameError:
                type_names.append(f"'{f.name}': {schema.__name__}")
                continue
            inner_schema, inner_docs = get_schema_description(resolved, _seen)
            type_names.append(f"'{f.name}': {inner_schema}")
            if inner_docs:
                doc_entries.extend(inner_docs)
        json_schema = f"{schema.__name__}{{{', '.join(type_names)}}}"
        return json_schema, doc_entries

    # Enum: describe as ClassName("val1" | "val2" | ...)
    if origin is None and isinstance(schema, type) and issubclass(schema, Enum):
        allowed = " | ".join(f'"{m.value}"' for m in schema)
        docstring = (schema.__doc__ or "").strip()
        entry: tuple[str, str] = (schema.__name__, docstring) if docstring else ()
        return f'{schema.__name__}({allowed})', [entry] if entry else []

    # List with inner type
    if origin is list and args:
        inner_schema, inner_docs = get_schema_description(args[0], _seen)
        return f"list[{inner_schema}]", inner_docs

    # Union type
    is_union = origin is types.UnionType or (args and type(None) in args)
    if is_union:
        non_none = [t for t in args if t is not type(None)]
        parts = []
        doc_entries: list[tuple[str, str]] = []
        for t in non_none:
            desc = get_schema_description(t, _seen)
            if desc is not None:
                parts.append(desc[0])
                if desc[1]:
                    doc_entries.extend(desc[1])
            else:
                parts.append(_type_name(t))
        return " | ".join(parts), doc_entries

    # Generic dict: describe as key-type → value-type
    if origin is dict and args:
        key_desc = get_schema_description(args[0], _seen)
        val_desc = get_schema_description(args[1], _seen)
        key_str = key_desc[0] if key_desc else _type_name(args[0])
        val_str = val_desc[0] if val_desc else _type_name(args[1])
        doc_entries = []
        if key_desc[1]:
            doc_entries.extend(key_desc[1])
        if val_desc[1]:
            doc_entries.extend(val_desc[1])
        return f'{{{key_str}: {val_str}}}', doc_entries

    # Scalar primitives — return empty docs (type name is self-documenting)
    if schema is str:
        return "str", []
    if schema is int:
        return "int", []
    if schema is float:
        return "float", []
    if schema is bool:
        return "bool", []
    if schema is list:
        return "list", []
    if schema is dict:
        return "dict", []
    if schema is str | int | float | bool | list | dict | type(None):
        parts = []
        doc_entries: list[tuple[str, str]] = []
        for t in [str, int, float, bool, list, dict, type(None)]:
            if t in schema:
                desc = get_schema_description(t, _seen)
                if desc:
                    parts.append(desc[0])
                    if desc[1]:
                        doc_entries.extend(desc[1])
        return " | ".join(parts), doc_entries

    raise ValueError(f"unsupported schema type: {schema}")


def _recursive_cast(data: Any, schema: type) -> Any:
    """Recursively cast data to a schema, handling nested dataclasses.
    Resolves string annotations (e.g. 'str') to actual types when needed.

    Walks the parsed data tree and instantiates dataclass objects at every
    level.  Supports scalar types, dicts, lists, and arbitrary nesting depth.

    Args:
        data: The parsed data to cast.
        schema: The target type schema.

    Returns:
        The data cast to the schema type.

    Raises:
        ValueError: If data doesn't match the schema.

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
    # Any: pass through unchanged — validation happens elsewhere
    if schema is Any:
        return data

    # type(None): pass through unchanged
    if schema is type(None):
        return data

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

    # Bare and typed list
    if schema is list:
        if not isinstance(data, list):
            raise ValueError(f"expected list. Got instead: {type(data).__name__} {data!r}")
        return data
    if origin is list:
        if not isinstance(data, list):
            raise ValueError(f"expected list, got {type(data).__name__}")
        return [_recursive_cast(item, args[0]) for item in data]

    # Bare and typed dict
    if schema is dict:
        if not isinstance(data, dict):
            raise ValueError(f"expected dict. Got instead: {type(data).__name__} {data!r}")
        return data
    if origin is dict and args:
        if not isinstance(data, dict):
            raise ValueError(f"expected dict, got {type(data).__name__}")
        val_schema = args[1]
        return {k: _recursive_cast(v, val_schema) for k, v in data.items()}

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
    if schema is str | int | float | bool | list | dict | type(None):
        return data

    # No matching schema type
    raise ValueError(f"no matching schema type for schema {schema}")
