"""Tests for _recursive_cast with 10 progressively complex schemas."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest

from peteos.utils._schema import _recursive_cast, get_schema_description, parse_data


# ── Example 1: Simple scalar dataclass ───────────────────────────────

@dataclass
class TaskStatus:
    decision: str


# ── Example 2: List of dataclasses ───────────────────────────────────

@dataclass
class EdgeEvaluation:
    edge_id: str
    met: bool


# ── Example 3: Nested dataclass ──────────────────────────────────────

@dataclass
class Inner:
    name: str


@dataclass
class Outer:
    items: list[Inner]


# ── Example 6: Mixed scalar fields ───────────────────────────────────

@dataclass
class Metric:
    name: str
    count: int
    active: bool
    score: float


# ── Example 7: Nested with mixed fields ──────────────────────────────

@dataclass
class Report:
    title: str
    metrics: list[Metric]
    total: int


# ── Example 5: Deep nesting (3 levels) ───────────────────────────────

@dataclass
class L3:
    value: int


@dataclass
class L2:
    child: L3


@dataclass
class L1:
    children: list[L2]


# ── Example 8: Multiple nested lists ─────────────────────────────────

@dataclass
class Cell:
    val: int


@dataclass
class Row:
    cells: list[Cell]


@dataclass
class Grid:
    rows: list[Row]


# ── Example 9: Recursive (self-referential) ──────────────────────────

@dataclass
class TreeNode:
    value: str
    children: list[TreeNode]


# ── Example 10: Maximum complexity ───────────────────────────────────

@dataclass
class Node:
    label: str
    metrics: list[Metric]
    parent: Node | None


# ── Tests ────────────────────────────────────────────────────────────

class TestRecursiveCast:

    def test_1_simple_scalar(self):
        result = _recursive_cast({"decision": "ready"}, TaskStatus)
        assert isinstance(result, TaskStatus)
        assert result.decision == "ready"

    def test_2_list_of_dataclasses(self):
        result = _recursive_cast(
            [{"edge_id": "A", "met": True}, {"edge_id": "B", "met": False}],
            list[EdgeEvaluation],
        )
        assert len(result) == 2
        assert all(isinstance(e, EdgeEvaluation) for e in result)
        assert result[0].edge_id == "A"
        assert result[0].met is True

    def test_3_nested_dataclass(self):
        result = _recursive_cast(
            {"items": [{"name": "a"}, {"name": "b"}]},
            Outer,
        )
        assert isinstance(result, Outer)
        assert all(isinstance(i, Inner) for i in result.items)
        assert result.items[0].name == "a"

    def test_4_list_of_nested_dataclasses(self):
        result = _recursive_cast(
            [{"items": [{"name": "x"}]}, {"items": []}],
            list[Outer],
        )
        assert len(result) == 2
        assert isinstance(result[0], Outer)
        assert len(result[0].items) == 1
        assert isinstance(result[0].items[0], Inner)
        assert result[0].items[0].name == "x"
        assert len(result[1].items) == 0

    def test_5_deep_nesting_3_levels(self):
        result = _recursive_cast(
            {"children": [{"child": {"value": 42}}]},
            L1,
        )
        assert isinstance(result, L1)
        assert isinstance(result.children[0], L2)
        assert isinstance(result.children[0].child, L3)
        assert result.children[0].child.value == 42

    def test_6_mixed_scalar_fields(self):
        result = _recursive_cast(
            {"name": "cpu", "count": 100, "active": True, "score": 0.95},
            Metric,
        )
        assert isinstance(result, Metric)
        assert result.name == "cpu"
        assert result.count == 100
        assert result.active is True
        assert result.score == 0.95

    def test_7_nested_with_mixed_fields(self):
        result = _recursive_cast(
            {
                "title": "stats",
                "metrics": [{"name": "x", "count": 1, "active": True, "score": 0.5}],
                "total": 1,
            },
            Report,
        )
        assert isinstance(result, Report)
        assert result.title == "stats"
        assert result.total == 1
        assert isinstance(result.metrics[0], Metric)
        assert result.metrics[0].name == "x"

    def test_8_multiple_nested_lists(self):
        result = _recursive_cast(
            {"rows": [{"cells": [{"val": 1}, {"val": 2}]}]},
            Grid,
        )
        assert isinstance(result, Grid)
        assert isinstance(result.rows[0], Row)
        assert all(isinstance(c, Cell) for c in result.rows[0].cells)
        assert result.rows[0].cells[0].val == 1
        assert result.rows[0].cells[1].val == 2

    def test_9_recursive_self_referential(self):
        result = _recursive_cast(
            {"value": "root", "children": [{"value": "a", "children": []}]},
            TreeNode,
        )
        assert isinstance(result, TreeNode)
        assert result.value == "root"
        assert isinstance(result.children[0], TreeNode)
        assert result.children[0].value == "a"
        assert result.children[0].children == []

    def test_10_maximum_complexity(self):
        result = _recursive_cast(
            {
                "label": "tree",
                "metrics": [{"name": "depth", "count": 5, "active": True, "score": 3.0}],
                "parent": {
                    "label": "root",
                    "metrics": [],
                    "parent": None,
                },
            },
            Node,
        )
        assert isinstance(result, Node)
        assert result.label == "tree"
        assert isinstance(result.metrics[0], Metric)
        assert result.metrics[0].score == 3.0
        assert isinstance(result.parent, Node)
        assert result.parent.label == "root"
        assert result.parent.metrics == []
        assert result.parent.parent is None

    def test_11_scalar_types_direct(self):
        """Scalars pass through unchanged."""
        assert _recursive_cast(42, int) == 42
        assert _recursive_cast("hello", str) == "hello"
        assert _recursive_cast(True, bool) is True
        assert _recursive_cast(3.14, float) == 3.14

    def test_11b_unrecognised_type_passthrough(self):
        """Unrecognised types pass through unchanged."""
        assert _recursive_cast([1, 2, 3], list) == [1, 2, 3]
        assert _recursive_cast({"a": 1}, dict) == {"a": 1}
        # list[int] now validates and casts elements
        assert _recursive_cast([1, 2, 3], list[int]) == [1, 2, 3]

    def test_11c_no_schema_match_raises(self):
        """No matching schema type raises ValueError."""
        with pytest.raises(ValueError, match="no matching schema"):
            _recursive_cast(42, object)


class TestParseData:

    def test_parse_data_no_schema_returns_raw_string(self):
        assert parse_data("hello", None) == "hello"

    def test_parse_data_no_schema_returns_empty_string(self):
        assert parse_data("", None) == ""

    def test_parse_data_simple_schema(self):
        result = parse_data('{"decision": "ready"}', TaskStatus)
        assert isinstance(result, TaskStatus)
        assert result.decision == "ready"

    def test_parse_data_invalid_json_raises(self):
        with pytest.raises(ValueError):
            parse_data("not json", TaskStatus)

    def test_parse_data_validation_failure_raises(self):
        """A list schema raises if an item is missing a required field."""
        with pytest.raises(ValueError, match="missing required"):
            parse_data('[{"edge_id": "A"}]', list[EdgeEvaluation])

    def test_parse_data_list_of_dataclasses(self):
        result = parse_data(
            '[{"edge_id": "A", "met": true}]',
            list[EdgeEvaluation],
        )
        assert len(result) == 1
        assert isinstance(result[0], EdgeEvaluation)
        assert result[0].edge_id == "A"

    def test_parse_data_nested_deep(self):
        result = parse_data(
            '{"children": [{"child": {"value": 99}}]}',
            L1,
        )
        assert isinstance(result, L1)
        assert result.children[0].child.value == 99

    def test_parse_data_dataclass_extra_key_raises(self):
        """Agent returning extra keys should be caught for plain dataclass schema."""
        with pytest.raises(ValueError, match="unexpected key"):
            parse_data(
                '{"decision": "ready", "unknown_field": "oops"}',
                TaskStatus,
            )

    def test_parse_data_dataclass_missing_field_raises(self):
        """Agent omitting required fields should be caught."""
        with pytest.raises(ValueError, match="missing required"):
            parse_data('{}', TaskStatus)


# ── Scalar and shape validation via parse_data ─────────────────────────
# Validation moved into _recursive_cast; these tests exercise it
# through parse_data (the entry point).

class TestScalarValidation:

    def test_parse_data_no_schema_returns_raw(self):
        assert parse_data("hello", None) == "hello"

    def test_parse_str_valid(self):
        assert parse_data('"hello"', str) == "hello"

    def test_parse_str_invalid(self):
        with pytest.raises(ValueError, match="expected str. Got instead: int"):
            parse_data("42", str)

    def test_parse_int_valid(self):
        assert parse_data("42", int) == 42

    def test_parse_int_not_float(self):
        with pytest.raises(ValueError, match="expected int. Got instead: float"):
            parse_data("3.14", int)

    def test_parse_int_not_bool(self):
        with pytest.raises(ValueError, match="expected int. Got instead: bool"):
            parse_data("true", int)

    def test_parse_int_string_cast(self):
        """String representations of integers are cast."""
        assert parse_data("-7", int) == -7
        assert parse_data("0", int) == 0
        # Non-numeric strings still fail
        with pytest.raises(ValueError, match="expected int"):
            parse_data("notanumber", int)

    def test_parse_float_string_cast(self):
        """String representations of floats are cast."""
        assert parse_data("3.14", float) == 3.14
        assert parse_data("42", float) == 42.0
        # Non-numeric strings still fail
        with pytest.raises(ValueError, match="expected float"):
            parse_data("notanumber", float)

    def test_parse_bool_string_cast(self):
        """String 'true'/'false' are cast to bool."""
        assert parse_data("true", bool) is True
        assert parse_data("false", bool) is False
        assert parse_data("True", bool) is True
        assert parse_data("FALSE", bool) is False
        assert parse_data(" true ", bool) is True
        with pytest.raises(ValueError, match="expected bool"):
            parse_data("maybe", bool)

    def test_parse_float_valid(self):
        assert parse_data("3.14", float) == 3.14
        assert parse_data("3", float) == 3

    def test_parse_float_not_bool(self):
        with pytest.raises(ValueError, match="expected float. Got instead: bool"):
            parse_data("true", float)

    def test_parse_bool_valid(self):
        assert parse_data("true", bool) is True
        assert parse_data("false", bool) is False

    def test_parse_bool_not_int(self):
        with pytest.raises(ValueError, match="expected bool. Got instead: int"):
            parse_data("1", bool)

    def test_parse_list_valid(self):
        assert parse_data("[1, 2]", list) == [1, 2]

    def test_parse_list_not_str(self):
        with pytest.raises(ValueError, match="expected list. Got instead: str"):
            parse_data('"not a list"', list)

    def test_parse_dict_valid(self):
        assert parse_data('{"a": 1}', dict) == {"a": 1}

    def test_parse_dict_not_list(self):
        with pytest.raises(ValueError, match="expected dict. Got instead: list"):
            parse_data('[1, 2]', dict)

    def test_parse_union_type(self):
        assert parse_data('"hello"', str | int | None) == "hello"
        assert parse_data("42", str | int | None) == 42
        assert parse_data("null", str | int | None) is None

    def test_parse_list_int(self):
        """list[int] validates container and casts elements."""
        assert parse_data("[1, 2, 3]", list[int]) == [1, 2, 3]

    def test_parse_dataclass_extra_key_raises(self):
        with pytest.raises(ValueError, match="unexpected key"):
            parse_data('{"decision": "ready", "unknown_field": "oops"}', TaskStatus)

    def test_parse_dataclass_missing_field_raises(self):
        with pytest.raises(ValueError, match="missing required"):
            parse_data('{}', TaskStatus)

    def test_parse_dataclass_valid(self):
        result = parse_data('{"decision": "ready"}', TaskStatus)
        assert isinstance(result, TaskStatus)
        assert result.decision == "ready"

    def test_parse_dataclass_not_dict(self):
        with pytest.raises(ValueError):
            parse_data('"string"', TaskStatus)

    def test_parse_list_dataclass_extra_key_raises(self):
        with pytest.raises(ValueError, match="unexpected key"):
            parse_data('[{"edge_id": "A", "met": true, "bogus": "field"}]', list[EdgeEvaluation])

    def test_parse_list_dataclass_missing_field_raises(self):
        with pytest.raises(ValueError, match="missing required"):
            parse_data('[{"edge_id": "A"}]', list[EdgeEvaluation])

    def test_parse_list_dataclass_not_list_raises(self):
        with pytest.raises(ValueError, match="expected list"):
            parse_data('{"edge_id": "A"}', list[EdgeEvaluation])

    def test_parse_list_dataclass_empty_list(self):
        result = parse_data('[]', list[EdgeEvaluation])
        assert result == []

    def test_parse_list_dataclass_valid(self):
        result = parse_data('[{"edge_id": "A", "met": true}]', list[EdgeEvaluation])
        assert len(result) == 1
        assert isinstance(result[0], EdgeEvaluation)


# ── Bare-string fallback (parse_data) ──────────────────────────────────

class JobRole(Enum):
    FRONTEND = "frontend-developer"
    BACKEND = "backend-developer"
    FULLSTACK = "fullstack-developer"
    NOT_FITTING = "not-fitting"


class TestBareStringFallback:
    """When the LLM sends a bare string instead of JSON-encoded, parse_data
    should fall back to treating it as the raw value for scalar/Enum schemas."""

    def test_parse_enum_bare_string(self):
        """LLM sends fullstack-developer instead of \"fullstack-developer\"."""
        result = parse_data("fullstack-developer", JobRole)
        assert result is JobRole.FULLSTACK

    def test_parse_str_bare_string(self):
        """LLM sends hello instead of \"hello\"."""
        result = parse_data("hello", str)
        assert result == "hello"

    def test_parse_int_bare_number(self):
        """json.loads handles this normally, but fallback should also work."""
        result = parse_data("42", int)
        assert result == 42

    def test_parse_dataclass_still_requires_json(self):
        """Complex types still need JSON syntax; bare string should fail."""
        with pytest.raises(ValueError):
            parse_data("hello", TaskStatus)

    def test_parse_dict_still_requires_json(self):
        """Dict schema requires JSON syntax."""
        with pytest.raises(ValueError):
            parse_data("hello", dict)

    def test_parse_enum_dict_value_shows_value(self):
        """When LLM sends a dict instead of string, error shows the dict."""
        with pytest.raises(ValueError, match="Expected JobRole"):
            parse_data('{"role": "fullstack-developer"}', JobRole)

class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class TestEnumSupport:

    def test_recursive_cast_enum_valid(self):
        result = _recursive_cast("red", Color)
        assert result is Color.RED

    def test_recursive_cast_enum_invalid_raises(self):
        with pytest.raises(ValueError, match="Expected Color"):
            _recursive_cast("purple", Color)

    def test_recursive_cast_enum_invalid_shows_value(self):
        with pytest.raises(ValueError, match="dict.*role"):
            _recursive_cast({"role": "red"}, Color)

    def test_parse_data_enum_valid(self):
        result = parse_data('"red"', Color)
        assert result is Color.RED

    def test_parse_data_enum_invalid_raises(self):
        with pytest.raises(ValueError, match="Expected Color"):
            parse_data('"purple"', Color)

    def test_recursive_cast_enum_in_dataclass(self):
        @dataclass
        class Ticket:
            color: Color

        result = _recursive_cast({"color": "blue"}, Ticket)
        assert isinstance(result, Ticket)
        assert result.color is Color.BLUE

    def test_parse_data_enum_in_dataclass(self):
        @dataclass
        class Task:
            priority: str
            color: Color

        result = parse_data('{"priority": "high", "color": "green"}', Task)
        assert isinstance(result, Task)
        assert result.color is Color.GREEN
        assert result.priority == "high"

    def test_recursive_cast_enum_in_list(self):
        result = _recursive_cast(["red", "blue"], list[Color])
        assert result == [Color.RED, Color.BLUE]


# ── get_schema_description tests ────────────────────────────────────────

class TestSchemaDescription:

    def test_no_schema_returns_none(self):
        result = get_schema_description(None)
        assert result[0] == "any"

    def test_scalar_schema_description(self):
        assert get_schema_description(int) == ("int", [])
        assert get_schema_description(str) == ("str", [])
        assert get_schema_description(bool) == ("bool", [])
        assert get_schema_description(float) == ("float", [])

    def test_simple_dataclass(self):
        result = get_schema_description(TaskStatus)
        assert result is not None
        schema_str, doc = result
        assert "decision" in schema_str
        assert "str" in schema_str

    def test_list_of_dataclass(self):
        result = get_schema_description(list[EdgeEvaluation])
        assert result is not None
        schema_str, _ = result
        assert "[" in schema_str

    def test_dataclass_docstring(self):
        @dataclass
        class WithDoc:
            """A documented schema."""
            name: str

        _, doc_entries = get_schema_description(WithDoc)
        assert doc_entries == [("WithDoc", "A documented schema.")]
