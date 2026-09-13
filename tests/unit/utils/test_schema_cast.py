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

    def test_parse_bool_string_one_parses(self):
        """String \"1\" parses to True for bool schema (per Table 4)."""
        assert parse_data("1", bool) is True

    def test_bool_int_zero_one(self):
        """bool schema accepts 0 and 1 as False and True."""
        assert _recursive_cast(0, bool) is False
        assert _recursive_cast(1, bool) is True

    def test_bool_int_other_raises(self):
        """int values other than 0 or 1 raise ValueError for bool schema."""
        with pytest.raises(ValueError, match="expected bool"):
            _recursive_cast(42, bool)

    def test_parse_list_valid(self):
        assert parse_data("[1, 2]", list) == [1, 2]

    def test_parse_list_not_str(self):
        with pytest.raises(ValueError, match="expected list. Got instead: str"):
            parse_data('"not a list"', list)

    def test_list_bare_parses_json_string(self):
        """Bare list schema handles JSON-string input directly."""
        result = _recursive_cast("[1, 2, 3]", list)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_list_typed_parses_json_string(self):
        """Typed list[int] handles JSON-string input directly."""
        result = _recursive_cast("[1, 2, 3]", list[int])
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_list_bad_json_string_raises(self):
        """Invalid JSON string for list raises ValueError."""
        with pytest.raises(ValueError):
            _recursive_cast("not-valid-json", list[int])

    def test_list_comma_sep_parses(self):
        """Comma-separated string fallback for bare list."""
        result = _recursive_cast("1, 2, 3", list)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_list_typed_comma_sep_parses(self):
        """Comma-separated string for typed list[int]."""
        result = _recursive_cast("1, 2, 3", list[int])
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_list_comma_sep_bad_raises(self):
        """Comma-separated string that can't be coerced raises ValueError."""
        with pytest.raises(ValueError):
            _recursive_cast("a, b, c", list[int])

    def test_list_parens_string_parses(self):
        """Parentheses string: replaces () with [] then JSON-parses."""
        result = _recursive_cast("(1, 2, 3)", list)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_list_typed_parens_string_parses(self):
        """Typed list[int] with parentheses string."""
        result = _recursive_cast("(1, 2, 3)", list[int])
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_list_parens_string_bad_raises(self):
        """Parentheses string that can't be coerced raises ValueError."""
        with pytest.raises(ValueError):
            _recursive_cast("(a, b, c)", list[int])

    def test_parse_dict_valid(self):
        assert parse_data('{"a": 1}', dict) == {"a": 1}

    def test_parse_dict_not_list(self):
        with pytest.raises(ValueError, match="expected dict. Got instead: list"):
            parse_data('[1, 2]', dict)

    def test_dict_bare_parses_json_string(self):
        """Bare dict schema handles JSON-string input directly."""
        result = _recursive_cast('{"a": 1, "b": 2}', dict)
        assert result == {"a": 1, "b": 2}

    def test_dict_typed_parses_json_string(self):
        """Typed dict[str, int] handles JSON-string input directly."""
        result = _recursive_cast('{"x": 10, "y": 20}', dict[str, int])
        assert result == {"x": 10, "y": 20}

    def test_dict_bad_json_string_raises(self):
        """Invalid JSON string for dict raises ValueError."""
        with pytest.raises(ValueError):
            _recursive_cast("not-valid-json", dict[str, int])

    def test_parse_union_type(self):
        assert parse_data('"hello"', str | int | None) == "hello"
        assert parse_data("42", str | int | None) == 42
        assert parse_data("null", str | int | None) is None

    def test_nothing_values_become_none_in_union(self):
        """0.0, [], {} become None when type(None) is in union (no exact match available).

        Note: '' matches str exactly (type matches), so '' in str | None returns ''.
        And 0.0 matches float exactly, so 0.0 in int | float returns 0.0.
        Nothing values only fall through to NoneType when no other member matches.
        """
        result = _recursive_cast(0.0, int | None)
        assert result is None
        result = _recursive_cast([], int | None)
        assert result is None
        result = _recursive_cast({}, int | None)
        assert result is None
        result = _recursive_cast([], str | None)
        assert result is None

    def test_type_none_accepts_nothing_values(self):
        """Bare type(None) schema accepts nothing values as None."""
        assert _recursive_cast("", type(None)) is None
        assert _recursive_cast(0.0, type(None)) is None
        assert _recursive_cast([], type(None)) is None
        assert _recursive_cast({}, type(None)) is None

    def test_nothing_values_error_in_non_none_unions(self):
        """[], {} raise ValueError when union has no NoneType and no matching member."""
        with pytest.raises(ValueError):
            _recursive_cast([], int | float)
        with pytest.raises(ValueError):
            _recursive_cast({}, int | float)

    def test_nothing_values_error_for_bare_primitives(self):
        """Strings that are not valid JSON raise for typed schemas."""
        with pytest.raises(ValueError):
            _recursive_cast("", int)
        with pytest.raises(ValueError):
            _recursive_cast(0.0, int)
        with pytest.raises(ValueError):
            _recursive_cast("not-a-list", list[int])
        with pytest.raises(ValueError):
            _recursive_cast("not-a-dict", dict[str, int])

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

    def test_fixed_tuple_description(self):
        result = get_schema_description(tuple[str, int, float, bool])
        assert result[0] == "tuple[str, int, float, bool]"

    def test_variable_tuple_description(self):
        result = get_schema_description(tuple[str, ...])
        assert result[0] == "tuple[str, ...]"

    def test_nested_tuple_in_dataclass_description(self):
        @dataclass
        class Row:
            coords: tuple[float, float]

        result = get_schema_description(Row)
        assert result is not None
        schema_str, _ = result
        assert "tuple[float, float]" in schema_str


class TestTupleCast:
    """Tests for _recursive_cast with tuple schemas."""

    def test_fixed_tuple_scalars(self):
        result = _recursive_cast(["a", 42, 3.14, True], tuple[str, int, float, bool])
        assert result == ("a", 42, 3.14, True)
        assert isinstance(result, tuple)

    def test_variable_tuple(self):
        result = _recursive_cast([1, 2, 3, 4], tuple[int, ...])
        assert result == (1, 2, 3, 4)
        assert isinstance(result, tuple)

    def test_variable_tuple_empty(self):
        result = _recursive_cast([], tuple[str, ...])
        assert result == ()

    def test_nested_tuple_in_dataclass(self):
        @dataclass
        class Point:
            coords: tuple[float, float]

        result = _recursive_cast({"coords": [1.5, 2.5]}, Point)
        assert isinstance(result, Point)
        assert result.coords == (1.5, 2.5)
        assert isinstance(result.coords, tuple)

    def test_fixed_tuple_wrong_length_raises(self):
        with pytest.raises(ValueError, match="expected tuple of length 2"):
            _recursive_cast([1, 2, 3], tuple[int, int])

    def test_fixed_tuple_not_list_raises(self):
        with pytest.raises(ValueError, match="expected tuple"):
            _recursive_cast("not a list", tuple[str, int])

    def test_variable_tuple_wrong_element_type(self):
        with pytest.raises(ValueError):
            _recursive_cast([1, "two", 3], tuple[int, ...])

    def test_tuple_variable_parses_json_string(self):
        """JSON-string is JSON-parsed then recursed — no union needed."""
        result = _recursive_cast("[1, 2, 3]", tuple[int, ...])
        assert result == (1, 2, 3)
        assert isinstance(result, tuple)

    def test_tuple_fixed_parses_json_string(self):
        """Fixed-length tuple also handles JSON-string input directly."""
        result = _recursive_cast('[1, 2]', tuple[int, int])
        assert result == (1, 2)

    def test_tuple_bad_json_string_raises(self):
        """Invalid JSON string raises ValueError, not silently passes."""
        with pytest.raises(ValueError):
            _recursive_cast("not-valid-json", tuple[int, ...])

    def test_tuple_comma_sep_parses(self):
        """Comma-separated string fallback: wraps in brackets and JSON-parses."""
        result = _recursive_cast("1, 2, 3", tuple[int, int, int])
        assert result == (1, 2, 3)

    def test_tuple_variable_comma_sep_parses(self):
        """Variable-length tuple with comma-separated string."""
        result = _recursive_cast("1, 2, 3, 4", tuple[int, ...])
        assert result == (1, 2, 3, 4)

    def test_tuple_comma_sep_bad_raises(self):
        """Comma-separated string that can't be coerced raises ValueError."""
        with pytest.raises(ValueError):
            _recursive_cast("a, b, c", tuple[int, int])

    def test_tuple_parens_string_parses(self):
        """Parentheses string: replaces () with [] then JSON-parses."""
        result = _recursive_cast("(1, 2, 3)", tuple[int, int, int])
        assert result == (1, 2, 3)

    def test_tuple_variable_parens_string_parses(self):
        """Variable-length tuple with parentheses string."""
        result = _recursive_cast("(1, 2, 3, 4)", tuple[int, ...])
        assert result == (1, 2, 3, 4)

    def test_tuple_parens_string_bad_raises(self):
        """Parentheses string that can't be coerced raises ValueError."""
        with pytest.raises(ValueError):
            _recursive_cast("(a, b, c)", tuple[int, int])

    def test_parse_data_fixed_tuple(self):
        result = parse_data('["a", 42, 3.14]', tuple[str, int, float])
        assert result == ("a", 42, 3.14)
        assert isinstance(result, tuple)

    def test_parse_data_variable_tuple(self):
        result = parse_data('[1, 2, 3]', tuple[int, ...])
        assert result == (1, 2, 3)

    def test_parse_data_nested_tuple_in_dataclass(self):
        @dataclass
        class Pair:
            values: tuple[int, int]

        result = parse_data('{"values": [10, 20]}', Pair)
        assert result.values == (10, 20)
        assert isinstance(result.values, tuple)

    def test_direct_tuple_input_fixed_length(self):
        """Test that direct tuple input (not from JSON) works for fixed-length tuples."""
        result = _recursive_cast((10, 20), tuple[int, int])
        assert result == (10, 20)
        assert isinstance(result, tuple)

    def test_direct_tuple_input_variable_length(self):
        """Test that direct tuple input (not from JSON) works for variable-length tuples."""
        result = _recursive_cast((1, 2, 3, 4), tuple[int, ...])
        assert result == (1, 2, 3, 4)
        assert isinstance(result, tuple)

    def test_direct_tuple_input_nested_in_dataclass(self):
        """Test that direct tuple input works for nested tuples in dataclasses."""
        @dataclass
        class Point:
            coords: tuple[float, float]

        result = _recursive_cast({"coords": (1.5, 2.5)}, Point)
        assert isinstance(result, Point)
        assert result.coords == (1.5, 2.5)
        assert isinstance(result.coords, tuple)

    def test_direct_tuple_input_wrong_length_raises(self):
        """Test that wrong length direct tuple input raises appropriate error."""
        with pytest.raises(ValueError, match="expected tuple of length 2"):
            _recursive_cast((1, 2, 3), tuple[int, int])

    def test_direct_tuple_input_wrong_element_type_raises(self):
        """Test that wrong element type in direct tuple input raises appropriate error."""
        with pytest.raises(ValueError):
            _recursive_cast((1, "two", 3), tuple[int, ...])


# ── Union scalar coercion tests ───────────────────────────────────────

@dataclass
class NestedInner:
    value: int


@dataclass
class NestedOuter:
    items: list[NestedInner]


class TestUnionWithScalarCoercion:
    """Regression tests: union types with None coerce strings to the non-None
    type, but only when no exact type match is available. Exact match is
    preferred — if the data's type already matches a union member, it is
    returned as-is without coercion."""

    def test_int_none_coerces_string_to_int(self):
        """LLM passes "42" but schema is int | None → should become 42."""
        assert _recursive_cast("42", int | None) == 42
        assert isinstance(_recursive_cast("42", int | None), int)

    def test_int_none_coerces_float_string_to_int(self):
        """LLM passes "3.14" but int | None schema → should raise (not coerce)."""
        with pytest.raises(ValueError, match="does not match any type in union"):
            _recursive_cast("3.14", int | None)

    def test_int_none_keeps_none(self):
        """None stays None regardless of schema."""
        assert _recursive_cast(None, int | None) is None

    def test_int_none_keeps_int(self):
        """int passes through unchanged."""
        assert _recursive_cast(42, int | None) == 42

    def test_float_none_coerces_numeric_string(self):
        """String numeric values are coerced to float in float | None."""
        result = _recursive_cast("3.14", float | None)
        assert result == 3.14
        assert isinstance(result, float)

    def test_str_none_rejects_incompatible_type(self):
        """For str | None, int is not compatible with str and raises."""
        with pytest.raises(ValueError, match="does not match any type in union"):
            _recursive_cast(42, str | None)

    def test_bool_none_coerces_string(self):
        """String 'true'/'false' coerced to bool in bool | None."""
        assert _recursive_cast("true", bool | None) is True
        assert _recursive_cast("false", bool | None) is False

    def test_int_none_rejects_non_numeric_string(self):
        """Non-numeric string in int | None raises."""
        with pytest.raises(ValueError, match="does not match any type in union"):
            _recursive_cast("hello", int | None)

    def test_parse_data_union_scalar_coerces(self):
        """parse_data entry point also handles this (calls _recursive_cast)."""
        assert parse_data("42", int | None) == 42

    def test_union_multiple_scalars_preserves_exact_type(self):
        """str | int | None: exact type match is checked before coercion.
        An int stays int (not coerced to str). A string stays string."""
        # String data: exact match on str — no coercion
        result = _recursive_cast("hello", str | int | None)
        assert result == "hello"
        assert isinstance(result, str)
        # String that looks numeric: exact match on str still wins — stays string
        result = _recursive_cast("42", str | int | None)
        assert result == "42"
        assert isinstance(result, str)
        # Actual int: exact match on int — not coerced to str
        result = _recursive_cast(42, str | int | None)
        assert result == 42
        assert isinstance(result, int)
        # None stays None
        assert _recursive_cast(None, str | int | None) is None

    def test_union_exact_match_int_first(self):
        """int | str: actual int data is tested first (type(data) match)."""
        result = _recursive_cast(42, int | str)
        assert result == 42
        assert isinstance(result, int)

    def test_union_exact_match_str_first(self):
        """int | str: actual str data is tested first (type(data) match)."""
        result = _recursive_cast("hello", int | str)
        assert result == "hello"
        assert isinstance(result, str)

    def test_union_type_in_args_reordering(self):
        """type(data) is in args → tested first, before other members."""
        result = _recursive_cast([1, 2], list[int] | tuple[int, ...])
        assert result == [1, 2]
        assert isinstance(result, list)
        result = _recursive_cast((1, 2), list[int] | tuple[int, ...])
        assert result == (1, 2)
        assert isinstance(result, tuple)

    def test_tuple_union_parses_json_string(self):
        """LLM sends '[1400, 1000]' (string) for tuple[int, ...] | None."""
        result = _recursive_cast("[1400, 1000]", tuple[int, ...] | None)
        assert result == (1400, 1000)
        assert isinstance(result, tuple)

    def test_tuple_union_parses_nested_json_string(self):
        """Nested tuples as JSON strings work."""
        result = _recursive_cast("[[1, 2], [3, 4]]", tuple[tuple[int, ...], ...] | None)
        assert result == ((1, 2), (3, 4))

    def test_tuple_union_with_python_list(self):
        """LLM passes actual Python list to tuple[int, ...] | None — no JSON parse."""
        result = _recursive_cast([1516, 1012, 2516, 2012], tuple[int, ...] | None)
        assert result == (1516, 1012, 2516, 2012)
        assert isinstance(result, tuple)

    def test_tuple_union_with_python_tuple(self):
        """LLM passes actual Python tuple to tuple[int, ...] | None."""
        result = _recursive_cast((42, 99), tuple[int, int] | None)
        assert result == (42, 99)
        assert isinstance(result, tuple)

    def test_list_union_parses_json_string(self):
        """list as JSON string is parsed and cast."""
        result = _recursive_cast('[1, 2, 3]', list[int] | None)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_dataclass_union_parses_json_string(self):
        """Dataclass inside union receives JSON string, not dict."""
        from dataclasses import dataclass

        @dataclass
        class Point:
            x: int
            y: int

        result = _recursive_cast('{"x": 10, "y": 20}', Point | None)
        assert isinstance(result, Point)
        assert result.x == 10
        assert result.y == 20

    def test_nested_dataclass_union_parses_json_string(self):
        """Nested dataclass as JSON string is parsed."""
        result = _recursive_cast(
            '{"items": [{"value": 1}, {"value": 2}]}',
            NestedOuter | None,
        )
        assert isinstance(result, NestedOuter)
        assert len(result.items) == 2
        assert result.items[0].value == 1

    def test_dataclass_union_parses_comma_sep(self):
        """Dataclass union with comma-separated string input (4-field dataclass)."""
        from dataclasses import dataclass

        @dataclass
        class Rect:
            x: int
            y: int
            w: int
            h: int

        result = _recursive_cast("0,0,100,100", Rect | None)
        assert isinstance(result, Rect)
        assert result.x == 0
        assert result.y == 0
        assert result.w == 100
        assert result.h == 100
