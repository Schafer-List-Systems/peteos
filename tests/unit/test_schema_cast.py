"""Tests for _recursive_cast with 10 progressively complex schemas."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from peteos.oap._schema import _recursive_cast, parse_data, validate_data


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
        with pytest.raises(ValueError, match="invalid JSON"):
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
