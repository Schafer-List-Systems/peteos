"""Unit tests for ToolDefinitionsMessage."""

import json
import hashlib

import pytest

from peteos.conversation.message import ContentPart
from peteos.conversation.tool_definitions_message import ToolDefinitionsMessage

TOOL_LIST_HOOK_NAME = "tool_list"
TOOL_FILTER_HOOK_NAME = "tool_filter"


def _hook_id(name: str) -> str:
    """Compute the deterministic sha256 hook ID for a hook name."""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _make_tool(name: str, description: str, parameters: dict | None = None) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": parameters or {},
    }


class TestToolDefinitionsMessage:
    """Tests for ToolDefinitionsMessage base functionality."""

    def test_init_default_role(self):
        msg = ToolDefinitionsMessage()
        assert msg.raw_dict["role"] == "tool"

    def test_init_preserves_role(self):
        msg = ToolDefinitionsMessage({"role": "user"})
        assert msg.raw_dict["role"] == "user"

    def test_init_empty_dict(self):
        msg = ToolDefinitionsMessage({})
        assert msg.raw_dict["role"] == "tool"

    def test_tool_list_hook_id(self):
        msg = ToolDefinitionsMessage()
        assert msg._tool_list_hook_id == _hook_id(TOOL_LIST_HOOK_NAME)

    def test_tool_filter_hook_id(self):
        msg = ToolDefinitionsMessage()
        assert msg._tool_filter_hook_id == _hook_id(TOOL_FILTER_HOOK_NAME)


class TestToolFilter:
    """Tests for the tool filter feature."""

    def test_filter_single_pattern(self):
        msg = ToolDefinitionsMessage()
        tool_list = json.dumps([_make_tool("internal_tool", "visible")])
        tool_filter = json.dumps(["internal_.*"])
        msg.materialize(
            materialized_hooks={
                msg._tool_list_hook_id: "hash_tool_list",
                msg._tool_filter_hook_id: "hash_filter",
            },
            content_map={
                "hash_tool_list": tool_list,
                "hash_filter": tool_filter,
            },
        )
        assert len(msg.content) == 1
        assert msg.content[0].name == "internal_tool"

    def test_filter_excludes_matching_tools(self):
        msg = ToolDefinitionsMessage()
        tool_list = json.dumps([
            _make_tool("public", "ok"),
            _make_tool("internal_get", "included"),
            _make_tool("internal_set", "included"),
        ])
        tool_filter = json.dumps(["internal_.*"])
        msg.materialize(
            materialized_hooks={
                msg._tool_list_hook_id: "hash_tool_list",
                msg._tool_filter_hook_id: "hash_filter",
            },
            content_map={
                "hash_tool_list": tool_list,
                "hash_filter": tool_filter,
            },
        )
        assert len(msg.content) == 2
        assert msg.content[0].name == "internal_get"
        assert msg.content[1].name == "internal_set"

    def test_filter_fullmatch_vs_prefix(self):
        msg = ToolDefinitionsMessage()
        tool_list = json.dumps([
            _make_tool("internal_get", "exact match"),
            _make_tool("internal_get_extra", "prefix match, not included"),
        ])
        tool_filter = json.dumps(["internal_get"])
        msg.materialize(
            materialized_hooks={
                msg._tool_list_hook_id: "hash_tool_list",
                msg._tool_filter_hook_id: "hash_filter",
            },
            content_map={
                "hash_tool_list": tool_list,
                "hash_filter": tool_filter,
            },
        )
        # internal_get matches exactly via fullmatch, internal_get_extra does not
        assert len(msg.content) == 1
        assert msg.content[0].name == "internal_get"

    def test_filter_no_filter_hook(self):
        msg = ToolDefinitionsMessage()
        tool_list = json.dumps([_make_tool("tool_a", "a"), _make_tool("tool_b", "b")])
        msg.materialize(
            materialized_hooks={
                msg._tool_list_hook_id: "hash_tool_list",
            },
            content_map={
                "hash_tool_list": tool_list,
            },
        )
        assert len(msg.content) == 2

    def test_filter_empty_set(self):
        msg = ToolDefinitionsMessage()
        tool_list = json.dumps([_make_tool("tool_a", "a")])
        tool_filter = json.dumps([])
        msg.materialize(
            materialized_hooks={
                msg._tool_list_hook_id: "hash_tool_list",
                msg._tool_filter_hook_id: "hash_filter",
            },
            content_map={
                "hash_tool_list": tool_list,
                "hash_filter": tool_filter,
            },
        )
        assert len(msg.content) == 1


class TestToolListMaterialization:
    """Tests for tool list materialization."""

    def test_materialize_with_one_tool(self):
        msg = ToolDefinitionsMessage()
        tool_list = json.dumps([_make_tool("greet", "Greets someone", {"name": {"type": "string"}})])
        msg.materialize(
            materialized_hooks={msg._tool_list_hook_id: "hash_tl"},
            content_map={"hash_tl": tool_list},
        )
        assert len(msg.content) == 1
        assert msg.content[0].type == "tool"
        assert msg.content[0].name == "greet"
        assert msg.content[0].description == "Greets someone"

    def test_materialize_with_multiple_tools(self):
        msg = ToolDefinitionsMessage()
        tool_list = json.dumps([
            _make_tool("tool_a", "First tool"),
            _make_tool("tool_b", "Second tool"),
            _make_tool("tool_c", "Third tool"),
        ])
        msg.materialize(
            materialized_hooks={msg._tool_list_hook_id: "hash_tl"},
            content_map={"hash_tl": tool_list},
        )
        assert len(msg.content) == 3

    def test_materialize_empty_tool_list(self):
        msg = ToolDefinitionsMessage()
        msg.materialize(
            materialized_hooks={msg._tool_list_hook_id: "hash_tl"},
            content_map={"hash_tl": "[]"},
        )
        assert len(msg.content) == 0

    def test_materialize_invalid_json_logs_error(self):
        msg = ToolDefinitionsMessage()
        msg.materialize(
            materialized_hooks={msg._tool_list_hook_id: "hash_tl"},
            content_map={"hash_tl": "{invalid json"},
        )
        assert msg.content == []

    def test_materialize_missing_tool_list_hook(self):
        msg = ToolDefinitionsMessage()
        msg.materialize(
            materialized_hooks={},
            content_map={},
        )
        assert msg.content == []

    def test_materialize_missing_content_hash(self):
        msg = ToolDefinitionsMessage()
        msg.materialize(
            materialized_hooks={msg._tool_list_hook_id: "nonexistent_hash"},
            content_map={},
        )
        assert msg.content == []

    def test_materialize_none_params(self):
        msg = ToolDefinitionsMessage()
        tool_list = json.dumps([_make_tool("no_params", "no params spec", None)])
        msg.materialize(
            materialized_hooks={msg._tool_list_hook_id: "hash_tl"},
            content_map={"hash_tl": tool_list},
        )
        assert msg.content[0].parameters == {}

    def test_materialize_preserves_params(self):
        msg = ToolDefinitionsMessage()
        params = {"type": "object", "properties": {"x": {"type": "number"}}}
        tool_list = json.dumps([_make_tool("calc", "Calculator", params)])
        msg.materialize(
            materialized_hooks={msg._tool_list_hook_id: "hash_tl"},
            content_map={"hash_tl": tool_list},
        )
        assert msg.content[0].parameters == params


class TestMaterializeNoop:
    """Tests that materialize is a no-op when data is missing."""

    def test_none_materialized_hooks(self):
        msg = ToolDefinitionsMessage()
        msg.materialize(materialized_hooks=None, content_map={"any": "val"})
        assert msg.content == []

    def test_none_content_map(self):
        msg = ToolDefinitionsMessage()
        msg.materialize(materialized_hooks={"any": "hash"}, content_map=None)
        assert msg.content == []


class TestFormatToolList:
    """Tests for the _format_tool_list static method."""

    def test_format_single_tool(self):
        tools = [{"name": "f", "description": "d", "parameters": {}}]
        result = ToolDefinitionsMessage._format_tool_list(tools)
        assert result == [{"type": "tool", "name": "f", "description": "d", "parameters": {}}]

    def test_format_empty_list(self):
        assert ToolDefinitionsMessage._format_tool_list([]) == []

    def test_format_missing_fields(self):
        tools = [{"name": "only_name"}]
        result = ToolDefinitionsMessage._format_tool_list(tools)
        assert result == [{"type": "tool", "name": "only_name", "description": "", "parameters": {}}]

    def test_format_adds_type(self):
        tools = [{"name": "t"}]
        result = ToolDefinitionsMessage._format_tool_list(tools)
        assert result[0]["type"] == "tool"
