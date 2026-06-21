"""Unit tests for the persona ToolManager class."""

import re
import pytest

from peteos.persona.toolmanager import Tool, ToolManager


class TestToolManagerRegister:
    """Test ToolManager.register_tool."""

    def test_register_by_instance(self):
        def fn() -> None:
            """Doc."""
            pass
        manager = ToolManager()
        manager.register_tool(Tool(name="inc", description="increment", func=fn))
        assert manager.get_tool("inc") is not None

    def test_register_by_callable(self):
        def greet(name: str) -> str:
            """Greet a person."""
            return f"Hello, {name}"
        manager = ToolManager()
        manager.register_tool(func=greet)
        tool = manager.get_tool("greet")
        assert tool is not None
        assert tool.description == "Greet a person."

    def test_register_by_callable_with_explicit_name_description(self):
        def fn(x: int) -> int:
            return x
        manager = ToolManager()
        manager.register_tool(func=fn, name="custom_name", description="Custom desc")
        tool = manager.get_tool("custom_name")
        assert tool is not None
        assert tool.description == "Custom desc"

    def test_register_by_callable_with_custom_parameters(self):
        def fn(x: int) -> int:
            return x
        manager = ToolManager()
        manager.register_tool(func=fn, name="custom", description="c", parameters={"x": {"type": "int"}})
        tool = manager.get_tool("custom")
        assert tool.parameters == {"x": {"type": "int"}}

    def test_register_overwrites_existing_tool(self):
        def fn1() -> None:
            """First."""
            pass
        def fn2() -> None:
            """Second."""
            pass
        manager = ToolManager()
        manager.register_tool(Tool(name="dup", description="first", func=fn1))
        manager.register_tool(Tool(name="dup", description="second", func=fn2))
        assert manager.get_tool("dup").description == "second"

    def test_register_missing_args_raises_value_error(self):
        manager = ToolManager()
        with pytest.raises(ValueError, match="Either 'tool' or 'func'"):
            manager.register_tool()

    def test_register_multiple_tools(self):
        def fn1() -> None:
            """First."""
            pass
        def fn2() -> None:
            """Second."""
            pass
        manager = ToolManager()
        manager.register_tool(Tool(name="first", description="first", func=fn1))
        manager.register_tool(Tool(name="second", description="second", func=fn2))
        assert manager.get_tool("first") is not None
        assert manager.get_tool("second") is not None


class TestToolManagerGet:
    """Test ToolManager retrieval methods."""

    def test_get_nonexistent_returns_none(self):
        assert ToolManager().get_tool("missing") is None

    def test_get_tool_list_no_filter(self):
        manager = ToolManager()
        manager.register_tool(Tool(name="a", description="a", func=lambda: None))
        manager.register_tool(Tool(name="b", description="b", func=lambda: None))
        tools = manager.get_tool_list()
        assert len(tools) == 2

    def test_get_tool_list_filter_by_exact_name(self):
        manager = ToolManager()
        manager.register_tool(Tool(name="alpha", description="a", func=lambda: None))
        manager.register_tool(Tool(name="beta", description="b", func=lambda: None))
        tools = manager.get_tool_list(filter_patterns=["beta"])
        assert len(tools) == 1
        assert tools[0].name == "beta"

    def test_get_tool_list_filter_by_regex(self):
        manager = ToolManager()
        manager.register_tool(Tool(name="router_in", description="a", func=lambda: None))
        manager.register_tool(Tool(name="router_out", description="b", func=lambda: None))
        manager.register_tool(Tool(name="other", description="c", func=lambda: None))
        tools = manager.get_tool_list(filter_patterns=["router_.*"])
        assert len(tools) == 2
        assert {t.name for t in tools} == {"router_in", "router_out"}

    def test_get_tool_list_filter_no_match(self):
        manager = ToolManager()
        manager.register_tool(Tool(name="alpha", description="a", func=lambda: None))
        tools = manager.get_tool_list(filter_patterns=[".*nonexistent.*"])
        assert tools == []

    def test_get_tool_list_empty_filter_returns_all(self):
        manager = ToolManager()
        manager.register_tool(Tool(name="a", description="a", func=lambda: None))
        tools = manager.get_tool_list(filter_patterns=[])
        assert len(tools) == 1

    def test_get_tool_list_none_filter_returns_all(self):
        manager = ToolManager()
        manager.register_tool(Tool(name="a", description="a", func=lambda: None))
        tools = manager.get_tool_list(filter_patterns=None)
        assert len(tools) == 1
