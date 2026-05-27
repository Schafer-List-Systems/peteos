"""Tests for OAP engine utilities."""

from peteos.oap.engine import (
    _discover_bound_tools,
    _extract_method_params,
)
from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import agentic_object, tool


class TestDiscoverBoundTools:
    def test_discovers_decorated_methods(self):
        @agentic_object()
        class MyObj(AgenticObjectBase):
            @tool()
            def hello(self, name: str) -> str:
                """Say hello."""
                return f"Hello, {name}"

            def not_a_tool(self):
                pass

        tools = _discover_bound_tools(MyObj())
        assert len(tools) == 1
        assert tools[0].name == "hello"
        assert "Say hello" in tools[0].description

    def test_discovers_no_tools(self):
        @agentic_object()
        class MyObj(AgenticObjectBase):
            pass

        tools = _discover_bound_tools(MyObj())
        assert len(tools) == 0

    def test_tools_are_bound(self):
        @agentic_object()
        class MyObj(AgenticObjectBase):
            def __init__(self):
                self.value = "test"

            @tool()
            def get_value(self) -> str:
                """Get the value."""
                return self.value

        tools = _discover_bound_tools(MyObj())
        assert len(tools) == 1
        # Tool function should be bound to the instance
        result = tools[0].func()
        assert result == "test"


class TestExtractMethodParams:
    def test_extracts_string_param(self):
        def greet(name: str) -> str:
            return f"Hello, {name}"

        params = _extract_method_params(greet)
        assert params["name"]["type"] == "str"
        assert params["name"]["required"] is True

    def test_extracts_optional_param(self):
        def greet(name: str = "World") -> str:
            return f"Hello, {name}"

        params = _extract_method_params(greet)
        assert params["name"]["type"] == "str"
        assert params["name"]["required"] is False
        assert params["name"]["default"] == "World"

    def test_extracts_multiple_params(self):
        def add(a: int, b: float) -> float:
            return a + b

        params = _extract_method_params(add)
        assert params["a"]["type"] == "int"
        assert params["b"]["type"] == "float"

    def test_skips_self(self):
        class MyObj(AgenticObjectBase):
            @tool()
            def method(self, x: int) -> int:
                return x

        params = _extract_method_params(MyObj().method)
        assert "self" not in params
        assert "x" in params
