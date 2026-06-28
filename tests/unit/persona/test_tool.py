"""Unit tests for the persona Tool class."""

import pytest

from peteos.persona.toolmanager import Tool


class TestToolFromCallable:
    """Test Tool.from_callable."""

    def test_name_extracted(self):
        def my_tool(x: str) -> str:
            """Do stuff."""
            return x
        tool = Tool.from_callable(my_tool)
        assert tool.name == "my_tool"

    def test_description_from_docstring(self):
        def calc(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b
        tool = Tool.from_callable(calc)
        assert tool.description == "Add two numbers together."

    def test_description_defaults_to_empty_string(self):
        def naked(x: int) -> int:
            return x
        tool = Tool.from_callable(naked)
        assert tool.description == ""

    def test_parameters_extracted(self):
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        tool = Tool.from_callable(add)
        assert "a" in tool.parameters
        assert "b" in tool.parameters

    def test_runner_excluded_from_parameters(self):
        def with_runner(runner=None) -> str:
            """A tool with runner param."""
            return ""
        tool = Tool.from_callable(with_runner)
        assert "runner" not in tool.parameters

    def test_signature_preserved_by_wraps(self):
        def original(x: int, y: str = "default") -> None:
            """Original."""
            pass
        tool = Tool.from_callable(original)
        assert tool.parameters["x"]["required"] is True
        assert tool.parameters["y"]["required"] is False
        assert tool.parameters["y"]["default"] == "default"


class TestToolInit:
    """Test Tool.__init__."""

    def test_init_stores_name_description(self):
        def fn() -> None:
            """Doc."""
            pass
        tool = Tool(name="my_tool", description="My tool", func=fn)
        assert tool.name == "my_tool"
        assert tool.description == "My tool"

    def test_init_defaults_parameters(self):
        def fn() -> None:
            """Doc."""
            pass
        tool = Tool(name="no_params", description="no params", func=fn)
        assert tool.parameters == {}

    def test_init_accepts_custom_parameters(self):
        def fn() -> None:
            """Doc."""
            pass
        tool = Tool(name="custom", description="custom", func=fn, parameters={"x": {"type": "int", "required": True}})
        assert tool.parameters == {"x": {"type": "int", "required": True}}

    def test_func_property_returns_wrapped(self):
        def fn() -> None:
            """Doc."""
            pass
        tool = Tool(name="p", description="p", func=fn)
        assert callable(tool.func)

    def test_call_delegates_to_func(self):
        def fn(x: int) -> int:
            return x * 2
        tool = Tool(name="double", description="double", func=fn)
        assert tool(x=5) == 10

    def test_execute_delegates_to_func(self):
        def fn(x: int) -> int:
            return x * 2
        tool = Tool(name="double", description="double", func=fn)
        assert tool.execute(x=5) == 10


class TestToolWrapForRunner:
    """Test _wrap_for_runner behaviour."""

    def test_fn_accepting_runner_is_unchanged(self):
        def with_runner(x: int, runner=None) -> int:
            return x
        tool = Tool(name="test", description="t", func=with_runner)
        assert tool(x=5) == 5

    def test_fn_without_runner_does_not_error_on_runner_kwarg(self):
        def bare(x: int) -> int:
            return x + 1
        tool = Tool(name="bare", description="b", func=bare)
        assert tool(x=5, runner="some_runner") == 6


class TestToolParameterExtraction:
    """Test Tool._extract_parameters."""

    def test_no_annotation_yields_any_type(self):
        def naked(x) -> None:
            pass
        tool = Tool(name="naked", description="n", func=naked)
        assert tool.parameters["x"]["type"] == "any"

    def test_with_annotation_yields_type_name(self):
        def typed(x: str) -> None:
            pass
        tool = Tool(name="typed", description="t", func=typed)
        assert tool.parameters["x"]["type"] == "str"

    def test_required_param_has_no_default(self):
        def req(x: int, y: str) -> None:
            pass
        tool = Tool(name="req", description="r", func=req)
        assert tool.parameters["x"]["required"] is True
        assert "default" not in tool.parameters["x"]

    def test_optional_param_has_default(self):
        def opt(x: int = 42) -> None:
            pass
        tool = Tool(name="opt", description="o", func=opt)
        assert tool.parameters["x"]["required"] is False
        assert tool.parameters["x"]["default"] == 42
