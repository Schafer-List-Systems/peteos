"""Unit and integration tests for AdaptiveObject."""

import pytest

from peteos.oap import AdaptiveObject, AgenticObject, agentic_object, tool


@agentic_object()
class SimpleStaticTool(AgenticObject):
    """AgenticObject subclass with @tool-decorated static methods (no function definitions)."""

    @tool(name="add", description="Add two numbers")
    def add(self, a: int, b: int) -> int:
        return a + b


class SimpleAdaptiveObject(AdaptiveObject):
    """AgenticObject subclass with define_functions enabled via AdaptiveObject."""

    @tool(name="add", description="Add two numbers")
    def add(self, a: int, b: int) -> int:
        return a + b


class TestAdaptiveObjectBasic:
    """Tests for basic AdaptiveObject initialization."""

    def test_regular_object_has_no_define_functions(self):
        """AgenticObject without define_functions does not get define_function/remove_function."""
        obj = SimpleStaticTool()
        tools = [t.name for t in obj._oap_tool_manager.get_tool_list()]
        assert "define_function" not in tools
        assert "remove_function" not in tools

    def test_adaptive_object_has_defined_tools(self):
        """AdaptiveObject with define_functions gets define_function and remove_function."""
        obj = SimpleAdaptiveObject()
        tools = [t.name for t in obj._oap_tool_manager.get_tool_list()]
        assert "define_function" in tools
        assert "remove_function" in tools


class TestDefineFunctionTool:
    """Tests for the define_function tool."""

    @pytest.fixture
    def obj(self):
        return SimpleAdaptiveObject()

    def test_persist_simple_function(self, obj):
        """Persisting a simple function returns OK with the function name."""
        code = "def my_math(x: int) -> int:\n    return x * 2"
        result = obj.define_function(code, "Multiply by 2")
        assert result == "OK: registered as 'my_math'"

    def test_defined_function_appears_in_tool_list(self, obj):
        """A defined function is registered in the ToolManager's tool list."""
        code = "def my_math(x: int) -> int:\n    return x * 2"
        obj.define_function(code, "Multiply by 2")
        tools = [t.name for t in obj._oap_tool_manager.get_tool_list()]
        assert "my_math" in tools

    def test_defined_function_tracks_metadata(self, obj):
        """Defined functions are tracked in _oap_define_functions."""
        code = "def my_math(x: int) -> int:\n    return x * 2"
        obj.define_function(code, "Multiply by 2")
        assert "my_math" in obj._oap_define_functions
        meta = obj._oap_define_functions["my_math"]
        assert meta["code"] == code
        assert meta["docstring"] == "Multiply by 2"
        assert "x" in meta["parameters"]

    def test_define_function_with_multiple_parameters(self, obj):
        """Persisting a function with multiple parameters extracts all of them."""
        code = "def combine(a: str, b: str) -> str:\n    return a + b"
        result = obj.define_function(code, "Concatenate two strings")
        assert result == "OK: registered as 'combine'"
        meta = obj._oap_define_functions["combine"]
        assert "a" in meta["parameters"]
        assert "b" in meta["parameters"]
        assert meta["parameters"]["a"]["required"] is True
        assert meta["parameters"]["b"]["required"] is True

    def test_define_function_with_defaults(self, obj):
        """Persisting a function with default values marks params as not required."""
        code = "def greet(name: str, greeting: str = 'Hello') -> str:\n    return f'{greeting}, {name}'"
        result = obj.define_function(code, "Create a greeting")
        assert result == "OK: registered as 'greet'"
        meta = obj._oap_define_functions["greet"]
        assert meta["parameters"]["name"]["required"] is True
        assert meta["parameters"]["greeting"]["required"] is False
        assert meta["parameters"]["greeting"]["default"] == "Hello"

    def test_define_function_syntax_error(self, obj):
        """Persisting code with a syntax error returns an error."""
        code = "def broken(:\n    return"
        result = obj.define_function(code, "Should fail")
        assert result.startswith("Error:")

    def test_define_function_no_function_defined(self, obj):
        """Persisting code with no function returns an error."""
        code = "x = 42"
        result = obj.define_function(code, "No function")
        assert result.startswith("Error:")
        assert "exactly one function" in result

    def test_define_function_multiple_functions(self, obj):
        """Persisting code with multiple functions returns an error."""
        code = "def a(): pass\ndef b(): pass"
        result = obj.define_function(code, "Too many")
        assert result.startswith("Error:")
        assert "exactly one function" in result

    def test_define_function_name_collision(self, obj):
        """Persisting a function that collides with an existing tool returns an error."""
        code = "def add(x: int) -> int:\n    return x"
        result = obj.define_function(code, "Collides")
        assert result.startswith("Error:")
        assert "already exists" in result

    def test_define_function_exec_error(self, obj):
        """Persisting code that raises during exec returns an error."""
        code = "import os\nx = 1"
        result = obj.define_function(code, "Exec error")
        assert result.startswith("Error:")

    def test_define_function_with_self_param(self, obj):
        """The 'self' parameter is excluded from the tool parameters schema."""
        code = "def process(self, data: int) -> int:\n    return data"
        result = obj.define_function(code, "Process data")
        assert result == "OK: registered as 'process'"
        meta = obj._oap_define_functions["process"]
        assert "self" not in meta["parameters"]
        assert "data" in meta["parameters"]


class TestRemoveToolTool:
    """Tests for the remove_function tool."""

    @pytest.fixture
    def obj(self):
        return SimpleAdaptiveObject()

    def test_remove_existing_defined_tool(self, obj):
        """Removing a defined tool succeeds and removes it from both registries."""
        code = "def my_math(x: int) -> int:\n    return x * 2"
        obj.define_function(code, "Multiply by 2")
        assert "my_math" in obj._oap_define_functions
        assert obj._oap_tool_manager.get_tool("my_math") is not None

        result = obj.remove_function("my_math")
        assert result == "OK"
        assert "my_math" not in obj._oap_define_functions
        assert obj._oap_tool_manager.get_tool("my_math") is None

    def test_remove_nonexistent_tool(self, obj):
        """Removing a tool that was never defined returns an error."""
        result = obj.remove_function("nonexistent")
        assert result.startswith("Error:")
        assert "not found or not defined" in result

    def test_remove_does_not_affect_static_tools(self, obj):
        """Removing a name that matches a static tool returns an error."""
        result = obj.remove_function("add")
        assert result.startswith("Error:")
        assert obj._oap_tool_manager.get_tool("add") is not None


class TestPersistedFunctionSandboxIsolation:
    """Tests that defined functions execute in SandboxSelf isolation."""

    @pytest.fixture
    def obj(self):
        return SimpleAdaptiveObject()

    def test_defined_function_can_be_called(self, obj):
        """Calling a defined function through the ToolManager executes it."""
        code = "def double(x: int) -> int:\n    return x * 2"
        obj.define_function(code, "Double a number")
        tool = obj._oap_tool_manager.get_tool("double")
        assert tool is not None
        result = tool.execute(x=21)
        assert result == "42"

    def test_defined_function_with_string_output(self, obj):
        """Defined functions that return strings work correctly."""
        code = "def upper(text: str) -> str:\n    return text.upper()"
        obj.define_function(code, "Uppercase a string")
        tool = obj._oap_tool_manager.get_tool("upper")
        result = tool.execute(text="hello")
        assert result == "HELLO"

    def test_defined_function_with_none_returns_ok(self, obj):
        """Defined functions that return None return None (matching _python_exec)."""
        code = "def noop(x: int) -> None:\n    pass"
        obj.define_function(code, "No-op function")
        tool = obj._oap_tool_manager.get_tool("noop")
        result = tool.execute(x=42)
        assert result is None


class TestDefineFunctionConfigGating:
    """Tests that define_functions config gates tool registration."""

    def test_define_functions_false_by_default(self):
        """@agentic_object without define_functions defaults to False."""
        @agentic_object()
        class PlainAgenticObject(AgenticObject):
            pass

        obj = PlainAgenticObject()
        tools = [t.name for t in obj._oap_tool_manager.get_tool_list()]
        assert "define_function" not in tools
        assert "remove_function" not in tools

    def test_define_functions_explicit_false(self):
        """@agentic_object(define_functions=False) does not enable persistence."""
        @agentic_object(define_functions=False)
        class ExplicitAgenticObject(AgenticObject):
            pass

        obj = ExplicitAgenticObject()
        tools = [t.name for t in obj._oap_tool_manager.get_tool_list()]
        assert "define_function" not in tools


class TestCollectOapConfigDefineFunctions:
    """Tests that _collect_oap_config merges define_functions across MRO."""

    def test_subclass_inherits_define_functions(self):
        """A subclass of AdaptiveObject inherits define_functions."""
        @agentic_object(define_functions=True)
        class BaseAdaptive(AgenticObject):
            pass

        class DerivedAdaptive(BaseAdaptive):
            pass

        from peteos.oap.agentic_object import _collect_oap_config
        cfg = _collect_oap_config(DerivedAdaptive)
        assert cfg["define_functions"] is True
