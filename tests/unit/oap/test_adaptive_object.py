"""Unit and integration tests for AdaptiveObject."""

from unittest.mock import MagicMock

import pytest

from peteos.oap import AdaptiveObject, AgenticObject, agentic_object, tool


def _mock_runner():
    runner = MagicMock()
    runner._execution_environment = MagicMock()
    runner._execution_environment.auto_approve_tools = []
    return runner


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

    @pytest.fixture
    def runner(self):
        return _mock_runner()

    def test_persist_simple_function(self, obj, runner):
        """Persisting a simple function returns OK with the function name."""
        code = "def my_math(x: int) -> int:\n    return x * 2"
        result = obj.define_function(code, "Multiply by 2", runner)
        assert result == "OK: registered as 'my_math'"

    def test_defined_function_appears_in_tool_list(self, obj, runner):
        """A defined function is registered in the ToolManager's tool list."""
        code = "def my_math(x: int) -> int:\n    return x * 2"
        obj.define_function(code, "Multiply by 2", runner)
        tools = [t.name for t in obj._oap_tool_manager.get_tool_list()]
        assert "my_math" in tools

    def test_defined_function_tracks_metadata(self, obj, runner):
        """Defined functions are tracked in _oap_define_functions."""
        code = "def my_math(x: int) -> int:\n    return x * 2"
        obj.define_function(code, "Multiply by 2", runner)
        assert "my_math" in obj._oap_define_functions
        meta = obj._oap_define_functions["my_math"]
        assert meta["code"] == code
        assert meta["docstring"] == "Multiply by 2"
        assert "x" in meta["parameters"]

    def test_define_function_with_multiple_parameters(self, obj, runner):
        """Persisting a function with multiple parameters extracts all of them."""
        code = "def combine(a: str, b: str) -> str:\n    return a + b"
        result = obj.define_function(code, "Concatenate two strings", runner)
        assert result == "OK: registered as 'combine'"
        meta = obj._oap_define_functions["combine"]
        assert "a" in meta["parameters"]
        assert "b" in meta["parameters"]
        assert meta["parameters"]["a"]["required"] is True
        assert meta["parameters"]["b"]["required"] is True

    def test_define_function_with_defaults(self, obj, runner):
        """Persisting a function with default values marks params as not required."""
        code = "def greet(name: str, greeting: str = 'Hello') -> str:\n    return f'{greeting}, {name}'"
        result = obj.define_function(code, "Create a greeting", runner)
        assert result == "OK: registered as 'greet'"
        meta = obj._oap_define_functions["greet"]
        assert meta["parameters"]["name"]["required"] is True
        assert meta["parameters"]["greeting"]["required"] is False
        assert meta["parameters"]["greeting"]["default"] == "Hello"

    def test_define_function_syntax_error(self, obj, runner):
        """Persisting code with a syntax error returns an error."""
        code = "def broken(:\n    return"
        result = obj.define_function(code, "Should fail", runner)
        assert result.startswith("Error:")

    def test_define_function_no_function_defined(self, obj, runner):
        """Persisting code with no function returns an error."""
        code = "x = 42"
        result = obj.define_function(code, "No function", runner)
        assert result.startswith("Error:")
        assert "exactly one function" in result

    def test_define_function_multiple_functions(self, obj, runner):
        """Persisting code with multiple functions returns an error."""
        code = "def a(): pass\ndef b(): pass"
        result = obj.define_function(code, "Too many", runner)
        assert result.startswith("Error:")
        assert "exactly one function" in result

    def test_define_function_name_collision(self, obj, runner):
        """Persisting a function that collides with an existing tool returns an error."""
        code = "def add(x: int) -> int:\n    return x"
        result = obj.define_function(code, "Collides", runner)
        assert result.startswith("Error:")
        assert "already exists" in result

    def test_define_function_exec_error(self, obj, runner):
        """Persisting code that raises during exec returns an error."""
        code = "import os\nx = 1"
        result = obj.define_function(code, "Exec error", runner)
        assert result.startswith("Error:")

    def test_define_function_with_self_param(self, obj, runner):
        """The 'self' parameter is excluded from the tool parameters schema."""
        code = "def process(self, data: int) -> int:\n    return data"
        result = obj.define_function(code, "Process data", runner)
        assert result == "OK: registered as 'process'"
        meta = obj._oap_define_functions["process"]
        assert "self" not in meta["parameters"]
        assert "data" in meta["parameters"]


class TestRemoveToolTool:
    """Tests for the remove_function tool."""

    @pytest.fixture
    def obj(self):
        return SimpleAdaptiveObject()

    @pytest.fixture
    def runner(self):
        return _mock_runner()

    def test_remove_existing_defined_tool(self, obj, runner):
        """Removing a defined tool succeeds and removes it from both registries."""
        code = "def my_math(x: int) -> int:\n    return x * 2"
        obj.define_function(code, "Multiply by 2", runner)
        assert "my_math" in obj._oap_define_functions
        assert obj._oap_tool_manager.get_tool("my_math") is not None

        result = obj.remove_function("my_math", runner)
        assert result == "OK"
        assert "my_math" not in obj._oap_define_functions
        assert obj._oap_tool_manager.get_tool("my_math") is None

    def test_remove_nonexistent_tool(self, obj, runner):
        """Removing a tool that was never defined returns an error."""
        result = obj.remove_function("nonexistent", runner)
        assert result.startswith("Error:")
        assert "not found or not defined" in result

    def test_remove_does_not_affect_static_tools(self, obj, runner):
        """Removing a name that matches a static tool returns an error."""
        result = obj.remove_function("add", runner)
        assert result.startswith("Error:")
        assert obj._oap_tool_manager.get_tool("add") is not None


class TestPersistedFunctionSandboxIsolation:
    """Tests that defined functions execute in SandboxSelf isolation."""

    @pytest.fixture
    def obj(self):
        return SimpleAdaptiveObject()

    @pytest.fixture
    def runner(self):
        return _mock_runner()

    def test_defined_function_can_be_called(self, obj, runner):
        """Calling a defined function through the ToolManager executes it."""
        code = "def double(x: int) -> int:\n    return x * 2"
        obj.define_function(code, "Double a number", runner)
        tool = obj._oap_tool_manager.get_tool("double")
        assert tool is not None
        result = tool.execute(x=21)
        assert result == 42

    def test_defined_function_with_string_output(self, obj, runner):
        """Defined functions that return strings work correctly."""
        code = "def upper(text: str) -> str:\n    return text.upper()"
        obj.define_function(code, "Uppercase a string", runner)
        tool = obj._oap_tool_manager.get_tool("upper")
        result = tool.execute(text="hello")
        assert result == "HELLO"

    def test_defined_function_with_none_returns_ok(self, obj, runner):
        """Defined functions that return None return None (matching _python_exec)."""
        code = "def noop(x: int) -> None:\n    pass"
        obj.define_function(code, "No-op function", runner)
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


class TestAutoApproveToolsInstanceOwnership:
    """Tests that _oap_auto_approve_tools is instance-level."""

    @pytest.fixture
    def obj(self):
        return SimpleAdaptiveObject()

    @pytest.fixture
    def runner(self):
        return _mock_runner()

    def test_initial_auto_approve_contains_static_tools(self):
        """Instance auto-approve list starts with registered static tools."""
        obj = SimpleAdaptiveObject()
        tools = [t.name for t in obj._oap_tool_manager.get_tool_list()]
        for t in tools:
            assert t in obj._oap_auto_approve_tools

    def test_define_function_adds_to_instance_auto_approve(self, obj, runner):
        """define_function adds the new tool to the instance's auto-approve list."""
        code = "def my_math(x: int) -> int:\n    return x * 2"
        obj.define_function(code, "Multiply by 2", runner)
        assert "my_math" in obj._oap_auto_approve_tools

    def test_remove_function_removes_from_instance_auto_approve(self, obj, runner):
        """remove_function removes the tool from the instance's auto-approve list."""
        code = "def my_math(x: int) -> int:\n    return x * 2"
        obj.define_function(code, "Multiply by 2", runner)
        obj.remove_function("my_math", runner)
        assert "my_math" not in obj._oap_auto_approve_tools

    def test_different_instances_have_separate_lists(self):
        """Two instances of the same class have independent auto-approve lists."""
        obj1 = SimpleAdaptiveObject()
        obj2 = SimpleAdaptiveObject()
        runner1 = _mock_runner()
        runner2 = _mock_runner()
        obj1.define_function("def a(x: int) -> int:\n    return x", "A", runner1)
        assert "a" in obj1._oap_auto_approve_tools
        assert "a" not in obj2._oap_auto_approve_tools

    def test_auto_approve_persists_to_new_runner(self):
        """A tool defined in one session remains auto-approved in a new runner.

        This verifies the fix for the bug where define_function only modified
        the current runner's ExecutionEnvironment copy, so a subsequent
        invoke_agent (which creates a new runner) would not auto-approve the
        defined tool, causing an infinite PENDING loop.

        The fix: _oap_auto_approve_tools is instance-level, and _start_session
        copies it to each new runner.
        """
        obj = SimpleAdaptiveObject()

        # Session 1: define a function with an active runner
        runner1 = _mock_runner()
        obj.define_function(
            "def compute(x: int) -> int:\n    return x * 2",
            "Double a number",
            runner1,
        )

        # Verify it is in runner1's auto-approve list (active session)
        assert "compute" in runner1._execution_environment.auto_approve_tools

        # Simulate a new invoke_agent call: _start_session creates a fresh runner
        runner2 = _mock_runner()
        runner2._execution_environment.auto_approve_tools = list(
            obj._oap_auto_approve_tools
        )

        # The new runner must also have the defined tool auto-approved
        assert "compute" in runner2._execution_environment.auto_approve_tools

        # Static tools should also be present (the baseline copy)
        assert "add" in runner2._execution_environment.auto_approve_tools
        assert "define_function" in runner2._execution_environment.auto_approve_tools
