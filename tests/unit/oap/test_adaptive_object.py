"""Unit and integration tests for AdaptiveObject."""

from unittest.mock import MagicMock

import pytest

from peteos.oap import AdaptiveObject, AgenticObject, agentic_object, tool


def _mock_runner():
    runner = MagicMock()
    runner._execution_environment = MagicMock()
    runner._execution_environment.auto_approve_tools = []
    return runner


def _wire_runner(obj, runner):
    """Wire runner.sandbox_builder and runner.sandbox using the object's own _create_sandbox_builder."""
    config = {"invoke_sub_agents": False}
    sandbox_builder = obj._create_sandbox_builder(config, runner=runner)
    runner.sandbox_builder = sandbox_builder
    runner.sandbox = sandbox_builder.get_sandbox(freeze_namespaces=False)


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
        """Persisting code with a syntax error raises ValueError."""
        code = "def broken(:\n    return"
        with pytest.raises(ValueError):
            obj.define_function(code, "Should fail")

    def test_define_function_no_function_defined(self, obj):
        """Persisting code with no function raises ValueError."""
        code = "x = 42"
        with pytest.raises(ValueError):
            obj.define_function(code, "No function")

    def test_define_function_multiple_functions(self, obj):
        """Persisting code with multiple functions raises ValueError."""
        code = "def a(): pass\ndef b(): pass"
        with pytest.raises(ValueError):
            obj.define_function(code, "Too many")

    def test_define_function_name_collision(self, obj):
        """Persisting a function that collides with an existing tool raises ValueError."""
        code = "def add(x: int) -> int:\n    return x"
        with pytest.raises(ValueError):
            obj.define_function(code, "Collides")

    def test_define_function_exec_error(self, obj):
        """Persisting code that raises during exec raises ValueError."""
        code = "import os\nx = 1"
        with pytest.raises(ValueError):
            obj.define_function(code, "Exec error")

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

    @pytest.fixture
    def runner(self):
        return _mock_runner()

    def test_defined_function_can_be_called(self, obj, runner):
        """Calling a defined function through the ToolManager executes it."""
        _wire_runner(obj, runner)
        code = "def double(x: int) -> int:\n    return x * 2"
        obj.define_function(code, "Double a number")
        tool = obj._oap_tool_manager.get_tool("double")
        assert tool is not None
        result = tool.execute(x=21, runner=runner)
        assert result == 42

    def test_defined_function_with_string_output(self, obj, runner):
        """Defined functions that return strings work correctly."""
        _wire_runner(obj, runner)
        code = "def upper(text: str) -> str:\n    return text.upper()"
        obj.define_function(code, "Uppercase a string")
        tool = obj._oap_tool_manager.get_tool("upper")
        result = tool.execute(text="hello", runner=runner)
        assert result == "HELLO"

    def test_defined_function_with_none_returns_ok(self, obj, runner):
        """Defined functions that return None return None (matching _python_exec)."""
        _wire_runner(obj, runner)
        code = "def noop(x: int) -> None:\n    pass"
        obj.define_function(code, "No-op function")
        tool = obj._oap_tool_manager.get_tool("noop")
        result = tool.execute(x=42, runner=runner)
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
        obj.define_function(code, "Multiply by 2")
        assert "my_math" in obj._oap_auto_approve_tools

    def test_remove_function_removes_from_instance_auto_approve(self, obj, runner):
        """remove_function removes the tool from the instance's auto-approve list."""
        code = "def my_math(x: int) -> int:\n    return x * 2"
        obj.define_function(code, "Multiply by 2")
        obj.remove_function("my_math")
        assert "my_math" not in obj._oap_auto_approve_tools

    def test_different_instances_have_separate_lists(self):
        """Two instances of the same class have independent auto-approve lists."""
        obj1 = SimpleAdaptiveObject()
        obj2 = SimpleAdaptiveObject()
        obj1._define_function("def a(x: int) -> int:\n    return x", "A")
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
        obj._define_function(
            "def compute(x: int) -> int:\n    return x * 2",
            "Double a number",
        )

        # Verify it is in instance-level auto-approve list
        assert "compute" in obj._oap_auto_approve_tools

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

    def test_python_exec_sees_dynamically_defined_function_as_bare_name(self):
        """A function defined via define_function is callable as a bare name in python_exec.

        This covers the bug observed in the debug log: after the agent defines a
        function with define_function, it calls python_exec with code that
        references the defined function as a bare name
        (compute_sequence_element(i)). The call fails with

            NameError: name 'compute_sequence_element' is not defined

        because the proxy is only available as self.compute_sequence_element on
        SandboxSelf, not in the sandbox globals dict.
        """
        obj = SimpleAdaptiveObject()
        runner = _mock_runner()

        # Step 1: define a function
        code = (
            "def compute_sequence_element(n: int) -> int:\n"
            "    if n == 0:\n"
            "        return 0\n"
            "    if n == 1:\n"
            "        return 1\n"
            "    a, b = 0, 1\n"
            "    for _ in range(2, n + 1):\n"
            "        a, b = b, a**2 + b**2\n"
            "    return b\n"
        )
        result = obj._define_function(code, "n-th element of the sequence")
        assert result == "OK: registered as 'compute_sequence_element'"

        # Wire sandbox_builder for sandboxed execution
        _wire_runner(obj, runner)
        runner.state = MagicMock()
        runner.state.create = MagicMock()
        runner.state.get.return_value = None
        exec_code = (
            "def func(self):\n"
            "    results = [self.compute_sequence_element(i) for i in range(6)]\n"
            "    return self.produce_output(results[5])\n"
        )
        obj._python_exec(exec_code, runner=runner)
        # Verify the produced value (sequence: 0, 1, 1, 2, 5, 29)
        runner.state.create.assert_called_once()
        data = runner.state.create.call_args[0][1]
        assert data == 29

    def test_python_exec_calls_self_defined_function_via_method_call(self):
        """Test that a function defined on the object can be called as a method on self."""
        from peteos.conversation import Message, ContentPart
        from peteos.persona.agent import Agent
        from peteos.conversation.session import Session
        from peteos.conversation.system_prompt_message import SystemPromptMessage
        from peteos.conversation.tool_definitions_message import ToolDefinitionsMessage

        obj = SimpleAdaptiveObject()
        runner = _mock_runner()
        runner.state = MagicMock()
        runner.state.create = MagicMock()
        runner.state.get.return_value = None

        # Define a function with self as first argument
        code = "def compute_double(self, x: int) -> int:\n    return x * 2"
        result = obj._define_function(code, "Double a number")
        assert result == "OK: registered as 'compute_double'"

        # Wire sandbox_builder for sandboxed execution
        _wire_runner(obj, runner)

        # Call it via python_exec, using self.compute_double(21) as a method
        exec_code = (
            "def func(self):\n"
            "    return self.produce_output(self.compute_double(21))\n"
        )
        obj._python_exec(exec_code, runner=runner)
        runner.state.create.assert_called_once()
        data = runner.state.create.call_args[0][1]
        assert data == 42


class TestDefineFunctionWithUnitTests:
    """Tests for define_function tool."""

    @pytest.fixture
    def obj(self):
        return SimpleAdaptiveObject()

    @pytest.fixture
    def runner(self):
        return _mock_runner()

    def test_no_mocks_no_tests_calls_define_function(self, obj, runner):
        """When no mocks or tests provided, delegates to define_function."""
        code = "def double_it(x: int) -> int:\n    return x * 2"
        result = obj.define_function(code, "Double")
        assert result == "OK: registered as 'double_it'"
        assert "double_it" in obj._oap_define_functions

    def test_all_tests_pass_gets_registered(self, obj, runner):
        """When all tests pass, the function is registered via define_function."""
        code = "def double_it(x: int) -> int:\n    return x * 2"
        tests = (
            "def test_basic(self):\n    assert self.double_it(5) == 10\n"
            "def test_zero(self):\n    assert self.double_it(0) == 0"
        )
        result = obj.define_function(code, "Double", tests=tests)
        assert result == "OK: registered as 'double_it'"
        assert "double_it" in obj._oap_define_functions

    def test_mock_function_used_by_tested_function(self, obj, runner):
        """A mock function on SandboxSelf is called by the tested function."""
        code = (
            "def compute(self, x: int) -> int:\n"
            "    return self.transform(x) * 2"
        )
        mocks = {
            "transform": "def transform(self, x: int) -> int:\n    return x + 1",
        }
        tests = "def test_mocked(self):\n    assert self.compute(5) == 12"
        result = obj.define_function(
            code, "Compute", mocked_functions=mocks, tests=tests,
        )
        assert result == "OK: registered as 'compute'"

    def test_mock_makes_test_pass_that_would_fail_without(self, obj, runner):
        """Without mock, the function would fail, but mock makes it pass."""
        code = (
            "def compute(self, x: int) -> int:\n"
            "    return self.db_query(x) * 2"
        )
        tests = "def test_result(self):\n    assert self.compute(7) == 14"
        with pytest.raises(ValueError):
            obj._test_function_in_sandbox(code, {}, tests)

    def test_mock_makes_test_pass(self, obj, runner):
        """With mock providing db_query, the test passes."""
        code = (
            "def compute(self, x: int) -> int:\n"
            "    return self.db_query(x) * 2"
        )
        mocks = {
            "db_query": "def db_query(self, x: int) -> int:\n    return x",
        }
        tests = "def test_result(self):\n    assert self.compute(7) == 14"
        obj._test_function_in_sandbox(code, mocks, tests)  # no exception = pass

    def test_produce_output_available_in_test(self, obj, runner):
        """produce_output is available as a method on SandboxSelf."""
        code = "def compute(self, x: int) -> int:\n    return x"
        tests = "def test_produce(self):\n    result = self.produce_output('hello')\n    assert result.startswith('produce_output(')"
        obj._test_function_in_sandbox(code, {}, tests)  # no exception = pass

    def test_produce_error_available_in_test(self, obj, runner):
        """produce_error is available as a method on SandboxSelf."""
        code = "def compute(self, x: int) -> int:\n    return x"
        tests = "def test_produce_error(self):\n    result = self.produce_error('oops')\n    assert result.startswith('produce_error(')"
        obj._test_function_in_sandbox(code, {}, tests)  # no exception = pass

    def test_mock_compilation_error(self, obj, runner):
        """A mock with syntax error raises ValueError."""
        code = "def compute(self, x: int) -> int:\n    return x"
        mocks = {
            "bad_mock": "def broken(:\n    return",
        }
        with pytest.raises(ValueError):
            obj._test_function_in_sandbox(code, mocks, "")

    def test_main_compilation_error(self, obj, runner):
        """A main function with syntax error raises ValueError."""
        code = "def broken(:\n    return"
        tests = "def test_ok(self):\n    pass"
        with pytest.raises(ValueError):
            obj._test_function_in_sandbox(code, {}, tests)

    def test_main_name_collision_with_static_tool(self, obj, runner):
        """A function that collides with a static tool raises ValueError."""
        code = "def add(self, x: int) -> int:\n    return x"
        tests = "def test_add(self):\n    assert self.add(1) == 1"
        with pytest.raises(ValueError):
            obj._test_function_in_sandbox(code, {}, tests)

    def test_stop_on_first_test_failure(self, obj, runner):
        """Testing stops on the first failed assertion."""
        code = "def compute(self, x: int) -> int:\n    return x"
        tests = (
            "def test_ok(self):\n    assert self.compute(1) == 1\n"
            "def test_fail(self):\n    assert self.compute(1) == 99\n"
            "def test_never_runs(self):\n    assert self.compute(1) == 1"
        )
        with pytest.raises(ValueError):
            obj._test_function_in_sandbox(code, {}, tests)
