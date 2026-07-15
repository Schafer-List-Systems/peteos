"""Tests for sandboxed python_exec with produce_output and dynamic functions."""

from unittest.mock import MagicMock

import pytest

from peteos.oap.adaptive_object import AdaptiveObject
from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import agentic_object


@pytest.fixture
def mock_runner():
    """Mock runner with a state that supports create/delete (no sandbox_builder pre-wired)."""
    runner = MagicMock()
    runner.state = MagicMock()
    runner.state.create = MagicMock()
    runner.state.delete = MagicMock()
    runner.state.get.return_value = None
    runner.session_uuid = "test-session"
    return runner


def _wire_builder(obj, runner):
    """Wire runner.sandbox_builder using the object's own _create_sandbox_builder."""
    config = {"invoke_sub_agents": False}
    runner.sandbox_builder = obj._create_sandbox_builder(config, runner=runner)


@agentic_object(allow_code_execution=True)
class CodeExecAO(AgenticObject):
    """Minimal AgenticObject with code execution enabled."""
    pass


@agentic_object(allow_code_execution=True)
class AdaptiveCodeExec(AdaptiveObject):
    """AdaptiveObject with code execution enabled for dynamic function tests."""
    pass


class TestPythonExecProduceOutput:
    """Exercise python_exec calling produce_output from within the sandbox."""

    def test_produce_output_returns_value_with_mock_runner(self, mock_runner):
        """Call produce_output with a valid runner — should return a success string."""
        ao = CodeExecAO()
        _wire_builder(ao, mock_runner)
        code = (
            "def func(self):\n"
            "    seq = [0, 1]\n"
            "    for i in range(2, 7):\n"
            "        seq.append(seq[i-1]**2 + seq[i-2]**2)\n"
            "    return self.produce_output(seq[6])"
        )
        result = ao._python_exec(code, runner=mock_runner)
        assert result is None
        mock_runner.state.create.assert_called_once()
        data = mock_runner.state.create.call_args[0][1]
        assert data == 866

    def test_produce_output_fails_with_broken_runner(self, mock_runner):
        """Call produce_output with a runner that can't hold state — should get an error string."""
        ao = CodeExecAO()
        _wire_builder(ao, mock_runner)
        mock_runner.state.create.side_effect = ValueError("state backend unavailable")
        code = (
            "def func(self):\n"
            "    return self.produce_output('hello')"
        )
        result = ao._python_exec(code, runner=mock_runner)
        assert "Error" in result


class TestPythonExecWithDynamicFunction:
    """Verify that functions defined via define_function are callable inside python_exec."""

    def test_dynamic_function_callable_from_python_exec(self, mock_runner):
        """Define a function then call it from within a sandboxed python_exec."""
        obj = AdaptiveCodeExec()
        _wire_builder(obj, mock_runner)

        # Step 1: define the function
        code = (
            "def compute_sequence_element(n: int) -> int:\n"
            "    if n == 0:\n"
            "        return 0\n"
            "    if n == 1:\n"
            "        return 1\n"
            "    a, b = 0, 1\n"
            "    for _ in range(2, n + 1):\n"
            "        a, b = b, b**2 + a**2\n"
            "    return b\n"
        )
        result = obj._define_function(code, "n-th element of the sequence", mock_runner)
        assert result == "OK: registered as 'compute_sequence_element'"
        assert "compute_sequence_element" in obj._oap_define_functions

        # Step 2: call the defined function from python_exec
        exec_code = (
            "def func(self):\n"
            "    result = [self.compute_sequence_element(i) for i in range(6)]\n"
            "    return self.produce_output(result[5])\n"
        )
        result = obj._python_exec(exec_code, runner=mock_runner)
        assert result is None
        mock_runner.state.create.assert_called_once()
        data = mock_runner.state.create.call_args[0][1]
        # 0, 1, 1, 2, 5, 29
        assert data == 29

    def test_undefined_builtin_raises_name_error(self):
        """Access a symbol that was never imported and expect a NameError."""
        obj = CodeExecAO()
        runner = MagicMock()
        runner.state = MagicMock()
        runner.state.create = MagicMock()
        runner.state.delete = MagicMock()
        runner.state.get.return_value = None
        runner.session_uuid = "test-session"
        _wire_builder(obj, runner)
        exec_code = (
            "def func(self):\n"
            "    return nonexistent_module.__name__\n"
        )
        with pytest.raises(NameError, match="nonexistent_module"):
            obj._python_exec(exec_code, runner=runner)
