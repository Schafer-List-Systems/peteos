"""Tests for sandboxed python_exec with produce_output."""

from unittest.mock import MagicMock

import pytest

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import agentic_object


@pytest.fixture
def mock_runner():
    """Mock runner with a state that supports create/delete."""
    runner = MagicMock()
    runner.state = MagicMock()
    runner.state.create = MagicMock()
    runner.state.delete = MagicMock()
    runner.state.get.return_value = None
    runner.session_uuid = "test-session"
    return runner


@agentic_object(allow_code_execution=True)
class CodeExecAO(AgenticObject):
    """Minimal AgenticObject with code execution enabled."""
    pass


class TestPythonExecProduceOutput:
    """Exercise python_exec calling produce_output from within the sandbox."""

    def test_produce_output_returns_value_with_mock_runner(self, mock_runner):
        """Call produce_output with a valid runner — should return a success string."""
        ao = CodeExecAO()
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
        mock_runner.state.create.side_effect = ValueError("state backend unavailable")
        ao = CodeExecAO()
        code = (
            "def func(self):\n"
            "    return self.produce_output('hello')"
        )
        result = ao._python_exec(code, runner=mock_runner)
        assert "Error" in result
