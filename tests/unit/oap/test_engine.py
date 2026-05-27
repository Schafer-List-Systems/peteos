"""Tests for OAP engine utilities."""

from peteos.oap.engine import (
    _detect_task_failure,
    _discover_tools,
    _generate_system_prompt,
)
from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import agentic_object, tool


class TestDiscoverTools:
    def test_discovers_decorated_methods(self):
        @agentic_object()
        class MyObj(AgenticObjectBase):
            @tool()
            def hello(self, name: str) -> str:
                """Say hello."""
                return f"Hello, {name}"

            def not_a_tool(self):
                pass

        tm = _discover_tools(MyObj())
        tools = tm.get_tool_list()
        assert len(tools) == 1
        assert tools[0].name == "hello"
        assert "Say hello" in tools[0].description

    def test_discovers_no_tools(self):
        @agentic_object()
        class MyObj(AgenticObjectBase):
            pass

        tm = _discover_tools(MyObj())
        assert len(tm.get_tool_list()) == 0


class TestGenerateSystemPrompt:
    def test_includes_docstring(self):
        @agentic_object()
        class Greeter(AgenticObjectBase):
            """A greeter object."""

        prompt = _generate_system_prompt(Greeter(), None)
        assert "greeter" in prompt.lower() or "Greeter" in prompt

    def test_includes_code_execution_capability(self):
        @agentic_object(allow_code_execution=True)
        class CodeObj(AgenticObjectBase):
            pass

        prompt = _generate_system_prompt(CodeObj(), None)
        assert "code execution" in prompt.lower()

    def test_includes_sub_agent_capability(self):
        @agentic_object(invoke_sub_agents=True)
        class ParentObj(AgenticObjectBase):
            pass

        prompt = _generate_system_prompt(ParentObj(), None)
        assert "sub-agent" in prompt.lower() or "sub agent" in prompt.lower()


class TestDetectTaskFailure:
    def test_detects_error_prefix(self):
        assert _detect_task_failure("Error: something failed") is True

    def test_detects_i_cannot(self):
        assert _detect_task_failure("I cannot process this request") is True

    def test_detects_unable_to(self):
        assert _detect_task_failure("Unable to complete the task") is True

    def test_no_failure_on_normal_response(self):
        assert _detect_task_failure("The answer is 42") is False

    def test_no_failure_on_json(self):
        assert _detect_task_failure('{"result": "success"}') is False
