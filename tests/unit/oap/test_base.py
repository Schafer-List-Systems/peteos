"""Tests for AgenticObjectBase."""

import pytest

from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import agentic_object, tool
from peteos.oap.error import Error
from peteos.persona.toolmanager import ToolManager


class TestAgenticObjectBaseInit:
    def test_default_init(self):
        obj = AgenticObjectBase()
        assert obj.agent is not None
        assert obj.role is not None
        assert obj.role.name == "oap_AgenticObjectBase"
        assert isinstance(obj._oap_tool_manager, ToolManager)

    def test_agent_property(self):
        obj = AgenticObjectBase()
        original = obj.agent
        assert original is not None
        obj.agent = "mock_agent"
        assert obj.agent == "mock_agent"
        # Should be able to restore
        obj.agent = original

    def test_role_auto_created(self):
        obj = AgenticObjectBase()
        assert isinstance(obj.role, type(obj.role))

    def test_system_prompt_default_no_docstring(self):
        obj = AgenticObjectBase()
        assert "You are an agent working on a AgenticObjectBase object" in obj.role.system_prompt

    def test_system_prompt_from_class_docstring(self):
        class DocstringObj(AgenticObjectBase):
            """You are a special analysis agent."""
        obj = DocstringObj()
        assert obj.role.system_prompt == "You are a special analysis agent."
        assert "You are an agent working on a DocstringObj object" not in obj.role.system_prompt

    def test_system_prompt_empty_docstring_falls_back(self):
        class EmptyDocstringObj(AgenticObjectBase):
            """"""
        obj = EmptyDocstringObj()
        assert "You are an agent working on a EmptyDocstringObj object" in obj.role.system_prompt

    def test_system_prompt_strips_whitespace(self):
        class WhitespaceDocstringObj(AgenticObjectBase):
            """

            You are a whitespacey agent.

            """
        obj = WhitespaceDocstringObj()
        assert obj.role.system_prompt == "You are a whitespacey agent."


class TestToolRegistry:
    def test_tool_decorated_methods_are_registered(self):
        obj = SimpleToolObj()
        tools = obj._oap_tool_manager.get_tool_list()
        assert len(tools) == 3
        assert {t.name for t in tools} == {"hello", "produce_output", "produce_error"}

    def test_tool_execution_works(self):
        obj = SimpleToolObj()
        t = obj._oap_tool_manager.get_tool("hello")
        assert t.func(name="World") == "Hello, World"

    def test_multiple_tools_registered(self):
        obj = MultiToolObj()
        tools = obj._oap_tool_manager.get_tool_list()
        assert len(tools) == 4
        assert {t.name for t in tools} == {"hello", "greet", "produce_output", "produce_error"}

    def test_inherited_tools_registered(self):
        obj = ChildToolObj()
        tools = obj._oap_tool_manager.get_tool_list()
        assert len(tools) == 4
        assert {t.name for t in tools} == {"parent_tool", "child_tool", "produce_output", "produce_error"}

    def test_override_replaces_parent_tool(self):
        obj = OverrideToolObj()
        tools = obj._oap_tool_manager.get_tool_list()
        assert len(tools) == 3
        assert tools[0].func() == "child"
        assert {t.name for t in tools} == {"tool_a", "produce_output", "produce_error"}

    def test_custom_name_used(self):
        obj = CustomNameToolObj()
        assert obj._oap_tool_manager.get_tool("custom_name") is not None
        assert obj._oap_tool_manager.get_tool("internal_method") is None

    def test_tool_description_from_docstring(self):
        obj = DocstringToolObj()
        t = obj._oap_tool_manager.get_tool("my_tool")
        assert t.description == "This is the description."

    def test_custom_description_used(self):
        obj = CustomDescToolObj()
        t = obj._oap_tool_manager.get_tool("my_tool")
        assert t.description == "custom desc"

    def test_no_tools_no_error(self):
        obj = NoToolObj()
        tools = obj._oap_tool_manager.get_tool_list()
        assert len(tools) == 2
        assert {t.name for t in tools} == {"produce_output", "produce_error"}


# --- Module-level class fixtures (avoids Python 3.12 closure issue) ---

@agentic_object()
class SimpleToolObj(AgenticObjectBase):
    @tool()
    def hello(self, name: str) -> str:
        """Say hello."""
        return f"Hello, {name}"

    def not_a_tool(self):
        pass


@agentic_object()
class MultiToolObj(AgenticObjectBase):
    @tool()
    def hello(self, name: str) -> str:
        """Say hello."""
        return f"Hello, {name}"

    @tool(name="greet")
    def greet(self, name: str) -> str:
        """Greet someone."""
        return f"Hi, {name}"


@agentic_object()
class ParentToolObj(AgenticObjectBase):
    @tool()
    def parent_tool(self) -> str:
        """Parent tool."""
        return "parent"


@agentic_object()
class ChildToolObj(ParentToolObj):
    @tool()
    def child_tool(self) -> str:
        """Child tool."""
        return "child"


@agentic_object()
class ParentOverrideObj(AgenticObjectBase):
    @tool()
    def tool_a(self) -> str:
        """Parent version."""
        return "parent"


@agentic_object()
class OverrideToolObj(ParentOverrideObj):
    @tool(name="tool_a")
    def tool_a_override(self) -> str:
        """Child version."""
        return "child"


@agentic_object()
class CustomNameToolObj(AgenticObjectBase):
    @tool(name="custom_name")
    def internal_method(self) -> str:
        """Custom name."""
        return "ok"


@agentic_object()
class DocstringToolObj(AgenticObjectBase):
    @tool()
    def my_tool(self) -> str:
        """This is the description."""
        return "ok"


@agentic_object()
class CustomDescToolObj(AgenticObjectBase):
    @tool(description="custom desc")
    def my_tool(self) -> str:
        """Docstring."""
        return "ok"


@agentic_object()
class NoToolObj(AgenticObjectBase):
    def regular_method(self):
        pass


# --- Diamond inheritance test classes ---

import cv2
import numpy


@agentic_object(allow_code_execution=True, imports=[numpy], import_aliases={"numpy": "np"})
class BBase(AgenticObjectBase):
    """Branch B with numpy support."""


@agentic_object(allow_code_execution=True, imports=[cv2])
class CBase(AgenticObjectBase):
    """Branch C with opencv support."""


@agentic_object()
class DiamondChild(BBase, CBase):
    """Diamond child inheriting from both branches."""


class TestDiamondConfigCollection:
    def test_config_merges_imports_from_both_branches(self):
        d = DiamondChild()
        tools = d._oap_tool_manager.get_tool_list()
        tool_names = {t.name for t in tools}
        assert "python_exec" in tool_names

    def test_python_exec_sandbox_has_combined_imports(self):
        d = DiamondChild()
        result = d._python_exec("x = numpy.array([1, 2, 3]); y = cv2.__name__")
        assert result == "OK"

    def test_python_exec_sandbox_has_import_alias(self):
        d = DiamondChild()
        result = d._python_exec("x = np.array([1, 2, 3])")
        assert result == "OK"
