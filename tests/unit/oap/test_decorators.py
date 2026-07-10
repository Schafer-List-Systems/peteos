"""Tests for OAP decorators."""

from peteos.oap.decorators import agentic_object, tool


class TestToolDecorator:
    def test_tool_defaults(self):
        @tool()
        def my_func(x: int, y: str = "default") -> str:
            """Do something."""
            return str(x)

        assert my_func._tool_name == "my_func"
        assert my_func._tool_description == "Do something."

    def test_tool_custom_name(self):
        @tool(name="custom_name")
        def func(x: int) -> int:
            pass

        assert func._tool_name == "custom_name"

    def test_tool_custom_description(self):
        @tool(description="custom desc")
        def func(x: int) -> int:
            """Docstring."""
            pass

        assert func._tool_description == "custom desc"

    def test_tool_no_docstring(self):
        @tool()
        def func(x: int) -> int:
            pass

        assert func._tool_description == ""

    def test_undecorated_has_no_tool_name(self):
        def not_a_tool(x: int) -> int:
            return x

        assert not hasattr(not_a_tool, "_tool_name")


class TestAgenticObjectDecorator:
    def test_decorator_defaults(self):
        @agentic_object()
        class MyObj:
            pass

        assert MyObj._oap_config == {
            "imports": [],
            "import_aliases": {},
            "invoke_sub_agents": False,
            "allow_code_execution": False,
            "define_functions": False,
            "role": None,
        }

    def test_decorator_config(self):
        @agentic_object(
            imports=["os", "sys"],
            invoke_sub_agents=True,
            allow_code_execution=True,
        )
        class MyObj:
            pass

        assert MyObj._oap_config == {
            "imports": ["os", "sys"],
            "import_aliases": {},
            "invoke_sub_agents": True,
            "allow_code_execution": True,
            "define_functions": False,
            "role": None,
        }

    def test_decorator_role_override(self):
        @agentic_object(role="custom_role")
        class MyObj:
            pass

        assert MyObj._oap_config == {
            "imports": [],
            "import_aliases": {},
            "invoke_sub_agents": False,
            "allow_code_execution": False,
            "define_functions": False,
            "role": "custom_role",
        }

    def test_imports_not_discoverable_via_attribute(self):
        @agentic_object(imports=["os"])
        class MyObj:
            pass

        obj = MyObj()
        # Access via instance should not reveal imports as a regular attribute
        assert not hasattr(obj, "imports")
