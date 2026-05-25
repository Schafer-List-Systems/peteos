from peteos.toolmanager import Tool, ToolManager


def test_tool_creation_from_callable():
    """Test creating a Tool from a callable."""

    def my_tool(arg1: str, arg2: int) -> str:
        """This is my tool description."""
        return f"{arg1}: {arg2}"

    tool = Tool.from_callable(my_tool)

    assert tool.name == "my_tool"
    assert tool.description == "This is my tool description."
    assert tool.func.__name__ == my_tool.__name__
    assert tool.func.__doc__ == my_tool.__doc__
    assert "arg1" in tool.parameters
    assert "arg2" in tool.parameters
    assert "session" not in tool.parameters


def test_tool_call():
    """Test calling a tool."""

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    tool = Tool(name="add", description="Add two numbers", func=add)
    result = tool(a=2, b=3)

    assert result == 5


def test_tool_manager_register_tool_instance():
    """Test registering a tool by instance."""

    def my_func(x: str) -> str:
        return x.upper()

    tool = Tool(name="uppercase", description="Convert to uppercase", func=my_func)
    manager = ToolManager()

    manager.register_tool(tool)

    assert manager.get_tool("uppercase") is not None
    assert manager.get_tool("uppercase").name == "uppercase"


def test_tool_manager_register_tool_from_callable():
    """Test registering a tool from a callable."""

    def greet(name: str) -> str:
        """Greet a person."""
        return f"Hello, {name}"

    manager = ToolManager()
    manager.register_tool(func=greet)

    tool = manager.get_tool("greet")
    assert tool is not None
    assert tool.name == "greet"
    assert tool.description == "Greet a person."


def test_tool_manager_register_tool_with_custom_metadata():
    """Test registering a tool with custom name and description."""

    def custom_func(x: int) -> int:
        return x * 2

    manager = ToolManager()
    manager.register_tool(
        func=custom_func,
        name="double",
        description="Double the input value",
        parameters={"x": {"type": "int", "required": True}}
    )

    tool = manager.get_tool("double")
    assert tool is not None
    assert tool.name == "double"
    assert tool.description == "Double the input value"


def test_tool_manager_get_nonexistent_tool():
    """Test getting a tool that doesn't exist."""

    manager = ToolManager()
    tool = manager.get_tool("nonexistent")

    assert tool is None


def test_tool_manager_get_tool_list_no_filter():
    """Test get_tool_list without filter returns all tools."""

    manager = ToolManager()
    manager.register_tool(Tool(name="alpha", description="a", func=lambda: None))
    manager.register_tool(Tool(name="beta", description="b", func=lambda: None))
    manager.register_tool(Tool(name="gamma", description="c", func=lambda: None))

    tools = manager.get_tool_list()
    assert len(tools) == 3


def test_tool_manager_get_tool_list_filter_patterns():
    """Test get_tool_list filters by regex patterns."""

    manager = ToolManager()
    manager.register_tool(Tool(name="mute_router", description="a", func=lambda: None))
    manager.register_tool(Tool(name="unmute_router", description="b", func=lambda: None))
    manager.register_tool(Tool(name="add_exclude_pattern", description="c", func=lambda: None))
    manager.register_tool(Tool(name="set_approval_result", description="d", func=lambda: None))

    tools = manager.get_tool_list(filter_patterns=["mute_router", "unmute_router"])
    assert len(tools) == 2
    assert {t.name for t in tools} == {"mute_router", "unmute_router"}


def test_tool_manager_get_tool_list_filter_patterns_regex():
    """Test get_tool_list filter supports regex patterns."""

    manager = ToolManager()
    manager.register_tool(Tool(name="mute_router", description="a", func=lambda: None))
    manager.register_tool(Tool(name="unmute_router", description="b", func=lambda: None))
    manager.register_tool(Tool(name="set_approval_result", description="c", func=lambda: None))

    tools = manager.get_tool_list(filter_patterns=[".*_router"])
    assert len(tools) == 2
    assert {t.name for t in tools} == {"mute_router", "unmute_router"}


def test_tool_manager_get_tool_list_filter_patterns_no_match():
    """Test get_tool_list returns empty when no patterns match."""

    manager = ToolManager()
    manager.register_tool(Tool(name="alpha", description="a", func=lambda: None))

    tools = manager.get_tool_list(filter_patterns=[".*nonexistent.*"])
    assert len(tools) == 0
