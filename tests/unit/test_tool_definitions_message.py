"""Tests for ToolDefinitionsMessage dynamic tool content."""

from peteos.chatbot import ContentPart, ToolDefinitionsMessage
from peteos.toolmanager import Tool, ToolManager


def _fn(name, doc=""):
    """Factory that creates a function with the given name and docstring."""
    def f(**kwargs):
        pass
    f.__name__ = name
    f.__doc__ = doc
    return f


class TestToolDefinitionsMessage:
    """Tests for dynamic tool definition content generation."""

    def test_dynamic_content_reflects_current_tool_list(self):
        tm = ToolManager()
        msg = ToolDefinitionsMessage(tool_manager=tm)
        # Initially empty
        parts = msg.content
        assert len(parts) == 0

        # Add a tool and access content again
        tm.register_tool(Tool.from_callable(_fn("foo", "desc")))
        parts = msg.content
        assert len(parts) == 1
        assert parts[0].type == "tool"
        assert parts[0].data["name"] == "foo"
        assert parts[0].data["description"] == "desc"

    def test_content_reflects_tool_replacement(self):
        tm = ToolManager()
        tm.register_tool(Tool.from_callable(_fn("bar", "old desc")))
        msg = ToolDefinitionsMessage(tool_manager=tm)
        assert len(msg.content) == 1
        assert msg.content[0].data["description"] == "old desc"

        # Re-register with same name, replaces
        tm.register_tool(Tool.from_callable(_fn("bar", "new desc")))
        parts = msg.content
        assert len(parts) == 1
        assert parts[0].data["description"] == "new desc"

    def test_multiple_tools(self):
        tm = ToolManager()
        tm.register_tool(Tool.from_callable(_fn("tool_a", "desc_a")))
        tm.register_tool(Tool.from_callable(_fn("tool_b", "desc_b")))
        tm.register_tool(Tool.from_callable(_fn("tool_c", "desc_c")))
        msg = ToolDefinitionsMessage(tool_manager=tm)

        parts = msg.content
        assert len(parts) == 3
        names = {p.data["name"] for p in parts}
        assert names == {"tool_a", "tool_b", "tool_c"}

    def test_role_is_tool(self):
        tm = ToolManager()
        msg = ToolDefinitionsMessage(tool_manager=tm)
        assert msg.get_role() == "tool"

    def test_to_dict_marker(self):
        tm = ToolManager()
        tm.register_tool(Tool.from_callable(_fn("test_tool", "desc")))
        msg = ToolDefinitionsMessage(tool_manager=tm)
        d = msg.to_dict()
        assert d["role"] == "tool"
        assert d["type"] == "tool_definitions"
        assert d["content"] == []

    def test_serialize_content_works(self):
        """Verify GenericChatBot._build_body can iterate the content."""
        tm = ToolManager()
        tm.register_tool(Tool.from_callable(_fn("my_tool", "desc")))
        msg = ToolDefinitionsMessage(tool_manager=tm)

        serialized = msg.serialize_content()
        assert len(serialized) == 1
        assert serialized[0]["type"] == "tool"
        assert serialized[0]["name"] == "my_tool"
        assert serialized[0]["description"] == "desc"

    def test_serialize_content_multiple_tools(self):
        tm = ToolManager()
        tm.register_tool(Tool.from_callable(_fn("tool1", "desc1")))
        tm.register_tool(Tool.from_callable(_fn("tool2", "desc2")))
        msg = ToolDefinitionsMessage(tool_manager=tm)

        serialized = msg.serialize_content()
        assert len(serialized) == 2

        # GenericChatBot._build_body extracts tools like this:
        tools_list = []
        for part in msg.content:
            if part.type == "tool":
                tools_list.append(part.data)
        assert len(tools_list) == 2
        assert tools_list[0]["name"] == "tool1"
        assert tools_list[1]["name"] == "tool2"

    def test_count_tokens_recomputes(self):
        tm = ToolManager()
        msg = ToolDefinitionsMessage(tool_manager=tm)
        # Should not crash with empty tool list
        count = msg.count_tokens()
        assert count > 0

        tm.register_tool(Tool.from_callable(_fn("big_tool", "some description")))
        count2 = msg.count_tokens()
        assert count2 > count  # More tokens with a tool definition
