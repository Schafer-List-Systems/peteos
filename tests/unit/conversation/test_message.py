"""Unit tests for MessageRegistry and Message."""

import pytest
from peteos.conversation.message import ContentPart, Message
from peteos.conversation.message_registry import MessageRegistry

# Import production classes at module level so they are registered once
# via __init_subclass__ and survive across tests.
from peteos.conversation.system_prompt_message import SystemPromptMessage  # noqa: F401
from peteos.conversation.tool_definitions_message import ToolDefinitionsMessage  # noqa: F401


class TestContentPart:
    """Tests for ContentPart factory methods and properties."""

    def test_create_text(self):
        part = ContentPart.create_text("Hello, world!")
        assert part.type == "text"
        assert part.text == "Hello, world!"
        assert part.raw_dict == {"type": "text", "text": "Hello, world!"}

    def test_create_thinking(self):
        part = ContentPart.create_thinking("Let me think about this...")
        assert part.type == "thinking"
        assert part.text == "Let me think about this..."

    def test_create_image_base64(self):
        source = {"type": "base64", "data": "abc123", "media_type": "image/png"}
        part = ContentPart.create_image(source)
        assert part.type == "image"
        assert part.source == source
        assert part.source["media_type"] == "image/png"

    def test_create_image_url(self):
        source = {"type": "url", "url": "https://example.com/img.png"}
        part = ContentPart.create_image(source)
        assert part.type == "image"
        assert part.source["type"] == "url"

    def test_create_video(self):
        source = {"type": "base64", "data": "vid", "media_type": "video/mp4"}
        part = ContentPart.create_video(source)
        assert part.type == "video"
        assert part.source["media_type"] == "video/mp4"

    def test_create_audio(self):
        source = {"type": "url", "url": "https://example.com/audio.mp3"}
        part = ContentPart.create_audio(source)
        assert part.type == "audio"
        assert part.source["type"] == "url"

    def test_create_tool_use(self):
        part = ContentPart.create_tool_use("call_123", "calculate", '{"a": 1, "b": 2}')
        assert part.type == "tool_use"
        assert part.call_id == "call_123"
        assert part.name == "calculate"
        assert part.arguments == '{"a": 1, "b": 2}'

    def test_create_tool_result(self):
        part = ContentPart.create_tool_result("call_123", "42")
        assert part.type == "tool_result"
        assert part.call_id == "call_123"
        assert part.content == "42"

    def test_create_tool_definition(self):
        params = {"type": "object", "properties": {"x": {"type": "number"}}}
        part = ContentPart.create_tool("add", "Adds two numbers", params)
        assert part.type == "tool"
        assert part.name == "add"
        assert part.description == "Adds two numbers"
        assert part.parameters == params

    def test_direct_init(self):
        raw = {"type": "text", "text": "Direct init"}
        part = ContentPart(raw)
        assert part.type == "text"
        assert part.text == "Direct init"

    def test_nullable_properties_return_none(self):
        part = ContentPart({"type": "text", "text": "no id here"})
        assert part.call_id is None
        assert part.name is None
        assert part.arguments is None
        assert part.content is None
        assert part.source is None
        assert part.description is None
        assert part.parameters is None


class TestMessageRegistry:
    """Tests for MessageRegistry."""

    def test_register_and_get(self):
        class CustomMessage(Message):
            pass

        cls = MessageRegistry.get("CustomMessage")
        assert cls is CustomMessage
        assert cls is not Message

    def test_get_nonexistent_returns_none(self):
        assert MessageRegistry.get("NonExistentClass") is None

    def test_get_all(self):
        class FirstMsg(Message):
            pass

        class SecondMsg(Message):
            pass

        all_reg = MessageRegistry.get_all()
        assert "FirstMsg" in all_reg
        assert "SecondMsg" in all_reg
        assert all_reg["FirstMsg"] is FirstMsg
        assert all_reg["SecondMsg"] is SecondMsg

    def test_clear(self):
        class TempMsg(Message):
            pass

        assert "TempMsg" in MessageRegistry.get_all()
        MessageRegistry.clear()
        assert "TempMsg" not in MessageRegistry.get_all()


class TestMessage:
    """Tests for Message base class."""

    def test_create_basic(self):
        part = ContentPart.create_text("Hello")
        msg = Message.create("user", [part])
        assert msg.role == "user"
        assert len(msg.content) == 1
        assert msg.content[0].text == "Hello"

    def test_create_with_metadata(self):
        part = ContentPart.create_text("Hi")
        msg = Message.create("user", [part], metadata={"key": "val"})
        assert msg.metadata == {"key": "val"}

    def test_create_without_metadata(self):
        part = ContentPart.create_text("Hi")
        msg = Message.create("user", [part])
        assert msg.metadata == {}

    def test_raw_dict(self):
        part = ContentPart.create_text("Raw test")
        msg = Message.create("assistant", [part])
        rd = msg.raw_dict
        assert rd["role"] == "assistant"
        assert rd["content"][0]["type"] == "text"
        assert "_type" in rd

    def test_automatic_type_field(self):
        part = ContentPart.create_text("type test")
        msg = Message.create("user", [part])
        assert msg.raw_dict["_type"] == "Message"

    def test_hook_ids_default_empty(self):
        part = ContentPart.create_text("hooks test")
        msg = Message.create("user", [part])
        assert msg.hook_ids == []

    def test_creation_timestamp(self):
        part = ContentPart.create_text("timestamp test")
        msg = Message.create("user", [part])
        assert msg.creation_timestamp is not None

    def test_from_dict_basic(self):
        raw = {"_type": "Message", "role": "user", "content": [{"type": "text", "text": "from dict"}]}
        msg = Message.from_dict(raw)
        assert msg.role == "user"
        assert msg.content[0].text == "from dict"

    def test_from_dict_with_type(self):
        class CustomMessage(Message):
            pass

        raw = {"_type": "CustomMessage", "role": "user", "content": []}
        msg = Message.from_dict(raw)
        assert isinstance(msg, CustomMessage)

    def test_from_dict_strict_missing_raises(self):
        raw = {"_type": "DoesNotExist", "role": "user", "content": []}
        with pytest.raises(ValueError, match="Registered message class not found"):
            Message.from_dict(raw, strict=True)

    def test_from_dict_non_strict_fallback(self):
        raw = {"_type": "DoesNotExist", "role": "user", "content": [{"type": "text", "text": "fallback"}]}
        msg = Message.from_dict(raw, strict=False)
        assert isinstance(msg, Message)
        assert msg.role == "user"

    def test_printable_text(self):
        part = ContentPart.create_text("Print me")
        msg = Message.create("user", [part])
        assert msg.printable() == "Print me"

    def test_printable_tool_use(self):
        part = ContentPart.create_tool_use("call_1", "my_tool", "{}")
        msg = Message.create("assistant", [part])
        printable = msg.printable()
        assert "ToolCall: my_tool" in printable
        assert "call_1" in printable

    def test_printable_tool_result(self):
        part = ContentPart.create_tool_result("call_1", "result data here")
        msg = Message.create("user", [part])
        printable = msg.printable()
        assert "ToolResult of call_1" in printable
        assert "16 chars" in printable

    def test_printable_multi_part(self):
        parts = [
            ContentPart.create_text("Hello"),
            ContentPart.create_tool_use("call_1", "calc", "{}"),
            ContentPart.create_text("Done"),
        ]
        msg = Message.create("assistant", parts)
        lines = msg.printable().split("\n")
        assert len(lines) == 3
        assert lines[0] == "Hello"
        assert "ToolCall: calc" in lines[1]
        assert lines[2] == "Done"

    def test_printable_image(self):
        part = ContentPart.create_image({"type": "base64", "data": "x", "media_type": "image/png"})
        msg = Message.create("user", [part])
        assert "[Image: media_type=image/png]" in msg.printable()

    def test_printable_unknown_type(self):
        part = ContentPart({"type": "custom_type", "some": "data"})
        msg = Message.create("user", [part])
        assert "[Unknown: custom_type]" in msg.printable()

    def test_count_tokens_caches(self):
        part = ContentPart.create_text("Tokenize me")
        msg = Message.create("user", [part])
        count1 = msg.count_tokens()
        count2 = msg.count_tokens()
        assert count1 == count2
        assert "token_count" in msg.raw_dict

    def test_count_tokens_reuses_cached(self):
        part = ContentPart.create_text("Cached tokens")
        msg = Message.create("user", [part])
        # Manually set a fake token count
        msg.raw_dict["token_count"] = 42
        assert msg.count_tokens() == 42

    def test_id_default_uuid(self):
        part = ContentPart.create_text("no id")
        msg = Message.create("user", [part])
        assert msg.id != ""

    def test_role_default_empty(self):
        raw = {"content": []}
        msg = Message(raw)
        assert msg.role == ""
