"""Unit tests for ChatHistory and related classes."""

import pytest
from datetime import datetime
from peteos.chatbot import ChatHistory, Message, ContentPart


class TestContentPart:
    """Tests for ContentPart class."""

    def test_text_part_creation(self):
        """Test creating a text ContentPart."""
        part = ContentPart(part_type="text", text="Hello world")
        assert part.type == "text"
        assert part.text == "Hello world"
        assert part.source is None

    def test_image_part_creation(self):
        """Test creating an image ContentPart."""
        part = ContentPart(
            part_type="image",
            source={"type": "base64", "data": "abc123", "mime_type": "image/jpeg"}
        )
        assert part.type == "image"
        assert part.source["type"] == "base64"
        assert part.text is None

    def test_to_dict_method(self):
        """Test converting ContentPart to dictionary."""
        part = ContentPart(part_type="text", text="Test")
        d = part.to_dict()
        assert d["type"] == "text"
        assert d["text"] == "Test"

    def test_from_dict_method(self):
        """Test creating ContentPart from dictionary."""
        d = {"type": "text", "text": "Test"}
        part = ContentPart.from_dict(d)
        assert part.type == "text"
        assert part.text == "Test"

    def test_repr_method(self):
        """Test ContentPart string representation."""
        part = ContentPart(part_type="text", text="Test")
        repr_str = repr(part)
        assert "ContentPart" in repr_str
        assert "text" in repr_str


class TestToolMessage:
    """Tests for tool messages (role="tool")."""

    def test_tool_message_creation(self):
        """Test creating a tool message."""
        msg = Message(
            role="tool",
            content=[ContentPart(
                part_type="tool",
                name="get_weather",
                description="Get weather for a city",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}}
            )]
        )
        assert msg.role == "tool"
        assert msg.content[0].type == "tool"
        assert msg.content[0].data["name"] == "get_weather"

    def test_tool_message_to_dict(self):
        """Test converting tool message to dictionary."""
        msg = Message(
            role="tool",
            content=[ContentPart(
                part_type="tool",
                name="test",
                description="desc",
                parameters={"type": "object"}
            )]
        )
        d = msg.to_dict()
        assert d["role"] == "tool"
        assert d["content"][0]["type"] == "tool"
        assert d["content"][0]["name"] == "test"


class TestMessage:
    """Tests for Message class."""

    def test_message_creation(self):
        """Test creating a Message."""
        msg = Message(
            role="user",
            content=[ContentPart(part_type="text", text="Hello")]
        )
        assert msg.role == "user"
        assert len(msg.content) == 1
        assert msg.text == "Hello"

    def test_multi_part_message(self):
        """Test creating a message with multiple content parts."""
        msg = Message(
            role="user",
            content=[
                ContentPart(part_type="text", text="What's in this image?"),
                ContentPart(part_type="image", source={"type": "base64", "data": "abc"})
            ]
        )
        assert len(msg.content) == 2
        assert "What's in this image?" in msg.text

    def test_message_text_property(self):
        """Test extracting text from message content."""
        msg = Message(
            role="assistant",
            content=[
                ContentPart(part_type="text", text="Hello "),
                ContentPart(part_type="text", text="world")
            ]
        )
        assert msg.text == "Hello world"

    def test_message_to_dict(self):
        """Test converting Message to dictionary."""
        msg = Message(
            role="user",
            content=[ContentPart(part_type="text", text="Test")]
        )
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"][0]["type"] == "text"
        assert d["content"][0]["text"] == "Test"

    def test_message_from_dict(self):
        """Test creating Message from dictionary."""
        d = {
            "role": "user",
            "content": [{"type": "text", "text": "Test"}],
            "metadata": {"source": "test"}
        }
        msg = Message.from_dict(d)
        assert msg.role == "user"
        assert len(msg.content) == 1
        assert msg.content[0].type == "text"

    def test_message_from_dict_missing_role_raises(self):
        """Test that missing role raises ValueError."""
        d = {"content": [{"type": "text", "text": "Test"}]}
        with pytest.raises(ValueError, match="must contain a 'role' field"):
            Message.from_dict(d)

    def test_message_from_string_content(self):
        """Test creating Message with string content (backward compatibility)."""
        d = {"role": "user", "content": "Hello"}
        msg = Message.from_dict(d)
        assert len(msg.content) == 1
        assert msg.content[0].type == "text"
        assert msg.text == "Hello"

    def test_message_with_metadata(self):
        """Test creating Message with metadata."""
        msg = Message(
            role="user",
            content=[ContentPart(part_type="text", text="Test")],
            metadata={"source": "api", "timestamp": 123}
        )
        assert msg.metadata["source"] == "api"

    def test_message_id_auto_generated(self):
        """Test that message ID is auto-generated."""
        msg = Message(role="user", content=[ContentPart(part_type="text", text="Test")])
        assert msg.id is not None
        assert len(msg.id) > 0

    def test_message_custom_id(self):
        """Test creating Message with custom ID."""
        custom_id = "custom-123"
        msg = Message(
            role="user",
            content=[ContentPart(part_type="text", text="Test")],
            message_id=custom_id
        )
        assert msg.id == custom_id

    def test_message_timestamp(self):
        """Test creating Message with custom timestamp."""
        custom_time = datetime(2026, 4, 4, 12, 0, 0)
        msg = Message(
            role="user",
            content=[ContentPart(part_type="text", text="Test")],
            creation_timestamp=custom_time
        )
        assert msg.creation_timestamp == custom_time

    def test_message_repr(self):
        """Test Message string representation."""
        msg = Message(role="user", content=[ContentPart(part_type="text", text="Test")])
        repr_str = repr(msg)
        assert "Message" in repr_str
        assert "user" in repr_str


class TestChatHistory:
    """Tests for ChatHistory class."""

    def test_history_initialization_empty(self):
        """Test initializing ChatHistory with no arguments."""
        history = ChatHistory()
        assert len(history.messages) == 0
        assert history.generation_config == {}

    def test_history_initialization_with_messages(self):
        """Test initializing ChatHistory with messages."""
        msg = Message(role="user", content=[ContentPart(part_type="text", text="Hello")])
        history = ChatHistory(messages=[msg])
        assert len(history.messages) == 1
        assert history.messages[0] is msg

    def test_history_initialization_with_config(self):
        """Test initializing ChatHistory with generation config."""
        history = ChatHistory(generation_config={"max_tokens": 4096})
        assert history.generation_config["max_tokens"] == 4096

    def test_append_message(self):
        """Test appending a message to history."""
        history = ChatHistory()
        msg = Message(role="user", content=[ContentPart(part_type="text", text="Hello")])
        history.append_message(msg)
        assert len(history.messages) == 1
        assert history.messages[0] is msg

    def test_set_generation_config(self):
        """Test setting generation config parameter."""
        history = ChatHistory()
        history.set_generation_config("max_tokens", 4096)
        history.set_generation_config("temperature", 0.7)
        assert history.generation_config["max_tokens"] == 4096
        assert history.generation_config["temperature"] == 0.7

    def test_get_generation_config(self):
        """Test getting generation config parameter."""
        history = ChatHistory(generation_config={"max_tokens": 4096})
        assert history.get_generation_config("max_tokens") == 4096
        assert history.get_generation_config("nonexistent", "default") == "default"

    def test_get_content(self):
        """Test getting content as list of dictionaries."""
        msg1 = Message(role="user", content=[ContentPart(part_type="text", text="Hi")])
        msg2 = Message(role="assistant", content=[ContentPart(part_type="text", text="Hello")])
        history = ChatHistory(messages=[msg1, msg2])
        contents = history.get_content()
        assert len(contents) == 2
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "assistant"

    def test_get_content_empty(self):
        """Test getting content from empty history."""
        history = ChatHistory()
        assert history.get_content() == []

    def test_len_method(self):
        """Test len() on ChatHistory."""
        history = ChatHistory(messages=[
            Message(role="user", content=[ContentPart(part_type="text", text="1")]),
            Message(role="assistant", content=[ContentPart(part_type="text", text="2")])
        ])
        assert len(history) == 2

    def test_iter_method(self):
        """Test iteration over ChatHistory."""
        history = ChatHistory(messages=[
            Message(role="user", content=[ContentPart(part_type="text", text="1")]),
            Message(role="assistant", content=[ContentPart(part_type="text", text="2")])
        ])
        roles = [msg.role for msg in history]
        assert roles == ["user", "assistant"]

    def test_system_message_position(self):
        """Test that system messages can be placed at position 0."""
        history = ChatHistory(messages=[
            Message(role="system", content=[ContentPart(part_type="text", text="You are helpful")]),
            Message(role="user", content=[ContentPart(part_type="text", text="Hello")])
        ])
        assert history.messages[0].role == "system"
        assert history.messages[1].role == "user"

    def test_history_with_tools_as_messages(self):
        """Test ChatHistory with tool messages."""
        tool_msg = Message(
            role="tool",
            content=[ContentPart(
                part_type="tool",
                name="get_weather",
                description="Get weather",
                parameters={"type": "object"}
            )]
        )
        history = ChatHistory(
            messages=[tool_msg, Message(role="user", content=[ContentPart(part_type="text", text="Hello")])],
            generation_config={"max_tokens": 4096, "tool_choice": {"type": "auto"}}
        )
        assert len(history.messages) == 2
        assert history.messages[0].role == "tool"
        assert history.generation_config["max_tokens"] == 4096
        assert history.generation_config["tool_choice"] == {"type": "auto"}

    def test_history_repr(self):
        """Test ChatHistory string representation."""
        history = ChatHistory(
            messages=[Message(role="user", content=[ContentPart(part_type="text", text="Test")])],
            generation_config={"max_tokens": 4096}
        )
        repr_str = repr(history)
        assert "ChatHistory" in repr_str
        assert "messages=1" in repr_str
        assert "tools" not in repr_str
