"""Tests for Session class."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from peteos.chatbot import ChatHistory
from peteos.chatbot import Message, ContentPart
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.session import Session
from peteos.toolmanager import ToolManager, Tool
from peteos.replexecutionenvironment import REPLExecutionEnvironment


def test_session_init():
    """Test Session initialization."""
    chatbot_manager = MagicMock()
    tool_manager = ToolManager()
    role = Role(name="test", description="A test role")

    session = Session(
        role=role,
        tool_manager=tool_manager,
        chatbot_manager=chatbot_manager
    )

    assert session.uuid is not None
    assert session.role == role
    assert session.chat_history is not None
    assert session.chatbot_manager == chatbot_manager
    assert isinstance(session.execution_environment, REPLExecutionEnvironment)
    assert session.event_queue is not None
    assert session._event_trigger is not None


def test_session_init_with_custom_uuid():
    """Test Session initialization with custom UUID."""
    import uuid
    chatbot_manager = MagicMock()
    tool_manager = ToolManager()
    role = Role(name="test", description="A test role")
    custom_uuid = uuid.uuid4()

    session = Session(
        role=role,
        tool_manager=tool_manager,
        chatbot_manager=chatbot_manager,
        session_uuid=custom_uuid
    )

    assert session.uuid == custom_uuid


def test_session_init_with_custom_chat_history():
    """Test Session initialization with custom ChatHistory."""
    chatbot_manager = MagicMock()
    tool_manager = ToolManager()
    role = Role(name="test", description="A test role")
    chat_history = ChatHistory()
    chat_history.append_message(Message(role="user", content=[ContentPart(part_type="text", text="Hi")]))

    session = Session(
        role=role,
        tool_manager=tool_manager,
        chatbot_manager=chatbot_manager,
        chat_history=chat_history
    )

    assert session.chat_history is chat_history
    assert len(session.chat_history.messages) == 1


def test_session_load_from_json():
    """Test loading Session from JSON dict."""
    chatbot_manager = MagicMock()
    role_manager = RoleManager()
    tool_manager = ToolManager()

    # Register a role
    role = Role(name="test", description="A test role")
    role_manager.register_role(role)

    # Register a required tool
    def dummy_tool():
        return "result"

    tool_manager.register_tool(tool=Tool.from_callable(dummy_tool))

    # Create session data
    session_data = {
        "uuid": "810fb120-e4e5-4e32-9718-88bbcaf7641a",
        "role": "test",
        "chat_history": [
            {
                "content": [{"type": "text", "text": "Hello"}],
                "role": "user",
                "creation_timestamp": "2026-03-31T12:00:00",
                "id": "msg-1"
            },
            {
                "content": [{"type": "text", "text": "Hi!"}],
                "role": "assistant",
                "creation_timestamp": "2026-03-31T12:00:01",
                "id": "msg-2"
            }
        ]
    }

    session = Session.load_from_json(
        session_data,
        chatbot_manager,
        role_manager,
        tool_manager
    )

    assert session.uuid.hex == "810fb120e4e54e32971888bbcaf7641a"
    assert session.role.name == "test"
    assert len(session.chat_history.messages) == 2
    assert session.chat_history.messages[0].get_role() == "user"
    assert session.chat_history.messages[1].text == "Hi!"


def test_session_load_from_json_without_uuid():
    """Test loading Session from JSON without UUID."""
    chatbot_manager = MagicMock()
    role_manager = RoleManager()
    tool_manager = ToolManager()

    role = Role(name="test", description="A test role")
    role_manager.register_role(role)

    session_data = {
        "role": "test",
        "chat_history": []
    }

    session = Session.load_from_json(
        session_data,
        chatbot_manager,
        role_manager,
        tool_manager
    )

    assert session.uuid is not None


def test_session_load_from_json_missing_role():
    """Test loading Session fails when role not found."""
    chatbot_manager = MagicMock()
    role_manager = RoleManager()
    tool_manager = ToolManager()

    session_data = {
        "uuid": "810fb120-e4e5-4e32-9718-88bbcaf7641a",
        "role": "nonexistent",
        "chat_history": []
    }

    with pytest.raises(ValueError, match="not found in RoleManager"):
        Session.load_from_json(
            session_data,
            chatbot_manager,
            role_manager,
            tool_manager
        )


def test_session_load_from_json_missing_required_tool():
    """Test loading Session fails when required tool is missing."""
    chatbot_manager = MagicMock()
    role_manager = RoleManager()
    tool_manager = ToolManager()

    role = Role(
        name="test",
        description="A test role",
        required_tools=["missing_tool"]
    )
    role_manager.register_role(role)

    session_data = {
        "uuid": "810fb120-e4e5-4e32-9718-88bbcaf7641a",
        "role": "test",
        "chat_history": []
    }

    with pytest.raises(ValueError, match="requires tool"):
        Session.load_from_json(
            session_data,
            chatbot_manager,
            role_manager,
            tool_manager
        )


def test_session_load_from_json_without_chat_history():
    """Test loading Session without chat_history in JSON."""
    chatbot_manager = MagicMock()
    role_manager = RoleManager()
    tool_manager = ToolManager()

    role = Role(name="test", description="A test role")
    role_manager.register_role(role)

    session_data = {
        "uuid": "810fb120-e4e5-4e32-9718-88bbcaf7641a",
        "role": "test"
    }

    session = Session.load_from_json(
        session_data,
        chatbot_manager,
        role_manager,
        tool_manager
    )

    assert session.chat_history is not None
    assert len(session.chat_history.messages) == 0


def test_session_load_from_file():
    """Test loading Session from JSON file."""
    chatbot_manager = MagicMock()
    role_manager = RoleManager()
    tool_manager = ToolManager()

    role = Role(name="test", description="A test role")
    role_manager.register_role(role)

    def dummy_tool():
        return "result"

    tool_manager.register_tool(tool=Tool.from_callable(dummy_tool))

    with tempfile.TemporaryDirectory() as tmpdir:
        session_file = Path(tmpdir) / "session.json"
        session_data = {
            "uuid": "810fb120-e4e5-4e32-9718-88bbcaf7641a",
            "role": "test",
            "chat_history": [
                {
                    "content": [{"type": "text", "text": "Hello from file"}],
                    "role": "user",
                    "creation_timestamp": "2026-03-31T12:00:00",
                    "id": "msg-file-1"
                }
            ]
        }

        session_file.write_text(json.dumps(session_data))

        session = Session.load_from_file(
            str(session_file),
            chatbot_manager,
            role_manager,
            tool_manager
        )

        assert session.uuid.hex == "810fb120e4e54e32971888bbcaf7641a"
        assert session.role.name == "test"
        assert len(session.chat_history.messages) == 1
        assert session.chat_history.messages[0].get_role() == "user"


def test_session_load_from_file_without_timestamps():
    """Test loading Session from file without timestamps."""
    chatbot_manager = MagicMock()
    role_manager = RoleManager()
    tool_manager = ToolManager()

    role = Role(name="test", description="A test role")
    role_manager.register_role(role)

    with tempfile.TemporaryDirectory() as tmpdir:
        session_file = Path(tmpdir) / "session.json"
        session_data = {
            "role": "test",
            "chat_history": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "No timestamp"}],
                    "id": "msg-no-ts"
                }
            ]
        }

        session_file.write_text(json.dumps(session_data))

        session = Session.load_from_file(
            str(session_file),
            chatbot_manager,
            role_manager,
            tool_manager
        )

        assert len(session.chat_history.messages) == 1
        # Message should have default timestamp
        assert session.chat_history.messages[0].creation_timestamp is not None


def test_session_load_from_file_no_uuid():
    """Test loading Session from file without UUID."""
    chatbot_manager = MagicMock()
    role_manager = RoleManager()
    tool_manager = ToolManager()

    role = Role(name="test", description="A test role")
    role_manager.register_role(role)

    with tempfile.TemporaryDirectory() as tmpdir:
        session_file = Path(tmpdir) / "session.json"
        session_data = {
            "role": "test",
            "chat_history": []
        }

        session_file.write_text(json.dumps(session_data))

        session = Session.load_from_file(
            str(session_file),
            chatbot_manager,
            role_manager,
            tool_manager
        )

        assert session.uuid is not None


def test_session_load_from_json_empty_chat_history():
    """Test loading Session with empty chat_history list."""
    chatbot_manager = MagicMock()
    role_manager = RoleManager()
    tool_manager = ToolManager()

    role = Role(name="test", description="A test role")
    role_manager.register_role(role)

    session_data = {
        "uuid": "810fb120-e4e5-4e32-9718-88bbcaf7641a",
        "role": "test",
        "chat_history": []
    }

    session = Session.load_from_json(
        session_data,
        chatbot_manager,
        role_manager,
        tool_manager
    )

    assert session.chat_history is not None
    assert len(session.chat_history.messages) == 0
