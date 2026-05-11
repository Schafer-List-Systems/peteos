"""Unit tests for Agent queue-based architecture."""

import asyncio
import uuid
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from peteos.agent import Agent
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.chatbot import Message, ContentPart
from peteos.toolmanager import ToolManager


@pytest.fixture
def agent():
    """Create a fresh Agent for each test."""
    role_manager = RoleManager()
    role_manager.register_role(
        Role(name="test", description="Test role", model=".*")
    )
    chatbot_manager = MagicMock()
    tool_manager = ToolManager()
    return Agent(role_manager, chatbot_manager, tool_manager)


@pytest.fixture
def agent_with_sessions():
    """Create a fresh Agent, auto-cleanup."""
    role_manager = RoleManager()
    role_manager.register_role(
        Role(name="test", description="Test role", model=".*")
    )
    role_manager.register_role(
        Role(name="assistant", description="Assistant role", model=".*")
    )
    chatbot_manager = MagicMock()
    tool_manager = ToolManager()
    ag = Agent(role_manager, chatbot_manager, tool_manager)
    yield ag
    # Cleanup: stop all sessions
    for session_uuid in list(ag._sessions.keys()):
        session = ag.get_session(session_uuid)
        if session and session.is_running():
            try:
                asyncio.get_event_loop().run_until_complete(session.stop())
            except RuntimeError:
                pass


def test_agent_creation(agent):
    """Test creating an Agent."""
    assert len(agent._sessions) == 0
    assert len(agent._channels) == 0


def test_agent_channels_registry(agent):
    """Test agent maintains channel registry."""
    assert len(agent._channels) == 0


def test_get_session_not_found(agent):
    """Test getting non-existent session returns None."""
    session_uuid = uuid.UUID("810fb120-e4e5-4e32-9718-88bbcaf7641a")
    retrieved = agent.get_session(session_uuid)
    assert retrieved is None


@pytest.mark.asyncio
async def test_create_session(agent_with_sessions):
    """Test creating a session."""
    session = await agent_with_sessions.create_session("test")

    assert session.role.name == "test"
    assert session.uuid in agent_with_sessions._sessions
    assert session.is_running()


@pytest.mark.asyncio
async def test_create_session_invalid_role(agent):
    """Test creating session with invalid role raises error."""
    with pytest.raises(ValueError, match="not found"):
        await agent.create_session("nonexistent")


@pytest.mark.asyncio
async def test_get_session(agent_with_sessions):
    """Test getting a session by UUID."""
    session = await agent_with_sessions.create_session("test")
    retrieved = agent_with_sessions.get_session(session.uuid)
    assert retrieved == session


@pytest.mark.asyncio
async def test_list_sessions(agent_with_sessions):
    """Test listing all sessions."""
    session1 = await agent_with_sessions.create_session("test")
    session2 = await agent_with_sessions.create_session("assistant")

    sessions = agent_with_sessions.list_sessions()
    assert len(sessions) == 2
    assert session1.uuid in sessions
    assert session2.uuid in sessions


@pytest.mark.asyncio
async def test_destroy_session(agent_with_sessions):
    """Test destroying a session."""
    session = await agent_with_sessions.create_session("test")
    result = await agent_with_sessions.destroy_session(session.uuid)

    assert result is True
    assert session.uuid not in agent_with_sessions._sessions


@pytest.mark.asyncio
async def test_destroy_session_not_found(agent_with_sessions):
    """Test destroying non-existent session returns False."""
    session_uuid = uuid.UUID("810fb120-e4e5-4e32-9718-88bbcaf7641a")
    result = await agent_with_sessions.destroy_session(session_uuid)
    assert result is False


def test_publish_notification_to_channels(agent_with_sessions):
    """Test publishing notification to subscribed channels."""
    session = asyncio.get_event_loop().run_until_complete(
        agent_with_sessions.create_session("test")
    )

    channel = MagicMock()
    channel.name = "shell"
    agent_with_sessions._session_channels[session.uuid] = {channel}

    agent_with_sessions._publish_notification(
        session.uuid, Message(role="user", content=[ContentPart(part_type="text", text="Test")])
    )

    channel.push_event.assert_called_once()


@pytest.mark.asyncio
async def test_on_before_tool_execution(agent_with_sessions):
    """Test before_tool_execution hook callback."""
    session = await agent_with_sessions.create_session("test")
    tool_call = {"name": "test_tool", "arguments": {"param": "value"}}

    result = agent_with_sessions._on_before_tool_execution(session.uuid, tool_call)
    assert result == ("pending", None)


@pytest.mark.asyncio
async def test_on_after_tool_execution(agent_with_sessions):
    """Test after_tool_execution hook callback."""
    session = await agent_with_sessions.create_session("test")
    tool_call = {"name": "test_tool", "arguments": {"param": "value"}}

    agent_with_sessions._on_after_tool_execution(session.uuid, tool_call, "result", True)


@pytest.mark.asyncio
async def test_on_before_loop_continue(agent_with_sessions):
    """Test before_loop_continue hook callback."""
    session = await agent_with_sessions.create_session("test")
    delta_messages = [
        Message(role="tool", content=[ContentPart(
            part_type="tool", name="test_tool", description="test", parameters={}
        )]),
        Message(role="assistant", content=[ContentPart(part_type="text", text="Hello")])
    ]

    agent_with_sessions._on_before_loop_continue(session.uuid, delta_messages)


@pytest.mark.asyncio
async def test_on_before_loop_exit(agent_with_sessions):
    """Test before_loop_exit hook callback."""
    session = await agent_with_sessions.create_session("test")
    session.chat_history.append_message(
        Message(role="assistant", content=[ContentPart(part_type="text", text="Final answer")])
    )

    agent_with_sessions._on_before_loop_exit(session.uuid, "final_answer")


@pytest.mark.asyncio
async def test_agent_creates_session_hooks(agent_with_sessions):
    """Test that creating a session registers hooks."""
    session = await agent_with_sessions.create_session("test")

    env = session.execution_environment
    assert "before_tool_execution" in env._hooks
    assert "after_tool_execution" in env._hooks
    assert "before_loop_continue" in env._hooks
    assert "before_loop_exit" in env._hooks
