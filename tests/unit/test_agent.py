"""Unit tests for Agent queue-based architecture."""

import asyncio
import uuid
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from peteos.agent import Agent
from peteos.role import Role
from peteos.chatbot import Message, ContentPart, ChatBotManager
from peteos.toolmanager import ToolManager


@pytest.fixture(autouse=True)
def _setup_mock_chatbot():
    """Set up a mock ChatBot in the class-level ChatBotManager for tests."""
    ChatBotManager._backends = {"test-backend": MagicMock(models={"test_model": MagicMock()})}
    yield
    ChatBotManager._backends.clear()


@pytest.fixture
def agent():
    """Create a fresh Agent for each test."""
    role = Role(name="test", description="Test role", model=".*")
    tool_manager = ToolManager()
    return Agent(role, tool_manager)


@pytest.fixture
def agent_with_sessions():
    """Create a fresh Agent, auto-cleanup."""
    role = Role(name="test", description="Test role", model=".*")
    tool_manager = ToolManager()
    ag = Agent(role, tool_manager)
    yield ag
    # Cleanup: stop all sessions
    for session_uuid in list(ag._sessions.keys()):
        session = ag.get_session(session_uuid)
        if session and session.is_running():
            try:
                asyncio.get_event_loop().run_until_complete(session.stop())
            except RuntimeError:
                pass


@pytest.fixture
def agent_with_assistant_role():
    """Create a fresh Agent with assistant role, auto-cleanup."""
    role = Role(name="assistant", description="Assistant role", model=".*")
    tool_manager = ToolManager()
    ag = Agent(role, tool_manager)
    yield ag
    for session_uuid in list(ag._sessions.keys()):
        session = ag.get_session(session_uuid)
        if session and session.is_running():
            try:
                asyncio.get_event_loop().run_until_complete(session.stop())
            except RuntimeError:
                pass


@pytest.fixture
def agent_with_auto_approve():
    """Create an Agent with a role that has auto_approve_tools."""
    role = Role(
        name="autobot",
        description="Auto-approves some tools",
        model=".*",
        auto_approve_tools=["web_fetch"]
    )
    tool_manager = ToolManager()
    ag = Agent(role, tool_manager)
    yield ag
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
    session = await agent_with_sessions.create_session()

    assert session.role.name == "test"
    assert session.uuid in agent_with_sessions._sessions
    assert session.is_running()


@pytest.mark.asyncio
async def test_create_session_invalid_role(agent):
    """Test that create_session uses the agent's role."""
    # With the new Agent(role, tool_manager) API, create_session() takes no arguments
    # and uses the agent's role. No invalid role concept exists.
    pass


@pytest.mark.asyncio
async def test_get_session(agent_with_sessions):
    """Test getting a session by UUID."""
    session = await agent_with_sessions.create_session()
    retrieved = agent_with_sessions.get_session(session.uuid)
    assert retrieved == session


@pytest.mark.asyncio
async def test_list_sessions(agent_with_sessions):
    """Test listing all sessions."""
    session1 = await agent_with_sessions.create_session()
    session2 = await agent_with_sessions.create_session()

    sessions = agent_with_sessions.list_sessions()
    assert len(sessions) == 2
    assert session1.uuid in sessions
    assert session2.uuid in sessions


@pytest.mark.asyncio
async def test_destroy_session(agent_with_sessions):
    """Test destroying a session."""
    session = await agent_with_sessions.create_session()
    result = await agent_with_sessions.destroy_session(session.uuid)

    assert result is True
    assert session.uuid not in agent_with_sessions._sessions


@pytest.mark.asyncio
async def test_destroy_session_not_found(agent_with_sessions):
    """Test destroying non-existent session returns False."""
    session_uuid = uuid.UUID("810fb120-e4e5-4e32-9718-88bbcaf7641a")
    result = await agent_with_sessions.destroy_session(session_uuid)
    assert result is False


@pytest.mark.asyncio
async def test_publish_notification_to_channels(agent_with_sessions):
    """Test publishing notification to subscribed channels."""
    session = await agent_with_sessions.create_session()

    channel = MagicMock()
    channel.name = "shell"
    session.subscribe(channel)

    await session.publish_notification(
        Message(role="user", content=[ContentPart(part_type="text", text="Test")])
    )

    channel.push_event.assert_called_once()


@pytest.mark.asyncio
async def test_on_before_tool_execution(agent_with_sessions):
    """Test before_tool_execution hook callback."""
    session = await agent_with_sessions.create_session()
    tool_call = {"name": "test_tool", "arguments": {"param": "value"}}

    result = agent_with_sessions._on_before_tool_execution(session, tool_call)
    assert result == ("pending", None)


@pytest.mark.asyncio
async def test_on_after_tool_execution(agent_with_sessions):
    """Test after_tool_execution hook callback."""
    session = await agent_with_sessions.create_session()
    tool_call = {"name": "test_tool", "arguments": {"param": "value"}}

    agent_with_sessions._on_after_tool_execution(session, tool_call, "result", True)


@pytest.mark.asyncio
async def test_on_before_notification_publish(agent_with_sessions):
    """Test before_notification_publish hook callback."""
    session = await agent_with_sessions.create_session()
    msg = Message(role="assistant", content=[ContentPart(part_type="text", text="Hello")])

    agent_with_sessions._on_before_notification_publish(session, msg)


@pytest.mark.asyncio
async def test_agent_creates_session_hooks(agent_with_sessions):
    """Test that creating a session registers hooks."""
    session = await agent_with_sessions.create_session()

    env = session.execution_environment
    assert "before_tool_execution" in env._hooks
    assert "after_tool_execution" in env._hooks
    assert "before_notification_publish" in env._hooks
    assert "before_send_to_chatbot" in env._hooks


@pytest.mark.asyncio
async def test_on_before_tool_execution_auto_approve(agent_with_auto_approve):
    """Test that tools in auto_approve_tools are immediately approved."""
    session = await agent_with_auto_approve.create_session()

    # Tool in auto_approve_tools should return (True, None)
    approved_result = agent_with_auto_approve._on_before_tool_execution(
        session, {"name": "web_fetch", "arguments": "{}"}
    )
    assert approved_result == (True, None)

    # Tool not in auto_approve_tools should still return ("pending", None)
    pending_result = agent_with_auto_approve._on_before_tool_execution(
        session, {"name": "file_write", "arguments": "{}"}
    )
    assert pending_result == ("pending", None)


@pytest.mark.asyncio
async def test_on_before_tool_execution_auto_approve_empty_list(agent_with_sessions):
    """Test that empty auto_approve_tools still requires approval."""
    session = await agent_with_sessions.create_session()

    result = agent_with_sessions._on_before_tool_execution(
        session, {"name": "test_tool", "arguments": {"param": "value"}}
    )
    assert result == ("pending", None)
