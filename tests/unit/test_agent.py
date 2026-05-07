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


class TestAgentInit:
    """Test Agent initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.role_manager = RoleManager()
        self.role_manager.register_role(
            Role(name="test", description="Test role", model=".*")
        )
        self.chatbot_manager = MagicMock()
        self.tool_manager = ToolManager()

    def teardown_method(self):
        """Clean up."""
        pass

    def test_agent_creation(self):
        """Test creating an Agent."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        assert agent.is_running() is False
        assert agent._loop_task is None
        assert len(agent._sessions) == 0
        assert len(agent._message_queues) == 0

    def test_agent_not_running(self):
        """Test agent starts not running."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        assert agent.is_running() is False

    def test_agent_channels_registry(self):
        """Test agent maintains channel registry."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        assert len(agent._channels) == 0


class TestAgentLifecycle:
    """Test Agent start/stop lifecycle."""

    def setup_method(self):
        """Set up test fixtures."""
        self.role_manager = RoleManager()
        self.role_manager.register_role(
            Role(name="test", description="Test role", model=".*")
        )
        self.chatbot_manager = MagicMock()
        self.tool_manager = ToolManager()

    def teardown_method(self):
        """Clean up."""
        pass

    @pytest.mark.asyncio
    async def test_agent_start(self):
        """Test starting the agent."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        await agent.start()

        assert agent.is_running() is True
        assert agent._loop_task is not None
        assert not agent._loop_task.done()

        await agent.stop()

    @pytest.mark.asyncio
    async def test_agent_stop(self):
        """Test stopping the agent."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        await agent.start()
        await agent.stop()

        assert agent.is_running() is False

    @pytest.mark.asyncio
    async def test_agent_start_already_running_raises(self):
        """Test starting an already-running agent raises error."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        await agent.start()

        with pytest.raises(RuntimeError, match="already running"):
            await agent.start()

        await agent.stop()

    @pytest.mark.asyncio
    async def test_agent_stop_not_running_is_noop(self):
        """Test stopping a not-running agent is a no-op."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        await agent.stop()  # should not raise

        assert agent.is_running() is False


class TestAgentSessionManagement:
    """Test Agent session management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.role_manager = RoleManager()
        self.role_manager.register_role(
            Role(name="test", description="Test role", model=".*")
        )
        self.role_manager.register_role(
            Role(name="assistant", description="Assistant role", model=".*")
        )
        self.chatbot_manager = MagicMock()
        self.tool_manager = ToolManager()

    def teardown_method(self):
        """Clean up."""
        pass

    def test_create_session(self):
        """Test creating a session."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        session = agent.create_session("test")

        assert session.role.name == "test"
        assert session.uuid in agent._sessions
        assert session.uuid in agent._message_queues
        assert session.uuid in agent._session_channels

    def test_create_session_invalid_role(self):
        """Test creating session with invalid role raises error."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        with pytest.raises(ValueError, match="not found"):
            agent.create_session("nonexistent")

    def test_get_session(self):
        """Test getting a session by UUID."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        session = agent.create_session("test")
        retrieved = agent.get_session(session.uuid)

        assert retrieved == session

    def test_get_session_not_found(self):
        """Test getting non-existent session returns None."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        session_uuid = uuid.UUID("810fb120-e4e5-4e32-9718-88bbcaf7641a")
        retrieved = agent.get_session(session_uuid)

        assert retrieved is None

    def test_list_sessions(self):
        """Test listing all sessions."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        session1 = agent.create_session("test")
        session2 = agent.create_session("assistant")

        sessions = agent.list_sessions()

        assert len(sessions) == 2
        assert session1.uuid in sessions
        assert session2.uuid in sessions

    def test_destroy_session(self):
        """Test destroying a session."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        session = agent.create_session("test")
        result = agent.destroy_session(session.uuid)

        assert result is True
        assert session.uuid not in agent._sessions
        assert session.uuid not in agent._message_queues
        assert session.uuid not in agent._session_channels

    def test_destroy_session_not_found(self):
        """Test destroying non-existent session returns False."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        session_uuid = uuid.UUID("810fb120-e4e5-4e32-9718-88bbcaf7641a")
        result = agent.destroy_session(session_uuid)

        assert result is False


class TestAgentMessageQueue:
    """Test Agent message queue operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.role_manager = RoleManager()
        self.role_manager.register_role(
            Role(name="test", description="Test role", model=".*")
        )
        self.chatbot_manager = MagicMock()
        self.tool_manager = ToolManager()

    def teardown_method(self):
        """Clean up."""
        pass

    @pytest.mark.asyncio
    async def test_post_message(self):
        """Test posting a message to a session queue."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        session = agent.create_session("test")
        message = Message(role="user", content=[ContentPart(part_type="text", text="Hello")])

        agent.post_message(session.uuid, message)

        assert not agent._message_queues[session.uuid].empty()
        retrieved = agent._message_queues[session.uuid].get_nowait()
        assert retrieved == message

    @pytest.mark.asyncio
    async def test_post_message_session_not_found(self):
        """Test posting message to non-existent session raises KeyError."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        session_uuid = uuid.UUID("810fb120-e4e5-4e32-9718-88bbcaf7641a")
        message = Message(role="user", content=[ContentPart(part_type="text", text="Hello")])

        with pytest.raises(KeyError):
            agent.post_message(session_uuid, message)


class TestAgentNotificationQueues:
    """Test Agent notification queue operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.role_manager = RoleManager()
        self.role_manager.register_role(
            Role(name="test", description="Test role", model=".*")
        )
        self.chatbot_manager = MagicMock()
        self.tool_manager = ToolManager()

    def teardown_method(self):
        """Clean up."""
        pass

    @pytest.mark.asyncio
    async def test_subscribe_notifications(self):
        """Test subscribing to session notifications."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        session = agent.create_session("test")

        async for notification in agent.subscribe_notifications("shell", session.uuid):
            # This should not complete without an active loop
            break

    @pytest.mark.asyncio
    async def test_unsubscribe_notifications(self):
        """Test unsubscribing from session notifications."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        session = agent.create_session("test")

        # Subscribe
        async for _ in agent.subscribe_notifications("shell", session.uuid):
            break

        # Unsubscribe — channel should be removed from session's channels
        agent.unsubscribe_notifications("shell", session.uuid)

    def test_publish_notification_to_channels(self):
        """Test publishing notification to subscribed channels."""
        from peteos.channels import InteractiveShellChannel

        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        session = agent.create_session("test")

        # Create a mock channel and subscribe it to the session
        channel = MagicMock()
        channel.name = "shell"
        agent._session_channels[session.uuid] = {channel}

        # Manually trigger notification publishing
        agent._publish_notification(session.uuid, Message(role="user", content=[ContentPart(part_type="text", text="Test")]))

        # Check push_event was called on the channel
        channel.push_event.assert_called_once()


class TestAgentHookCallbacks:
    """Test Agent hook callback methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.role_manager = RoleManager()
        self.role_manager.register_role(
            Role(name="test", description="Test role", model=".*")
        )
        self.chatbot_manager = MagicMock()
        self.tool_manager = ToolManager()

    def teardown_method(self):
        """Clean up."""
        pass

    def test_on_before_tool_execution(self):
        """Test before_tool_execution hook callback."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        session = agent.create_session("test")
        tool_call = {"name": "test_tool", "arguments": {"param": "value"}}

        result = agent._on_before_tool_execution(session.uuid, tool_call)

        assert result == (True, "")

    def test_on_after_tool_execution(self):
        """Test after_tool_execution hook callback."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        session = agent.create_session("test")
        tool_call = {"name": "test_tool", "arguments": {"param": "value"}}

        # Should not raise
        agent._on_after_tool_execution(session.uuid, tool_call, "result", True)

    def test_on_before_loop_continue(self):
        """Test before_loop_continue hook callback."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        session = agent.create_session("test")
        delta_messages = [
            Message(role="tool", content=[ContentPart(part_type="tool", name="test_tool", description="test", parameters={})]),
            Message(role="assistant", content=[ContentPart(part_type="text", text="Hello")])
        ]

        # Should not raise
        agent._on_before_loop_continue(session.uuid, delta_messages)

    def test_on_before_loop_exit(self):
        """Test before_loop_exit hook callback."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        session = agent.create_session("test")
        # Add a history message
        session.chat_history.append_message(
            Message(role="assistant", content=[ContentPart(part_type="text", text="Final answer")])
        )

        # Should not raise
        agent._on_before_loop_exit(session.uuid, "final_answer")


class TestAgentIntegration:
    """Integration tests for Agent with channels."""

    def setup_method(self):
        """Set up test fixtures."""
        self.role_manager = RoleManager()
        self.role_manager.register_role(
            Role(name="test", description="Test role", model=".*")
        )
        self.chatbot_manager = MagicMock()
        self.tool_manager = ToolManager()

    def teardown_method(self):
        """Clean up."""
        pass

    @pytest.mark.asyncio
    async def test_agent_creates_session_hooks(self):
        """Test that creating a session registers hooks."""
        agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

        session = agent.create_session("test")

        # Check hooks are registered on execution environment
        env = session.execution_environment
        assert "before_tool_execution" in env._hooks
        assert "after_tool_execution" in env._hooks
        assert "before_loop_continue" in env._hooks
        assert "before_loop_exit" in env._hooks
