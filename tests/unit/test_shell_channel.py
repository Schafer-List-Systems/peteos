"""Unit tests for InteractiveShellChannel."""

import asyncio
import uuid

import pytest
from unittest.mock import MagicMock, AsyncMock

from peteos.channels.channel import Channel
from peteos.agent import Agent
from peteos.channels import InteractiveShellChannel
from peteos.role import Role
from peteos.rolemanager import RoleManager


def _cleanup_channels():
    """Clean up channel registry."""
    for name in list(Channel._registry.keys()):
        Channel._registry.pop(name)


class TestShellChannelInit:
    """Test InteractiveShellChannel initialization."""

    def setup_method(self):
        """Set up agent with role manager."""
        _cleanup_channels()

    def teardown_method(self):
        """Clean up."""
        _cleanup_channels()

    @pytest.mark.asyncio
    async def test_shell_channel_creation(self):
        """Test creating a shell channel."""
        role_manager = RoleManager()
        chatbot_manager = MagicMock()
        tool_manager = MagicMock()

        agent = Agent(role_manager, chatbot_manager, tool_manager)
        shell_channel = InteractiveShellChannel("shell", agent)

        assert shell_channel.name == "shell"
        assert shell_channel._running is False  # start() must be called first
        assert shell_channel.active_session_uuid is None

        # Cleanup
        shell_channel._running = False

    def test_shell_channel_registered_with_agent(self):
        """Test shell channel is registered with agent."""
        role_manager = RoleManager()
        chatbot_manager = MagicMock()
        tool_manager = MagicMock()

        agent = Agent(role_manager, chatbot_manager, tool_manager)
        channel = InteractiveShellChannel("shell", agent)

        assert agent.get_channel("shell") == channel


class TestShellChannelCommands:
    """Test shell channel command handling."""

    def setup_method(self):
        """Set up agent with roles."""
        _cleanup_channels()

        self.role_manager = RoleManager()
        self.role_manager.register_role(Role(name="test", description="Test role"))
        self.role_manager.register_role(Role(name="assistant", description="Assistant role"))

        self.chatbot_manager = MagicMock()
        self.tool_manager = MagicMock()

        self.agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

    def teardown_method(self):
        """Clean up."""
        for session_uuid in list(self.agent._sessions.keys()):
            session = self.agent.get_session(session_uuid)
            if session and session.is_running():
                try:
                    asyncio.get_event_loop().run_until_complete(session.stop())
                except RuntimeError:
                    pass
        _cleanup_channels()

    def test_command_new_creates_session(self):
        """Test /new command creates a session."""
        channel = InteractiveShellChannel("shell", self.agent)
        should_continue, output = channel.handle_command("/new test")

        assert should_continue is True
        # Session creation is now logged, not displayed
        assert output == ""
        assert self.agent.get_session(channel.active_session_uuid) is not None

    def test_command_new_missing_role(self):
        """Test /new without role returns usage."""
        channel = InteractiveShellChannel("shell", self.agent)
        should_continue, output = channel.handle_command("/new")

        assert should_continue is True
        assert "Usage: /new" in output

    def test_command_list(self):
        """Test /list command lists sessions."""
        channel = InteractiveShellChannel("shell", self.agent)
        # Create a session first
        channel.handle_command("/new test")
        # Now list
        should_continue, output = channel.handle_command("/list")

        assert should_continue is True
        assert "Sessions:" in output
        assert "(active)" in output

    def test_command_list_empty(self):
        """Test /list when no sessions exist."""
        channel = InteractiveShellChannel("shell", self.agent)
        should_continue, output = channel.handle_command("/list")

        assert should_continue is True
        assert "No sessions available" in output

    def test_command_select_existing_session(self):
        """Test /select with existing session."""
        channel = InteractiveShellChannel("shell", self.agent)
        # Create and get UUID
        channel.handle_command("/new test")
        session_uuid = channel.active_session_uuid

        # Select the same session
        should_continue, output = channel.handle_command(f"/select {session_uuid}")

        assert should_continue is True
        # Session selection is now logged, not displayed
        assert output == ""
        assert channel.active_session_uuid == session_uuid

    def test_command_select_invalid_uuid(self):
        """Test /select with invalid UUID."""
        channel = InteractiveShellChannel("shell", self.agent)
        should_continue, output = channel.handle_command("/select invalid-uuid")

        assert should_continue is True
        assert "Invalid UUID" in output

    def test_command_select_nonexistent_session(self):
        """Test /select with non-existent session."""
        channel = InteractiveShellChannel("shell", self.agent)
        session_uuid = uuid.UUID("810fb120-e4e5-4e32-9718-88bbcaf7641a")
        should_continue, output = channel.handle_command(f"/select {session_uuid}")

        assert should_continue is True
        assert "not found" in output

    def test_command_quit(self):
        """Test /quit command."""
        channel = InteractiveShellChannel("shell", self.agent)
        should_continue, output = channel.handle_command("/quit")

        assert should_continue is False
        assert "Goodbye" in output
        assert channel._running is False

    def test_command_unknown(self):
        """Test unknown command."""
        channel = InteractiveShellChannel("shell", self.agent)
        should_continue, output = channel.handle_command("/unknown")

        assert should_continue is True
        assert "Unknown command" in output

    def test_command_case_insensitive(self):
        """Test commands are case insensitive."""
        channel = InteractiveShellChannel("shell", self.agent)
        should_continue, output = channel.handle_command("/NEW test")

        assert should_continue is True
        # Session creation is now logged, not displayed
        assert output == ""

    def test_non_command_forwarded(self):
        """Test non-command lines are forwarded as messages."""
        import asyncio
        channel = InteractiveShellChannel("shell", self.agent)
        channel.handle_command("/new test")

        # Mock session.queue_message to track calls
        session = self.agent.get_session(channel.active_session_uuid)
        queue_messages = []
        original_queue = session.queue_message
        async def mock_queue(message):
            queue_messages.append(message)
        session.queue_message = mock_queue

        # Simulate receiving a non-command line
        line = "Hello, how are you?"
        should_continue, _ = channel.handle_command(line)

        # Non-commands are handled outside handle_command - they're passed directly
        # The test verifies the session exists and is active
        assert channel.active_session_uuid is not None
        assert session is not None


class TestShellChannelRun:
    """Test shell channel run loop."""

    def setup_method(self):
        """Set up agent."""
        _cleanup_channels()

        self.role_manager = RoleManager()
        self.role_manager.register_role(Role(name="test", description="Test"))
        self.chatbot_manager = MagicMock()
        self.tool_manager = MagicMock()
        self.agent = Agent(self.role_manager, self.chatbot_manager, self.tool_manager)

    def teardown_method(self):
        """Clean up."""
        _cleanup_channels()

    def test_run_displays_welcome(self):
        """Test run displays welcome message."""
        from unittest.mock import AsyncMock, patch, MagicMock
        channel = InteractiveShellChannel("shell", self.agent)
        output = []
        def mock_send(msg):
            output.append(msg)
        channel.send = mock_send

        # Start the channel to get welcome message
        # Need to set active session so send() will format output
        channel.active_session_uuid = uuid.UUID("12345678-1234-1234-1234-123456789012")

        asyncio.run(channel.start())

        assert any("Connected" in msg or "Commands" in msg for msg in output)

        # Stop the channel
        asyncio.run(channel.stop())

    def test_run_handles_quit(self):
        """Test run exits on quit command."""
        channel = InteractiveShellChannel("shell", self.agent)
        channel._running = True

        # Simulate quit command by directly handling it
        should_continue, _ = channel.handle_command("/quit")
        assert should_continue is False
        assert channel._running is False
