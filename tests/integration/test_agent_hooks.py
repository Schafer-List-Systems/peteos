"""Integration tests for Agent hook callbacks with new Message format.

Tests verify that hook callbacks correctly handle the new Message format
where msg.role is a string and msg.content is List[ContentPart].
"""

import pytest
import pytest_asyncio
from unittest.mock import MagicMock
import asyncio

from peteos.agent import Agent
from peteos.chatbot.manager import ChatBotManager, BackendInfo
from peteos.chatbot import Message, ContentPart, OpenAIChatBot
from peteos.chatbot.httpclient import HTTPClient
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager


class TestAgentHooksMessageFormat:
    """Tests for Agent hook callbacks with Message format."""

    @pytest.mark.asyncio
    async def test_on_before_loop_continue_handles_tool_result_message(self):
        """Test that _on_before_loop_continue correctly processes tool_result messages."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )
        chatbot_manager = ChatBotManager()
        mock_chatbot = MagicMock(spec=OpenAIChatBot)
        backend_info = BackendInfo(
            name="test",
            url="http://test",
            api_type="openai",
            models={"test-model": mock_chatbot}
        )
        chatbot_manager._backends["test"] = backend_info
        tool_manager = ToolManager()

        agent = Agent(role_manager, chatbot_manager, tool_manager)
        await agent.start()

        # Create session
        session = agent.create_session("test")

        # Create a tool_result message with new Message format
        delta_message = Message(
            role="tool_result",
            content=[
                ContentPart(
                    part_type="tool_result",
                    name="test_tool",
                    content="tool output",
                    success=True
                )
            ]
        )

        # This should not raise AttributeError
        try:
            agent._on_before_loop_continue(session.uuid, [delta_message])
            # If we got here, the test passed
        except AttributeError as e:
            pytest.fail(f"_on_before_loop_continue raised AttributeError: {e}")

        await agent.stop()

    @pytest.mark.asyncio
    async def test_on_before_loop_continue_handles_assistant_message(self):
        """Test that _on_before_loop_continue correctly processes assistant messages."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )
        chatbot_manager = ChatBotManager()
        mock_chatbot = MagicMock(spec=OpenAIChatBot)
        backend_info = BackendInfo(
            name="test",
            url="http://test",
            api_type="openai",
            models={"test-model": mock_chatbot}
        )
        chatbot_manager._backends["test"] = backend_info
        tool_manager = ToolManager()

        agent = Agent(role_manager, chatbot_manager, tool_manager)
        await agent.start()

        # Create session
        session = agent.create_session("test")

        # Create an assistant message with new Message format
        delta_message = Message(
            role="assistant",
            content=[ContentPart(part_type="text", text="Hello")]
        )

        # This should not raise AttributeError
        try:
            agent._on_before_loop_continue(session.uuid, [delta_message])
            # If we got here, the test passed
        except AttributeError as e:
            pytest.fail(f"_on_before_loop_continue raised AttributeError: {e}")

        await agent.stop()

    @pytest.mark.asyncio
    async def test_on_before_loop_exit_handles_assistant_message(self):
        """Test that _on_before_loop_exit correctly processes assistant messages."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )
        chatbot_manager = ChatBotManager()
        mock_chatbot = MagicMock(spec=OpenAIChatBot)
        backend_info = BackendInfo(
            name="test",
            url="http://test",
            api_type="openai",
            models={"test-model": mock_chatbot}
        )
        chatbot_manager._backends["test"] = backend_info
        tool_manager = ToolManager()

        agent = Agent(role_manager, chatbot_manager, tool_manager)
        await agent.start()

        # Create session
        session = agent.create_session("test")

        # Add assistant message to session history
        assistant_message = Message(
            role="assistant",
            content=[ContentPart(part_type="text", text="Final response")]
        )
        session.chat_history.append_message(assistant_message)

        # This should not raise AttributeError
        try:
            agent._on_before_loop_exit(session.uuid, "final_answer")
            # If we got here, the test passed
        except AttributeError as e:
            pytest.fail(f"_on_before_loop_exit raised AttributeError: {e}")

        await agent.stop()

    @pytest.mark.asyncio
    async def test_hook_notifications_with_new_message_format(self):
        """Test that hook notifications work with the new Message format."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )
        chatbot_manager = ChatBotManager()
        mock_chatbot = MagicMock(spec=OpenAIChatBot)
        backend_info = BackendInfo(
            name="test",
            url="http://test",
            api_type="openai",
            models={"test-model": mock_chatbot}
        )
        chatbot_manager._backends["test"] = backend_info
        tool_manager = ToolManager()

        agent = Agent(role_manager, chatbot_manager, tool_manager)
        await agent.start()

        # Create session and shell channel
        session = agent.create_session("test")
        from peteos.channels import InteractiveShellChannel
        shell = InteractiveShellChannel("shell", agent)
        shell.select_session(session.uuid)

        # subscribe_to_session returns early when channel is not running,
        # so manually create the notification queue for this test
        queue_key = ("shell", session.uuid)
        agent._notification_queues[queue_key] = asyncio.Queue()
        if session.uuid not in agent._session_channels:
            agent._session_channels[session.uuid] = set()
        agent._session_channels[session.uuid].add(shell)

        # Trigger after_tool_execution hook — it publishes tool_result notifications
        agent._on_after_tool_execution(
            session.uuid,
            {"name": "weather_tool", "arguments": {}},
            "Sunny and 25C",
            True
        )

        # Check that notification was queued
        assert queue_key in agent._notification_queues
        assert agent._notification_queues[queue_key].qsize() > 0

        notification = agent._notification_queues[queue_key].get_nowait()
        assert notification.role == "tool_result"
        assert len(notification.content) > 0
        assert "Sunny and 25C" in notification.content[0].data.get("content", "")
        assert notification.metadata.get("tool_status") == "ok"

        await agent.stop()
