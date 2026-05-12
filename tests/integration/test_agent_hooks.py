"""Integration tests for Agent hook callbacks with new Message format.

Tests verify that hook callbacks correctly handle the new Message format
where msg.role is a string and msg.content is List[ContentPart].
"""

import pytest
from unittest.mock import MagicMock

from peteos.agent import Agent
from peteos.chatbot import Message, ContentPart
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
        chatbot_manager = MagicMock()
        tool_manager = ToolManager()

        agent = Agent(role_manager, chatbot_manager, tool_manager)

        # Create session
        session = await agent.create_session("test")

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
        except AttributeError as e:
            pytest.fail(f"_on_before_loop_continue raised AttributeError: {e}")

        await agent.destroy_session(session.uuid)

    @pytest.mark.asyncio
    async def test_on_before_loop_continue_handles_assistant_message(self):
        """Test that _on_before_loop_continue correctly processes assistant messages."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )
        chatbot_manager = MagicMock()
        tool_manager = ToolManager()

        agent = Agent(role_manager, chatbot_manager, tool_manager)

        session = await agent.create_session("test")

        delta_message = Message(
            role="assistant",
            content=[ContentPart(part_type="text", text="Hello")]
        )

        try:
            agent._on_before_loop_continue(session.uuid, [delta_message])
        except AttributeError as e:
            pytest.fail(f"_on_before_loop_continue raised AttributeError: {e}")

        await agent.destroy_session(session.uuid)

    @pytest.mark.asyncio
    async def test_on_before_loop_exit_handles_assistant_message(self):
        """Test that _on_before_loop_exit correctly processes assistant messages."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )
        chatbot_manager = MagicMock()
        tool_manager = ToolManager()

        agent = Agent(role_manager, chatbot_manager, tool_manager)

        session = await agent.create_session("test")

        assistant_message = Message(
            role="assistant",
            content=[ContentPart(part_type="text", text="Final response")]
        )
        session.chat_history.append_message(assistant_message)

        try:
            agent._on_before_loop_exit(session.uuid, "final_answer")
        except AttributeError as e:
            pytest.fail(f"_on_before_loop_exit raised AttributeError: {e}")

        await agent.destroy_session(session.uuid)

    @pytest.mark.asyncio
    async def test_hook_notifications_with_new_message_format(self):
        """Test that hook notifications work with the new Message format."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )
        chatbot_manager = MagicMock()
        tool_manager = ToolManager()

        agent = Agent(role_manager, chatbot_manager, tool_manager)

        session = await agent.create_session("test")
        from peteos.channels import InteractiveShellChannel
        shell = InteractiveShellChannel("shell", agent)
        shell.select_session(session.uuid)

        agent._session_channels[session.uuid] = {shell}

        notifications_received: list = []
        original_push_event = shell.push_event
        def track_push_event(msg):
            notifications_received.append(msg)
            original_push_event(msg)
        shell.push_event = track_push_event

        agent._on_after_tool_execution(
            session.uuid,
            {"name": "weather_tool", "arguments": {}},
            "Sunny and 25C",
            True
        )

        assert len(notifications_received) == 1
        notification = notifications_received[0].message
        assert notification.get_role() == "tool_result"
        assert len(notification.content) > 0
        assert "Sunny and 25C" in notification.content[0].data.get("content", "")
        assert notification.metadata.get("tool_status") == "ok"

        await agent.destroy_session(session.uuid)
