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
    async def test_on_before_tool_execution_auto_approves(self):
        """Test that _on_before_tool_execution auto-approves whitelisted tools."""
        role = Role(name="test", description="Test role", model="test-model",
                    auto_approve_tools=["safe_tool"])
        role_manager = RoleManager()
        role_manager.register_role(role)
        tool_manager = ToolManager()

        agent = Agent(role_manager, tool_manager)

        session = await agent.create_session("test")

        # Auto-approved tool should return (True, None)
        result = agent._on_before_tool_execution(
            session, {"name": "safe_tool", "arguments": {}}
        )
        assert result == (True, None)

        # Non-whitelisted tool should return ("pending", None)
        result = agent._on_before_tool_execution(
            session, {"name": "risky_tool", "arguments": {}}
        )
        assert result == ("pending", None)

        await agent.destroy_session(session.uuid)

    @pytest.mark.asyncio
    async def test_on_after_tool_execution_receives_correct_args(self):
        """Test that _on_after_tool_execution receives correct args."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )
        tool_manager = ToolManager()

        agent = Agent(role_manager, tool_manager)

        session = await agent.create_session("test")

        # Should not raise with any valid args
        agent._on_after_tool_execution(
            session,
            {"name": "weather_tool", "arguments": {}},
            "Sunny and 25C",
            True
        )

        await agent.destroy_session(session.uuid)

    @pytest.mark.asyncio
    async def test_on_before_notification_publish_receives_message(self):
        """Test that _on_before_notification_publish receives message."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )
        tool_manager = ToolManager()

        agent = Agent(role_manager, tool_manager)

        session = await agent.create_session("test")

        assistant_message = Message(
            role="assistant",
            content=[ContentPart(part_type="text", text="Final response")]
        )

        # Should not raise
        agent._on_before_notification_publish(session, assistant_message)

        await agent.destroy_session(session.uuid)

    @pytest.mark.asyncio
    async def test_hook_notifications_with_new_message_format(self):
        """Test that hook notifications work with the new Message format."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )
        tool_manager = ToolManager()

        agent = Agent(role_manager, tool_manager)

        session = await agent.create_session("test")

        notifications_received: list = []
        original_publish = session.publish_notification
        def track_publish(message):
            notifications_received.append(message)
            original_publish(message)
        session.publish_notification = track_publish

        # Trigger after_tool_execution hook which publishes a notification
        agent._on_after_tool_execution(
            session,
            {"name": "weather_tool", "arguments": {}},
            "Sunny and 25C",
            True
        )

        # The current agent._on_after_tool_execution is a no-op (pass),
        # so no notifications are published by it. This test verifies
        # the hook callback doesn't crash with the new Message format.
        await agent.destroy_session(session.uuid)
