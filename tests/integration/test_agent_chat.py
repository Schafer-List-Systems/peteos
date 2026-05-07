"""Integration tests for Agent chat functionality."""

import asyncio
import pytest
import sys
sys.path.insert(0, '/home/frygge/projects/private/peteos')

from unittest.mock import MagicMock, AsyncMock
from peteos.agent import Agent
from peteos.chatbot.manager import ChatBotManager
from peteos.chatbot import Message, ContentPart
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager


class TestAgentChatFlow:
    """Test that Agent properly processes messages and gets responses."""

    @pytest.fixture
    def setup_components(self):
        """Create mocked components for testing."""
        role_manager = RoleManager()
        role_manager.register_role(Role(name="test", description="Test", model=".*"))

        # Mock chatbot manager with a mock chatbot
        chatbot_manager = MagicMock()

        # Create proper mock response object
        mock_response_data = {"text": "This is a test response", "role": "assistant"}
        mock_response = MagicMock()
        mock_response.data = mock_response_data

        mock_chatbot = AsyncMock()
        mock_chatbot.send_message = AsyncMock(return_value=mock_response)

        chatbot_manager.list_chatbots = MagicMock(return_value=[
            ("test-model", mock_chatbot)
        ])

        tool_manager = ToolManager()

        return role_manager, chatbot_manager, tool_manager

    @pytest.mark.asyncio
    async def test_message_queue_processing(self, setup_components):
        """Test that posted messages are processed by the event loop."""
        role_manager, chatbot_manager, tool_manager = setup_components

        agent = Agent(role_manager, chatbot_manager, tool_manager)
        await agent.start()

        session = agent.create_session("test")

        # Post a message using new Message format
        msg = Message(
            role="user",
            content=[ContentPart(part_type="text", text="Test message")]
        )
        agent.post_message(session.uuid, msg)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Verify queue was processed
        assert agent._message_queues[session.uuid].empty(), "Message should be processed"

        # Verify message was added to history
        assert len(session.chat_history.messages) >= 1, "Message should be in history"
        assert session.chat_history.messages[0].role == "user"

        await agent.stop()

    @pytest.mark.asyncio
    async def test_message_triggers_chatbot(self, setup_components):
        """Test that messages trigger the chatbot to generate responses."""
        role_manager, chatbot_manager, tool_manager = setup_components

        agent = Agent(role_manager, chatbot_manager, tool_manager)
        await agent.start()

        session = agent.create_session("test")

        # Post a message using new Message format
        msg = Message(
            role="user",
            content=[ContentPart(part_type="text", text="What is 2+2?")]
        )
        agent.post_message(session.uuid, msg)

        # Wait for processing and chatbot response
        await asyncio.sleep(2)

        # Verify chatbot was called
        chatbot = chatbot_manager.list_chatbots(".*")[0][1]
        assert chatbot.send_message.called, "Chatbot should be called"

        # Verify response was added to history
        assert len(session.chat_history.messages) >= 2, "Should have user and agent messages"

        await agent.stop()

    @pytest.mark.asyncio
    async def test_full_chat_roundtrip(self, setup_components):
        """Test complete user question -> agent response flow."""
        role_manager, chatbot_manager, tool_manager = setup_components

        agent = Agent(role_manager, chatbot_manager, tool_manager)
        await agent.start()

        session = agent.create_session("test")

        # Send question using new Message format
        question = "What is the weather?"
        msg = Message(
            role="user",
            content=[ContentPart(part_type="text", text=question)]
        )
        agent.post_message(session.uuid, msg)

        # Wait for response
        await asyncio.sleep(3)

        # Verify we got a response
        history = session.chat_history.messages
        assert len(history) >= 2, "Should have user message and response"

        # Verify response exists - check role attribute, not content
        assistant_messages = [m for m in history if m.role == 'assistant']
        assert len(assistant_messages) > 0, "Should have at least one assistant response"

        await agent.stop()
