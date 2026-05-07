"""Integration tests for message queue processing.

Tests verify that messages posted to Agent's message queues are correctly
processed by sessions, even when the execution environment is running.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock
import asyncio

from peteos.agent import Agent
from peteos.chatbot.manager import ChatBotManager, BackendInfo
from peteos.chatbot import Message, ContentPart, OpenAIChatBot, ChatBotResponse
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager


class TestMessageQueueProcessing:
    """Tests for message queue processing through Agent."""

    @pytest.mark.asyncio
    async def test_message_processed_during_execution_env_run(self):
        """Test that messages posted to queue are processed correctly."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )
        chatbot_manager = ChatBotManager()

        # Create a mock chatbot that returns a proper response
        mock_chatbot = MagicMock(spec=OpenAIChatBot)

        async def mock_send_message(chat_history, streaming=True):
            # Return a response with role field
            class MockResponse:
                def __init__(self):
                    self._data = {"role": "assistant", "text": "Mock response"}
                    self._stream_complete = False

                @property
                def data(self):
                    return self._data

                def __aiter__(self):
                    async def inner():
                        if not self._stream_complete:
                            self._stream_complete = True
                            yield ("role", "assistant")
                            yield ("text", "Mock response")
                    return inner()

            return MockResponse()

        mock_chatbot.send_message = mock_send_message
        mock_chatbot.list_available_models = MagicMock(return_value=["test-model"])

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

        # Verify queue exists
        assert session.uuid in agent._message_queues
        queue = agent._message_queues[session.uuid]
        assert queue.empty()

        # Post a user message
        user_message = Message(
            role="user",
            content=[ContentPart(part_type="text", text="Hello")]
        )
        agent.post_message(session.uuid, user_message)

        # Verify message is in queue
        assert not queue.empty()
        assert queue.qsize() == 1

        # Give the agent loop time to process the message
        await asyncio.sleep(0.3)

        # Message should be processed - user message added to history
        assert len(session.chat_history.messages) >= 1
        assert session.chat_history.messages[0].role == "user"
        assert session.chat_history.messages[0].content[0].text == "Hello"

        await agent.stop()

    @pytest.mark.asyncio
    async def test_multiple_messages_processed_in_sequence(self):
        """Test that multiple messages are processed in sequence."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )
        chatbot_manager = ChatBotManager()

        # Create a mock chatbot that returns a proper response
        mock_chatbot = MagicMock(spec=OpenAIChatBot)

        async def mock_send_message(chat_history, streaming=True):
            class MockResponse:
                def __init__(self):
                    self._data = {"role": "assistant", "text": "Mock"}
                    self._stream_complete = False

                @property
                def data(self):
                    return self._data

                def __aiter__(self):
                    async def inner():
                        if not self._stream_complete:
                            self._stream_complete = True
                            yield ("role", "assistant")
                            yield ("text", "Mock")
                    return inner()

            return MockResponse()

        mock_chatbot.send_message = mock_send_message
        mock_chatbot.list_available_models = MagicMock(return_value=["test-model"])

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

        session = agent.create_session("test")

        # Post multiple messages
        for i in range(3):
            msg = Message(
                role="user",
                content=[ContentPart(part_type="text", text=f"Message {i}")]
            )
            agent.post_message(session.uuid, msg)

        # Give the agent loop time to process messages
        await asyncio.sleep(0.5)

        # All user messages should be processed and added to history
        user_messages = [m for m in session.chat_history.messages if m.role == "user"]
        assert len(user_messages) == 3
        for i in range(3):
            assert user_messages[i].content[0].text == f"Message {i}"

        await agent.stop()

    @pytest.mark.asyncio
    async def test_messages_not_lost_when_queue_polling(self):
        """Test that messages are not lost during queue polling."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )
        chatbot_manager = ChatBotManager()

        # Create a mock chatbot that returns a proper response
        mock_chatbot = MagicMock(spec=OpenAIChatBot)

        async def mock_send_message(chat_history, streaming=True):
            class MockResponse:
                def __init__(self):
                    self._data = {"role": "assistant", "text": "Mock"}
                    self._stream_complete = False

                @property
                def data(self):
                    return self._data

                def __aiter__(self):
                    async def inner():
                        if not self._stream_complete:
                            self._stream_complete = True
                            yield ("role", "assistant")
                            yield ("text", "Mock")
                    return inner()

            return MockResponse()

        mock_chatbot.send_message = mock_send_message
        mock_chatbot.list_available_models = MagicMock(return_value=["test-model"])

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

        session = agent.create_session("test")

        # Post message
        msg = Message(
            role="user",
            content=[ContentPart(part_type="text", text="Test")]
        )
        agent.post_message(session.uuid, msg)

        # Wait for message to be processed
        for _ in range(20):  # Try for up to 2 seconds
            await asyncio.sleep(0.1)
            if len(session.chat_history.messages) > 0:
                break

        # User message should be in history (may have assistant response after it)
        user_messages = [m for m in session.chat_history.messages if m.role == "user"]
        assert len(user_messages) > 0
        assert user_messages[0].content[0].text == "Test"

        await agent.stop()
