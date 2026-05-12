"""Integration tests for message queue processing.

Tests verify that messages posted to sessions are correctly processed
by the session's event loop.
"""

import asyncio
import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from peteos.agent import Agent
from peteos.chatbot.manager import ChatBotManager
from peteos.chatbot import Message, ContentPart
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager


class TestMessageQueueProcessing(AioHTTPTestCase):
    """Tests for message queue processing through sessions."""

    async def get_application(self):
        """Create the mock backend application."""
        app = web.Application()
        app.router.add_get("/", self.handle_root)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_post("/v1/chat/completions", self.handle_chat_completions)
        return app

    async def handle_root(self, request):
        return web.json_response({"status": "ok"})

    async def handle_models(self, request):
        return web.json_response({
            "data": [
                {"id": "test-model", "name": "Test Model", "object": "model"}
            ]
        })

    async def handle_chat_completions(self, request):
        body = await request.json()
        messages = body.get("messages", [])

        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                text = f"This is the agent's response to: {user_content}"
                break
        else:
            text = "Test response"

        async def event_generator():
            yield 'data: {"id": "chatcmpl-test", "object": "chat.completion", "created": 1234567890, "model": "' + body.get("model", "test-model") + '", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "' + text + '"}}], "usage": null}\n'
            yield 'data: [DONE]\n'

        response = web.StreamResponse(status=200)
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['Cache-Control'] = 'no-cache'
        await response.prepare(request)
        async for chunk in event_generator():
            await response.write(chunk.encode())
        await response.write_eof()
        return response

    @unittest_run_loop
    @pytest.mark.asyncio
    async def test_message_processed_during_execution_env_run(self):
        """Test that messages posted to session queue are processed correctly."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )

        chatbot_manager = ChatBotManager()
        await chatbot_manager.add_backend(
            "mock", f"http://{self.server.host}:{self.server.port}"
        )

        tool_manager = ToolManager()
        agent = Agent(role_manager, chatbot_manager, tool_manager)

        try:
            session = await agent.create_session("test")

            user_message = Message(
                role="user",
                content=[ContentPart(part_type="text", text="Hello")]
            )
            await session.queue_message(user_message)

            await asyncio.sleep(1)

            assert len(session.chat_history.messages) >= 1
            assert session.chat_history.messages[0].get_role() == "user"
            assert session.chat_history.messages[0].content[0].text == "Hello"
        finally:
            await session.stop()

    @unittest_run_loop
    @pytest.mark.asyncio
    async def test_multiple_messages_processed_in_sequence(self):
        """Test that multiple messages are processed in sequence."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )

        chatbot_manager = ChatBotManager()
        await chatbot_manager.add_backend(
            "mock", f"http://{self.server.host}:{self.server.port}"
        )

        tool_manager = ToolManager()
        agent = Agent(role_manager, chatbot_manager, tool_manager)

        try:
            session = await agent.create_session("test")

            for i in range(3):
                msg = Message(
                    role="user",
                    content=[ContentPart(part_type="text", text=f"Message {i}")]
                )
                await session.queue_message(msg)
                await asyncio.sleep(0.5)  # Wait for each response

            user_messages = [m for m in session.chat_history.messages if m.get_role() == "user"]
            assert len(user_messages) == 3
            for i in range(3):
                assert user_messages[i].content[0].text == f"Message {i}"
        finally:
            await session.stop()

    @unittest_run_loop
    @pytest.mark.asyncio
    async def test_messages_not_lost_when_queue_polling(self):
        """Test that messages are not lost during queue polling."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )

        chatbot_manager = ChatBotManager()
        await chatbot_manager.add_backend(
            "mock", f"http://{self.server.host}:{self.server.port}"
        )

        tool_manager = ToolManager()
        agent = Agent(role_manager, chatbot_manager, tool_manager)

        try:
            session = await agent.create_session("test")

            msg = Message(
                role="user",
                content=[ContentPart(part_type="text", text="Test")]
            )
            await session.queue_message(msg)

            for _ in range(20):
                await asyncio.sleep(0.1)
                if len(session.chat_history.messages) > 0:
                    break

            user_messages = [m for m in session.chat_history.messages if m.get_role() == "user"]
            assert len(user_messages) > 0
            assert user_messages[0].content[0].text == "Test"
        finally:
            await session.stop()
