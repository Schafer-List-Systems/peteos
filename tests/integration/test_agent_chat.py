"""Integration tests for Agent chat functionality."""

import asyncio
import pytest
import sys
sys.path.insert(0, '/home/frygge/projects/private/peteos')

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from peteos.agent import Agent
from peteos.chatbot.manager import ChatBotManager
from peteos.chatbot import Message, ContentPart
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager


class TestAgentChatFlow(AioHTTPTestCase):
    """Test that Agent properly processes messages and gets responses."""

    async def get_application(self):
        """Create the mock backend application."""
        app = web.Application()
        app.router.add_get("/", self.handle_root)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_post("/v1/chat/completions", self.handle_chat_completions)
        return app

    async def handle_root(self, request):
        """Root endpoint."""
        return web.json_response({"status": "ok"})

    async def handle_models(self, request):
        """Return available models in OpenAI format."""
        return web.json_response({
            "data": [
                {"id": "test-model", "name": "Test Model", "object": "model"}
            ]
        })

    async def handle_chat_completions(self, request):
        """Handle chat completions - echo back user message in SSE format."""
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
    async def test_message_queue_processing(self):
        """Test that posted messages are processed by the event loop."""
        role_manager = RoleManager()
        role_manager.register_role(Role(name="test", description="Test", model=".*"))

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
                content=[ContentPart(part_type="text", text="Test message")]
            )
            await session.queue_message(msg)

            await asyncio.sleep(1)

            assert len(session.chat_history.messages) >= 1, "Message should be in history"
            assert session.chat_history.messages[0].get_role() == "user"
        finally:
            await session.stop()

    @unittest_run_loop
    @pytest.mark.asyncio
    async def test_message_triggers_chatbot(self):
        """Test that messages trigger the chatbot to generate responses."""
        role_manager = RoleManager()
        role_manager.register_role(Role(name="test", description="Test", model=".*"))

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
                content=[ContentPart(part_type="text", text="What is 2+2?")]
            )
            await session.queue_message(msg)

            await asyncio.sleep(2)

            # Verify chatbot was called by checking history has assistant response
            assert len(session.chat_history.messages) >= 2, "Should have user and agent messages"
        finally:
            await session.stop()

    @unittest_run_loop
    @pytest.mark.asyncio
    async def test_full_chat_roundtrip(self):
        """Test complete user question -> agent response flow."""
        role_manager = RoleManager()
        role_manager.register_role(Role(name="test", description="Test", model=".*"))

        chatbot_manager = ChatBotManager()
        await chatbot_manager.add_backend(
            "mock", f"http://{self.server.host}:{self.server.port}"
        )

        tool_manager = ToolManager()
        agent = Agent(role_manager, chatbot_manager, tool_manager)

        try:
            session = await agent.create_session("test")

            question = "What is the weather?"
            msg = Message(
                role="user",
                content=[ContentPart(part_type="text", text=question)]
            )
            await session.queue_message(msg)

            await asyncio.sleep(2)

            history = session.chat_history.messages
            assert len(history) >= 2, "Should have user message and response"

            assistant_messages = [m for m in history if m.get_role() == 'assistant']
            assert len(assistant_messages) > 0, "Should have at least one assistant response"
        finally:
            await session.stop()
