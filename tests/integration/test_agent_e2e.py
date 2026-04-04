"""End-to-end integration tests using mock backend.

This tests the complete agent flow:
1. Agent receives message via queue
2. Session processes message through execution environment
3. Chatbot (mocked HTTP server) generates response
4. Response is added to session history
"""

import asyncio
import pytest
import sys
sys.path.insert(0, '/home/frygge/projects/private/peteos')

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from peteos.agent import Agent
from peteos.chatbot.manager import ChatBotManager
from peteos.chatbot import Message
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager


class TestAgentE2E(AioHTTPTestCase):
    """End-to-end agent tests with mock HTTP backend."""

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

        # Find last user message
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
    async def test_agent_question_answer_flow(self):
        """Test complete question -> answer flow."""
        # Setup
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test", model="test-model")
        )

        chatbot_manager = ChatBotManager()
        await chatbot_manager.add_backend(
            "mock",
            f"http://{self.server.host}:{self.server.port}"
        )

        tool_manager = ToolManager()

        agent = Agent(role_manager, chatbot_manager, tool_manager)
        await agent.start()

        try:
            # Create session
            session = agent.create_session("test")

            # Send question
            question = "What is 2 + 2?"
            msg = Message(content={"role": "user", "content": question})
            agent.post_message(session.uuid, msg)

            # Wait for response
            await asyncio.sleep(2)

            # Verify we got a response
            history = session.chat_history.messages
            assert len(history) == 2, f"Expected 2 messages, got {len(history)}"

            # Verify user message
            assert history[0].content.get("role") == "user"
            assert history[0].content.get("content") == question

            # Verify assistant response
            assert history[1].content.get("role") == "assistant"
            assert "2 + 2" in history[1].content.get("text", "")

        finally:
            await agent.stop()

    @unittest_run_loop
    async def test_agent_multiple_messages(self):
        """Test multiple messages in sequence."""
        # Setup
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test", model="test-model")
        )

        chatbot_manager = ChatBotManager()
        await chatbot_manager.add_backend(
            "mock",
            f"http://{self.server.host}:{self.server.port}"
        )

        tool_manager = ToolManager()

        agent = Agent(role_manager, chatbot_manager, tool_manager)
        await agent.start()

        try:
            session = agent.create_session("test")

            # Send first question
            agent.post_message(session.uuid, Message(content={
                "role": "user",
                "content": "Hello"
            }))
            await asyncio.sleep(1)

            # Send second question
            agent.post_message(session.uuid, Message(content={
                "role": "user",
                "content": "How are you?"
            }))
            await asyncio.sleep(1)

            # Verify we got responses
            history = session.chat_history.messages
            assert len(history) == 4, f"Expected 4 messages, got {len(history)}"

            # Verify alternating user/assistant
            roles = [m.content.get("role") for m in history]
            assert roles == ["user", "assistant", "user", "assistant"]

        finally:
            await agent.stop()

    @unittest_run_loop
    async def test_agent_tool_call_flow(self):
        """Test agent with tool calls."""
        # Setup
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test", model="test-model")
        )

        chatbot_manager = ChatBotManager()
        await chatbot_manager.add_backend(
            "mock",
            f"http://{self.server.host}:{self.server.port}"
        )

        # Add a tool
        tool_manager = ToolManager()

        def get_weather(city: str) -> str:
            return f"Weather in {city}: Sunny, 25°C"

        tool_manager.register_tool(func=get_weather)

        agent = Agent(role_manager, chatbot_manager, tool_manager)
        await agent.start()

        try:
            session = agent.create_session("test")

            # Send question that might use tool
            agent.post_message(session.uuid, Message(content={
                "role": "user",
                "content": "What's the weather in London?"
            }))
            await asyncio.sleep(2)

            # Verify agent processed the message
            history = session.chat_history.messages
            assert len(history) >= 2

        finally:
            await agent.stop()


class TestAgentQuickVerify:
    """Quick verification that the agent works end-to-end."""

    @pytest.mark.asyncio
    async def test_basic_flow_with_mock_backend(self, tmp_path):
        """Test basic flow using mock backend."""
        # This test is deprecated - use TestAgentE2E instead
        pass
