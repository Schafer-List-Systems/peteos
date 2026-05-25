"""End-to-end integration tests using mock backend.

This tests the complete agent flow:
1. Agent creates session with execution environment
2. Session receives message, drives execution environment
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
from peteos.chatbot import Message, ContentPart
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

        chatbot_manager = ChatBotManager(timeout=None)
        await chatbot_manager.add_backend(
            "mock",
            f"http://{self.server.host}:{self.server.port}"
        )

        tool_manager = ToolManager()

        agent = Agent(role_manager, chatbot_manager, tool_manager)

        try:
            # Create session (now async, auto-starts session event loop)
            session = await agent.create_session("test")

            # Send question directly to session (agent.post_message removed)
            question = "What is 2 + 2?"
            msg = Message(
                role="user",
                content=[ContentPart(part_type="text", text=question)]
            )
            await session.queue_message(msg)

            # Wait for response
            await asyncio.sleep(2)

            # Verify we got a response
            history = session.chat_history.messages
            assert len(history) >= 2, f"Expected 2 messages, got {len(history)}"

            # Verify user message using new format
            user_msgs = [m for m in history if m.get_role() == "user"]
            assert len(user_msgs) >= 1
            assert user_msgs[0].content[0].text == question

            # Verify assistant response using new format
            assistant_msgs = [m for m in history if m.get_role() == "assistant"]
            assert len(assistant_msgs) >= 1
            assert "2 + 2" in history[1].content[0].text

        finally:
            await session.stop()

    @unittest_run_loop
    async def test_agent_multiple_messages(self):
        """Test multiple messages in sequence."""
        # Setup
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test", model="test-model")
        )

        chatbot_manager = ChatBotManager(timeout=None)
        await chatbot_manager.add_backend(
            "mock",
            f"http://{self.server.host}:{self.server.port}"
        )

        tool_manager = ToolManager()

        agent = Agent(role_manager, chatbot_manager, tool_manager)

        try:
            session = await agent.create_session("test")

            # Send first question directly to session
            await session.queue_message(Message(
                role="user",
                content=[ContentPart(part_type="text", text="Hello")]
            ))
            await asyncio.sleep(1)

            # Send second question
            await session.queue_message(Message(
                role="user",
                content=[ContentPart(part_type="text", text="How are you?")]
            ))
            await asyncio.sleep(1)

            # Verify we got responses
            history = session.chat_history.messages
            assert len(history) >= 4, f"Expected 4 messages, got {len(history)}"

            # Verify alternating user/assistant using role attribute
            roles = [m.get_role() for m in history[:4]]
            assert "user" in roles and "assistant" in roles

        finally:
            await session.stop()

    @unittest_run_loop
    async def test_agent_tool_call_flow(self):
        """Test agent with tool calls."""
        # Setup
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test", model="test-model")
        )

        chatbot_manager = ChatBotManager(timeout=None)
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

        try:
            session = await agent.create_session("test")

            # Send question directly to session
            await session.queue_message(Message(
                role="user",
                content=[ContentPart(part_type="text", text="What's the weather in London?")]
            ))
            await asyncio.sleep(2)

            # Verify agent processed the message
            history = session.chat_history.messages
            assert len(history) >= 2

        finally:
            await session.stop()

    @unittest_run_loop
    async def test_session_initializes_with_tools(self):
        """Test that session initializes chat history with tool list."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test", model="test-model")
        )

        chatbot_manager = ChatBotManager(timeout=None)
        await chatbot_manager.add_backend(
            "mock",
            f"http://{self.server.host}:{self.server.port}"
        )

        tool_manager = ToolManager()

        def get_weather(city: str) -> str:
            """Get the current weather for a city."""
            return f"Weather in {city}: Sunny, 25°C"

        tool_manager.register_tool(func=get_weather)

        agent = Agent(role_manager, chatbot_manager, tool_manager)

        try:
            session = await agent.create_session("test")

            # Verify chat history has messages
            history = session.chat_history.messages
            assert len(history) >= 1

            # Find the tool list message - tools are stored as role="tool" messages
            tool_list_msg = None
            for msg in history:
                if msg.get_role() == "tool":
                    for part in msg.content:
                        if part.type == "tool" and "get_weather" in part.data.get("name", ""):
                            tool_list_msg = msg
                            break
                if tool_list_msg:
                    break

            assert tool_list_msg is not None, "Tool should be in chat history"

            # Verify the tool is mentioned
            assert any(part.type == "tool" and part.data.get("name") == "get_weather"
                      for msg in history for part in msg.content)

        finally:
            await session.stop()


class TestAgentQuickVerify:
    """Quick verification that the agent works end-to-end."""

    @pytest.mark.asyncio
    async def test_basic_flow_with_mock_backend(self, tmp_path):
        """Test basic flow using mock backend."""
        # This test is deprecated - use TestAgentE2E instead
        pass
