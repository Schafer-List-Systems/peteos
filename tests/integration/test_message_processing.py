"""Integration tests for message queue processing.

Tests verify that messages posted to sessions are correctly processed
by the session's event loop.
"""

import asyncio
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer, TestClient

from peteos.agent import Agent
from peteos.chatbot import ChatBotManager
from peteos.chatbot import Message, ContentPart
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager


def make_mock_app():
    """Create a mock backend application."""
    app = web.Application()

    async def handle_root(request):
        return web.json_response({"status": "ok"})

    async def handle_models(request):
        return web.json_response({
            "data": [
                {"id": "test-model", "name": "Test Model", "object": "model"}
            ]
        })

    async def handle_chat_completions(request):
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

    app.router.add_get("/", handle_root)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    return app


@pytest.mark.asyncio
async def test_message_processed_during_execution_env_run():
    """Test that messages posted to session queue are processed correctly."""
    app = make_mock_app()
    server = TestServer(app)
    client = TestClient(server)
    await server.start_server()
    await client.start_server()

    try:
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )

        ChatBotManager.reset()
        await ChatBotManager.add_backend(
            "mock", f"http://{server.host}:{server.port}"
        )

        tool_manager = ToolManager()
        agent = Agent(role_manager, tool_manager)

        session = await agent.create_session("test")

        user_message = Message(
            role="user",
            content=[ContentPart(part_type="text", text="Hello")]
        )
        await session.queue_message(user_message)

        await asyncio.sleep(1)

        assert len(session.chat_history.messages) >= 1
        user_msgs = [m for m in session.chat_history.messages if m.get_role() == "user"]
        assert len(user_msgs) >= 1
        assert user_msgs[0].content[0].text == "Hello"

        await session.stop()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_multiple_messages_processed_in_sequence():
    """Test that multiple messages are processed in sequence."""
    app = make_mock_app()
    server = TestServer(app)
    client = TestClient(server)
    await server.start_server()
    await client.start_server()

    try:
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )

        ChatBotManager.reset()
        await ChatBotManager.add_backend(
            "mock", f"http://{server.host}:{server.port}"
        )

        tool_manager = ToolManager()
        agent = Agent(role_manager, tool_manager)

        session = await agent.create_session("test")

        # Queue all 3 messages at once
        for i in range(3):
            msg = Message(
                role="user",
                content=[ContentPart(part_type="text", text=f"Message {i}")]
            )
            session.push_event(msg)

        # Wait for processing
        await asyncio.sleep(2)

        user_messages = [m for m in session.chat_history.messages if m.get_role() == "user"]
        assert len(user_messages) >= 3
        for i in range(3):
            assert user_messages[i].content[0].text == f"Message {i}"

        await session.stop()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_messages_not_lost_when_queue_polling():
    """Test that messages are not lost during queue polling."""
    app = make_mock_app()
    server = TestServer(app)
    client = TestClient(server)
    await server.start_server()
    await client.start_server()

    try:
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test role", model="test-model")
        )

        ChatBotManager.reset()
        await ChatBotManager.add_backend(
            "mock", f"http://{server.host}:{server.port}"
        )

        tool_manager = ToolManager()
        agent = Agent(role_manager, tool_manager)

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

        await session.stop()
    finally:
        await client.close()