"""Integration tests for invoke_role function."""

import asyncio
import json
import sys
sys.path.insert(0, '/home/frygge/projects/private/peteos')

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from peteos.agent import Agent
from peteos.chatbot.manager import ChatBotManager
from peteos.chatbot import Message, ContentPart
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager
from peteos.session import invoke_role, _extract_last_assistant_text, ExecStatus, Session


# --- Mock HTTP Backend ---

async def mock_models(request: web.Request) -> web.Response:
    return web.json_response({
        "data": [
            {"id": "test-model", "name": "Test Model", "object": "model"}
        ]
    })


async def mock_chat(request: web.Request) -> web.Response:
    body = await request.json()
    messages = body.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = f"I received your message: {msg.get('content', '')}"
            break
    else:
        text = "Test response"

    async def event_generator():
        yield f'data: {{"id": "chatcmpl-test", "object": "chat.completion", "created": 1234567890, "model": "{body.get("model", "test-model")}", "choices": [{{"index": 0, "delta": {{"role": "assistant", "content": "{text}"}}}}], "usage": null}}\n'
        yield 'data: [DONE]\n'

    response = web.StreamResponse(status=200)
    response.headers['Content-Type'] = 'text/event-stream'
    response.headers['Cache-Control'] = 'no-cache'
    await response.prepare(request)
    async for chunk in event_generator():
        await response.write(chunk.encode())
    await response.write_eof()
    return response


async def mock_root(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


def mock_app():
    app = web.Application()
    app.router.add_get("/", mock_root)
    app.router.add_get("/v1/models", mock_models)
    app.router.add_post("/v1/chat/completions", mock_chat)
    return app


@pytest.fixture
async def mock_server(event_loop) -> TestServer:
    """Start a mock chatbot HTTP server."""
    app = mock_app()
    server = TestServer(app)
    await server.start_server()
    yield server
    await server.close()


@pytest.fixture
async def mock_agent(mock_server):
    """Create an Agent with mock backend ready to use."""
    role_manager = RoleManager()
    role_manager.register_role(
        Role(name="test", description="Test", model="test-model")
    )

    chatbot_manager = ChatBotManager()
    await chatbot_manager.add_backend(
        "mock",
        f"http://{mock_server.host}:{mock_server.port}"
    )

    tool_manager = ToolManager()
    agent = Agent(role_manager, chatbot_manager, tool_manager)
    return agent


# --- Tests for invoke_role ---


class TestInvokeRole:
    """invoke_role integration tests using mock HTTP backend."""

    @pytest.mark.asyncio
    async def test_invoke_role_returns_answer(self, mock_agent, mock_server):
        """Basic invoke_role returns answer from chatbot."""
        result = await invoke_role(
            role_name="test",
            prompt="Hello world",
            agent=mock_agent,
            timeout=10.0,
        )

        assert result["answer"] != ""
        assert result["session"] is None  # keep_session=False
        assert len(result["history"]) >= 2  # system + user
        # Clean up
        if mock_agent._sessions:
            await mock_agent.destroy_session(list(mock_agent._sessions.keys())[0])

    @pytest.mark.asyncio
    async def test_invoke_role_keep_session(self, mock_agent, mock_server):
        """invoke_role with keep_session=True returns running session."""
        result = await invoke_role(
            role_name="test",
            prompt="Hello",
            agent=mock_agent,
            keep_session=True,
            timeout=10.0,
        )

        assert result["session"] is not None
        assert result["session"].is_running()

        # Can queue another message on the same session
        await result["session"].queue_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Follow-up")],
        ))
        await asyncio.sleep(2)

        # Clean up
        if mock_agent._sessions:
            await mock_agent.destroy_session(list(mock_agent._sessions.keys())[0])

    @pytest.mark.asyncio
    async def test_invoke_role_existing_session(self, mock_agent, mock_server):
        """invoke_role with existing_session for continuation."""
        session = await mock_agent.create_session("test")

        result = await invoke_role(
            role_name="test",
            prompt="Continuation",
            existing_session=session,
            keep_session=True,
            timeout=10.0,
        )

        assert result["session"] is session
        assert result["session"].is_running()
        assert len(result["history"]) >= 3  # system + user1 + user2

        await mock_agent.destroy_session(session.uuid)

    @pytest.mark.asyncio
    async def test_invoke_role_timeout(self, mock_agent):
        """invoke_role raises TimeoutError when chatbot doesn't respond."""
        # Use a port that won't respond
        chatbot_manager = ChatBotManager()
        await chatbot_manager.add_backend(
            "mock",
            "http://localhost:19999"  # No server listening
        )
        agent = Agent(mock_agent._role_manager, chatbot_manager, mock_agent._tool_manager)

        try:
            with pytest.raises(asyncio.TimeoutError):
                await invoke_role(
                    role_name="test",
                    prompt="Hello",
                    agent=agent,
                    timeout=1.0,
                )
        finally:
            if agent._sessions:
                await agent.destroy_session(list(agent._sessions.keys())[0])

    @pytest.mark.asyncio
    async def test_invoke_role_no_existing_session_or_agent(self):
        """invoke_role raises ValueError without agent or existing_session."""
        with pytest.raises(ValueError, match="Either agent or existing_session"):
            await invoke_role(role_name="test", prompt="Hello")

    @pytest.mark.asyncio
    async def test_invoke_role_existing_session_not_running(self):
        """invoke_role raises RuntimeError for non-running session."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test", model="test-model")
        )
        chatbot_manager = ChatBotManager()
        role = role_manager.get_role("test")
        tool_manager = ToolManager()
        session = Session(
            role=role,
            tool_manager=tool_manager,
            chatbot_manager=chatbot_manager,
        )

        with pytest.raises(RuntimeError, match="existing_session is not running"):
            await invoke_role(role_name="test", prompt="Hello", existing_session=session)


class TestExtractLastAssistantText:
    """Tests for the _extract_last_assistant_text helper."""

    @pytest.mark.asyncio
    async def test_extract_returns_last_assistant(self):
        """Extracts the last assistant message text."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test", model="test-model")
        )
        chatbot_manager = ChatBotManager()

        role = role_manager.get_role("test")
        tool_manager = ToolManager()
        session = Session(
            role=role,
            tool_manager=tool_manager,
            chatbot_manager=chatbot_manager,
        )

        session.append_and_notify(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Question 1")],
        ))
        session.append_and_notify(Message(
            role="assistant",
            content=[ContentPart(part_type="text", text="Answer 1")],
        ))
        session.append_and_notify(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Question 2")],
        ))
        session.append_and_notify(Message(
            role="assistant",
            content=[ContentPart(part_type="text", text="Answer 2")],
        ))

        result = _extract_last_assistant_text(session.chat_history)
        assert result == "Answer 2"

    @pytest.mark.asyncio
    async def test_extract_returns_empty_when_no_assistant(self):
        """Returns empty string when no assistant message exists."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test", model="test-model")
        )
        chatbot_manager = ChatBotManager()

        role = role_manager.get_role("test")
        tool_manager = ToolManager()
        session = Session(
            role=role,
            tool_manager=tool_manager,
            chatbot_manager=chatbot_manager,
        )

        session.append_and_notify(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Just a user message")],
        ))

        result = _extract_last_assistant_text(session.chat_history)
        assert result == ""

    @pytest.mark.asyncio
    async def test_extract_skips_non_assistant(self):
        """Skips non-assistant messages to find the last assistant."""
        role_manager = RoleManager()
        role_manager.register_role(
            Role(name="test", description="Test", model="test-model")
        )
        chatbot_manager = ChatBotManager()

        role = role_manager.get_role("test")
        tool_manager = ToolManager()
        session = Session(
            role=role,
            tool_manager=tool_manager,
            chatbot_manager=chatbot_manager,
        )

        session.append_and_notify(Message(
            role="assistant",
            content=[ContentPart(part_type="text", text="Assistant msg")],
        ))
        session.append_and_notify(Message(
            role="tool_result",
            content=[ContentPart(part_type="tool_result", name="test", content="result", tool_use_id="id1")],
        ))
        session.append_and_notify(Message(
            role="user",
            content=[ContentPart(part_type="text", text="User msg")],
        ))

        result = _extract_last_assistant_text(session.chat_history)
        assert result == "Assistant msg"
