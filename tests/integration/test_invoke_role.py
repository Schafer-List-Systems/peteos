"""Integration tests for invoke_agent function."""

import asyncio
import json
import sys
sys.path.insert(0, '/home/frygge/projects/private/peteos')

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from peteos.agent import Agent
from peteos.chatbot import ChatBotManager
from peteos.chatbot import Message, ContentPart, ChatHistory
from peteos.role import Role
from peteos.toolmanager import ToolManager
from peteos.session import invoke_agent, _extract_last_assistant_text, ExecStatus, Session
from unittest.mock import AsyncMock, MagicMock, patch


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
    role = Role(name="test", description="Test", model="test-model")

    ChatBotManager.reset()
    await ChatBotManager.add_backend(
        "mock",
        f"http://{mock_server.host}:{mock_server.port}"
    )

    tool_manager = ToolManager()
    agent = Agent(role, tool_manager)
    return agent


# --- Tests for invoke_agent ---


class TestInvokeRole:
    """invoke_agent integration tests using mock HTTP backend."""

    @pytest.mark.asyncio
    async def test_invoke_agent_returns_answer(self, mock_agent, mock_server):
        """Basic invoke_agent returns answer from chatbot."""
        result = await invoke_agent(
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
    async def test_invoke_agent_keep_session(self, mock_agent, mock_server):
        """invoke_agent with keep_session=True returns running session."""
        result = await invoke_agent(
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
    async def test_invoke_agent_existing_session(self, mock_agent, mock_server):
        """invoke_agent with existing_session for continuation."""
        session = await mock_agent.create_session()

        result = await invoke_agent(
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
    async def test_invoke_agent_timeout(self, mock_agent):
        """invoke_agent raises TimeoutError when chatbot doesn't respond."""
        # Patch the backend's model discovery so we don't hit the network
        async def noop_models(*args, **kwargs):
            return ["test-model"]

        # Create a mock agent with a broken backend using class-level ChatBotManager
        ChatBotManager.reset()
        ChatBotManager._timeout = 1.0
        with patch.object(
            ChatBotManager,
            "_list_models_for_api_type",
            noop_models,
        ):
            await ChatBotManager.add_backend(
                "mock",
                "http://localhost:19999",  # No server listening
                api_type="openai",
            )

        role = Role(name="test", description="Test", model="test-model")
        agent = Agent(role, mock_agent._tool_manager)

        try:
            with pytest.raises(asyncio.TimeoutError):
                await invoke_agent(
                    prompt="Hello",
                    agent=agent,
                    timeout=1.0,
                )
        finally:
            if agent._sessions:
                await agent.destroy_session(list(agent._sessions.keys())[0])

    @pytest.mark.asyncio
    async def test_invoke_agent_no_existing_session_or_agent(self):
        """invoke_agent raises ValueError without agent or existing_session."""
        with pytest.raises(ValueError, match="Either agent or existing_session"):
            await invoke_agent(prompt="Hello", timeout=10.0)

    @pytest.mark.asyncio
    async def test_invoke_agent_existing_session_not_running(self):
        """invoke_agent raises RuntimeError for non-running session."""
        role = Role(name="test", description="Test", model="test-model")
        tool_manager = ToolManager()
        mock_env = MagicMock()
        mock_env._hooks = {"before_tool_execution": [], "after_tool_execution": [],
                          "before_notification_publish": [], "before_send_to_chatbot": [],
                          "before_loop_continue": [], "after_step": []}
        session = Session(
            role=role,
            tool_manager=tool_manager,
            execution_environment=mock_env,
        )
        # Session must be started to be "running"
        # This session was never started, so is_running() should return False

        with pytest.raises(RuntimeError, match="existing_session is not running"):
            await invoke_agent(prompt="Hello", existing_session=session, timeout=10.0)


class TestExtractLastAssistantText:
    """Tests for the _extract_last_assistant_text helper."""

    @pytest.mark.asyncio
    async def test_extract_returns_last_assistant(self):
        """Extracts the last assistant message text."""
        chat_history = ChatHistory(messages=[])

        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Question 1")],
        ))
        chat_history.append_message(Message(
            role="assistant",
            content=[ContentPart(part_type="text", text="Answer 1")],
        ))
        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Question 2")],
        ))
        chat_history.append_message(Message(
            role="assistant",
            content=[ContentPart(part_type="text", text="Answer 2")],
        ))

        result = _extract_last_assistant_text(chat_history)
        assert result == "Answer 2"

    @pytest.mark.asyncio
    async def test_extract_returns_empty_when_no_assistant(self):
        """Returns empty string when no assistant message exists."""
        chat_history = ChatHistory(messages=[])

        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Just a user message")],
        ))

        result = _extract_last_assistant_text(chat_history)
        assert result == ""

    @pytest.mark.asyncio
    async def test_extract_skips_non_assistant(self):
        """Skips non-assistant messages to find the last assistant."""
        chat_history = ChatHistory(messages=[])

        chat_history.append_message(Message(
            role="assistant",
            content=[ContentPart(part_type="text", text="Assistant msg")],
        ))
        chat_history.append_message(Message(
            role="tool_result",
            content=[ContentPart(part_type="tool_result", name="test", content="result", tool_use_id="id1")],
        ))
        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="User msg")],
        ))

        result = _extract_last_assistant_text(chat_history)
        assert result == "Assistant msg"
