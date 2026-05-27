"""Integration test for OAP engine with mock backend."""

import json
import sys
sys.path.insert(0, '/home/frygge/projects/private/peteos')

import asyncio
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from peteos.agent import Agent
from peteos.chatbot.manager import ChatBotManager
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager

from peteos.oap import (
    AgenticObjectBase,
    Error,
    agentic_object,
    invoke as oap_invoke,
    tool,
)


# --- Mock HTTP Backend ---

_chat_call_count = 0

async def mock_models(request: web.Request) -> web.Response:
    return web.json_response({
        "data": [
            {"id": "test-model", "name": "Test Model", "object": "model"}
        ]
    })


async def mock_root(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def mock_chat(request: web.Request) -> web.Response:
    global _chat_call_count
    _chat_call_count += 1

    body = await request.json()
    messages = body.get("messages", [])

    # Check if produce_output is in the system prompt or messages
    has_produce_output = False
    system_text = body.get("system", "")
    if isinstance(system_text, str) and "produce_output" in system_text:
        has_produce_output = True
    elif isinstance(system_text, list):
        for item in system_text:
            if isinstance(item, dict) and "text" in item:
                if "produce_output" in str(item["text"]):
                    has_produce_output = True
                    break

    if not has_produce_output:
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and "produce_output" in content:
                has_produce_output = True
            elif isinstance(content, list):
                for part in content:
                    part_text = part.get("text", "") if isinstance(part, dict) else str(part)
                    if "produce_output" in str(part_text):
                        has_produce_output = True
                        break

    if has_produce_output and _chat_call_count == 1:
        # First call only: return produce_output tool_use
        async def event_generator():
            event = json.dumps({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": body.get("model", "test-model"),
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "tool_use",
                            "function": {
                                "name": "produce_output",
                                "arguments": json.dumps({
                                    "name": "test",
                                    "count": 42,
                                }),
                            },
                        }],
                    },
                }],
                "usage": None,
            })
            yield f'data: {event}\n'
            yield 'data: [DONE]\n'
    else:
        # Text response for all other cases
        text = "Hello!"
        async def event_generator():
            event = json.dumps({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": body.get("model", "test-model"),
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": text}
                }],
                "usage": None,
            })
            yield f'data: {event}\n'
            yield 'data: [DONE]\n'

    response = web.StreamResponse(status=200)
    response.headers['Content-Type'] = 'text/event-stream'
    response.headers['Cache-Control'] = 'no-cache'
    await response.prepare(request)
    async for chunk in event_generator():
        await response.write(chunk.encode())
    await response.write_eof()
    return response


def mock_app():
    app = web.Application()
    app.router.add_get("/", mock_root)
    app.router.add_get("/v1/models", mock_models)
    app.router.add_post("/v1/chat/completions", mock_chat)
    return app


@pytest.fixture
async def mock_server(event_loop) -> TestServer:
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
        Role(name="oap_test", description="OAP test role", model=".*")
    )
    chatbot_manager = ChatBotManager(timeout=None)
    await chatbot_manager.add_backend(
        "mock", f"http://{mock_server.host}:{mock_server.port}"
    )
    tool_manager = ToolManager()
    return Agent(role_manager, chatbot_manager, tool_manager)


class TestOAPIntegration:
    @pytest.mark.asyncio
    async def test_invoke_non_persistent(self, mock_agent):
        """Basic non-persistent OAP invocation."""
        global _chat_call_count
        _chat_call_count = 0

        @agentic_object()
        class Greeter(AgenticObjectBase):
            """A simple greeter."""

            @tool()
            def greet(self, name: str) -> str:
                """Say hello to someone."""
                return f"Hello, {name}!"

        greeter = Greeter()
        greeter.agent = mock_agent

        result = await oap_invoke(greeter, prompt="Say hello to Alice")
        assert result["success"] is True
        assert "thread_id" in result

    @pytest.mark.asyncio
    async def test_invoke_with_output_schema(self, mock_agent):
        """OAP invocation with structured output schema."""
        global _chat_call_count
        _chat_call_count = 0

        from dataclasses import dataclass

        @agentic_object()
        class Parser(AgenticObjectBase):
            """A data parser."""

            @tool()
            def parse(self, text: str) -> dict:
                """Parse text into structured data."""
                return {"name": "test", "count": 42}

        @dataclass
        class Report:
            name: str
            count: int

        parser = Parser()
        parser.agent = mock_agent

        result = await oap_invoke(
            parser,
            prompt="Parse the data",
            output_schema=Report,
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_invoke_without_agent_raises(self):
        """invoke_agent without agent raises ValueError."""

        @agentic_object()
        class MyObj(AgenticObjectBase):
            pass

        obj = MyObj()
        with pytest.raises(ValueError, match="No Agent available"):
            await oap_invoke(obj, prompt="test")

    @pytest.mark.asyncio
    async def test_invoke_persistent_thread(self, mock_agent):
        """Persistent thread carries state across invocations."""
        global _chat_call_count
        _chat_call_count = 0

        @agentic_object()
        class Counter(AgenticObjectBase):
            """Count messages."""

            @tool()
            def increment(self, by: int = 1) -> int:
                """Increment counter."""
                return 42

        counter = Counter()
        counter.agent = mock_agent

        result1 = await oap_invoke(
            counter,
            prompt="Start counting",
            thread_id="counter-thread-1",
        )
        assert result1["success"] is True

        # Second invocation on same thread
        result2 = await oap_invoke(
            counter,
            prompt="Continue counting",
            thread_id="counter-thread-1",
        )
        assert result2["success"] is True

    @pytest.mark.asyncio
    async def test_sub_agent_invoke_enabled(self, mock_agent):
        """Sub-agent invocation when target has invoke_sub_agents enabled."""

        @agentic_object(invoke_sub_agents=True)
        class Child(AgenticObjectBase):
            """A child object."""

            @tool()
            def child_action(self, value: str) -> str:
                """Do something."""
                return value

        @agentic_object()
        class Parent(AgenticObjectBase):
            """A parent object."""

        parent = Parent()
        parent.agent = mock_agent
        child = Child()

        result = parent.invoke(child, "prompt")
        assert not isinstance(result, Error)

    @pytest.mark.asyncio
    async def test_sub_agent_invoke_disabled(self, mock_agent):
        """Sub-agent invocation blocked when target has invoke_sub_agents disabled."""

        @agentic_object()  # no invoke_sub_agents
        class Child(AgenticObjectBase):
            """A child object."""

        @agentic_object()
        class Parent(AgenticObjectBase):
            """A parent object."""

        parent = Parent()
        child = Child()

        result = parent.invoke(child, "prompt")
        assert isinstance(result, Error)
        assert "not enabled" in result.message
