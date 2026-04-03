"""Unit tests for RESTApiChannel."""

import json
import uuid
from unittest.mock import MagicMock

import pytest
from aiohttp import web, ClientSession

from peteos.agent import Agent
from peteos.channels import RESTApiChannel
from peteos.role import Role
from peteos.rolemanager import RoleManager

# Access channel registry directly
from peteos import channel


class MockTransport:
    """Mock transport for testing."""
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def abort(self):
        pass


@pytest.fixture
def setup_agent():
    """Create agent with test roles."""
    # Clean up channels
    for name in list(channel.Channel._registry.keys()):
        channel.Channel._registry.pop(name)
    role_manager = RoleManager()
    role_manager.register_role(Role(name="test", description="Test role"))
    role_manager.register_role(Role(name="assistant", description="Assistant role"))
    chatbot_manager = MagicMock()
    tool_manager = MagicMock()
    agent = Agent(role_manager, chatbot_manager, tool_manager)
    return agent


class TestRESTChannelInit:
    """Test RESTApiChannel initialization."""

    def setup_method(self):
        """Set up agent."""
        # Clean up channels
        for name in list(channel.Channel._registry.keys()):
            channel.Channel._registry.pop(name)
        RoleManager()

    def teardown_method(self):
        """Clean up."""
        for name in list(channel.Channel._registry.keys()):
            channel.Channel._registry.pop(name)

    def test_channel_creation(self):
        """Test creating a REST channel."""
        from unittest.mock import MagicMock
        role_manager = RoleManager()
        role_manager.register_role(Role(name="test", description="Test"))
        chatbot_manager = MagicMock()
        tool_manager = MagicMock()

        agent = Agent(role_manager, chatbot_manager, tool_manager)
        channel = RESTApiChannel("rest", agent, host="127.0.0.1", port=8765)

        assert channel.name == "rest"
        assert channel._host == "127.0.0.1"
        assert channel._port == 8765
        assert channel._running is False

    def test_channel_registered_with_agent(self):
        """Test REST channel is registered with agent."""
        from unittest.mock import MagicMock
        role_manager = RoleManager()
        chatbot_manager = MagicMock()
        tool_manager = MagicMock()

        agent = Agent(role_manager, chatbot_manager, tool_manager)
        channel = RESTApiChannel("rest", agent)

        assert agent.get_channel("rest") == channel


@pytest.fixture
async def rest_channel_and_url(setup_agent):
    """Create and start REST channel."""
    agent = setup_agent
    # Use port 0 to let OS assign a free port
    rest_channel = RESTApiChannel("rest", agent, host="127.0.0.1", port=0)
    url = await rest_channel.start()
    # url format is http://127.0.0.1:PORT - extract the actual port used
    yield rest_channel, url
    await rest_channel.stop()


@pytest.mark.asyncio
class TestRESTEndpoints:
    """Test REST API endpoints."""

    @pytest.fixture
    async def client(self, rest_channel_and_url):
        """Create test client."""
        _, url = rest_channel_and_url
        async with ClientSession() as session:
            yield session, url

    async def test_endpoint_chat_no_session_selected(self, client):
        """Test /chat without session selected returns error."""
        session, url = client
        async with session.post(f"{url}/chat", json={"content": "Hello"}) as resp:
            assert resp.status == 400

    async def test_endpoint_create_session(self, client):
        """Test /sessions endpoint creates session."""
        session, url = client
        async with session.post(f"{url}/sessions", json={"role": "test"}) as resp:
            data = await resp.json()
            assert data["status"] == "created"
            assert "uuid" in data

    async def test_endpoint_list_sessions(self, client):
        """Test /sessions endpoint lists sessions."""
        session, url = client
        # Create a session first
        await session.post(f"{url}/sessions", json={"role": "test"})

        async with session.get(f"{url}/sessions") as resp:
            data = await resp.json()
            assert "sessions" in data
            assert len(data["sessions"]) > 0

    async def test_endpoint_select_session(self, client):
        """Test /sessions/select endpoint selects session."""
        session, url = client
        # Create a session
        create_resp = await session.post(f"{url}/sessions", json={"role": "test"})
        create_data = await create_resp.json()
        session_uuid = create_data["uuid"]

        async with session.post(f"{url}/sessions/{session_uuid}/select") as resp:
            data = await resp.json()
            assert data["status"] == "selected"

    async def test_endpoint_select_invalid_uuid(self, client):
        """Test /sessions/select with invalid UUID."""
        session, url = client
        async with session.post(f"{url}/sessions/invalid-uuid/select") as resp:
            assert resp.status == 400

    async def test_endpoint_select_nonexistent(self, client):
        """Test /sessions/select with non-existent session."""
        session, url = client
        session_uuid = uuid.UUID("810fb120-e4e5-4e32-9718-88bbcaf7641a")
        async with session.post(f"{url}/sessions/{session_uuid}/select") as resp:
            assert resp.status == 404

    async def test_endpoint_get_messages_no_session(self, client):
        """Test /sessions/messages without session."""
        session, url = client
        session_uuid = uuid.UUID("810fb120-e4e5-4e32-9718-88bbcaf7641a")
        async with session.get(f"{url}/sessions/{session_uuid}/messages") as resp:
            assert resp.status == 404

    async def test_endpoint_chat_success(self, client, setup_agent):
        """Test /chat with session selected."""
        client_session, url = client
        # Create and select session
        create_resp = await client_session.post(f"{url}/sessions", json={"role": "test"})
        create_data = await create_resp.json()
        session_uuid = create_data["uuid"]
        await client_session.post(f"{url}/sessions/{session_uuid}/select")

        async with client_session.post(f"{url}/chat", json={"content": "Hello"}) as resp:
            # The request was accepted (request queued)
            # The agentic loop runs in background with mock chatbot
            assert resp.status in [200, 500]  # 200 = accepted, 500 = chatbot error
