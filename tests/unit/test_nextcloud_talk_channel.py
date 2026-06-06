"""Unit tests for NextcloudTalkChannel."""

import asyncio
import hashlib
import hmac
import json
import logging
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from peteos.channels import NextcloudTalkChannel
from peteos.channels.channel import Channel
from peteos.chatbot import Message, ContentPart
from peteos.role import Role


def _cleanup_channels():
    for name in list(Channel._registry.keys()):
        Channel._registry.pop(name)


class MockTransport:
    """Mock transport for aiohttp tests."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def abort(self):
        pass


@pytest.fixture(autouse=True)
def clean_registry():
    _cleanup_channels()
    yield
    _cleanup_channels()


@pytest.fixture
def agent():
    role = Role(name="test", description="Test")
    tool_manager = MagicMock()
    agent_mock = MagicMock(
        role=role,
        create_session=AsyncMock(),
        get_session=MagicMock(),
        register_channel=MagicMock(),
    )
    # Make get_session return a valid session for subscribe_to_session mock
    mock_session = MagicMock()
    mock_session.uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    agent_mock.get_session.return_value = mock_session
    return agent_mock


def _make_config(**overrides) -> dict:
    """Create a minimal Nextcloud config dict with defaults."""
    config = {
        "nextcloud_url": "https://cloud.example.com",
        "bot_id": "abc123",
        "bot_secret": "secret",
        "host": "0.0.0.0",
        "port": 0,
        "default_role": "test",
        "nextcloud_api_timeout": 10.0,
        **overrides,
    }
    return config


def _make_signature(body: str, secret: str, random_nonce: str) -> str:
    return hmac.new(
        secret.encode(),
        (random_nonce + body).encode(),
        hashlib.sha256,
    ).hexdigest()


class TestNextcloudTalkChannelInit:
    """Test NextcloudTalkChannel initialization."""

    def test_channel_creation(self, agent):
        agent = MagicMock(create_session=MagicMock(), get_session=MagicMock())

        channel = NextcloudTalkChannel(
            name="nextcloud",
            agent=agent,
            config=_make_config(host="0.0.0.0", port=9999),
        )

        assert channel.name == "nextcloud"
        assert channel._config["nextcloud_url"] == "https://cloud.example.com"
        assert channel._config["bot_id"] == "abc123"
        assert channel._config["bot_secret"] == "secret"
        assert channel._running is False
        assert channel._rooms == {}
        assert channel._session_conversations == {}

    def test_channel_registered_with_agent(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(),
        )

        agent.register_channel.assert_called_with(channel)


class TestSignatureVerification:
    """Test HMAC-SHA256 webhook signature verification."""

    def test_valid_signature(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(bot_secret="mysecret"),
        )

        body = '{"type":"Create"}'
        random_nonce = "a" * 64
        signature = _make_signature(body, "mysecret", random_nonce)

        assert channel._verify_signature(body.encode(), random_nonce, signature) is True

    def test_invalid_signature(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(bot_secret="mysecret"),
        )

        body = '{"type":"Create"}'
        random_nonce = "b" * 64

        assert channel._verify_signature(body.encode(), random_nonce, "totallywrong") is False

    def test_signature_with_wrong_secret(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(bot_secret="secret_a"),
        )

        body = '{"type":"Create"}'
        random_nonce = "c" * 64
        signature = _make_signature(body, "secret_a", random_nonce)

        # Channel with different secret should reject
        channel._config["bot_secret"] = "secret_b"
        assert channel._verify_signature(body.encode(), random_nonce, signature) is False


class TestEventDispatch:
    """Test webhook event dispatch."""

    @pytest.mark.asyncio
    async def test_handle_create_message(self, agent):
        agent.role = Role(name="test", description="Test")

        session_uuid = uuid.uuid4()
        session_mock = MagicMock(uuid=session_uuid)
        session_mock.queue_message = AsyncMock(return_value=None)
        session_mock.start = AsyncMock(return_value=None)
        agent.get_session = MagicMock(return_value=session_mock)

        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(),
        )
        agent._session_channels = {}

        # App registers the room with a session
        await channel.register_room(session_uuid, "conv1")

        event = {
            "type": "Create",
            "actor": {"displayName": "TestUser", "id": "users/123"},
            "object": {
                "token": "conv1",
                "id": "msg1",
                "content": json.dumps({"message": "Hello bot!"}),
            },
        }
        await channel._handle_message(event)

        session_mock.queue_message.assert_called_once()
        queued_msg = session_mock.queue_message.call_args[0][0]
        assert queued_msg.content[0].text == "Hello bot!"

        # Await the thinking reaction task to prevent RuntimeWarning
        # _send_reaction will fail (no real aiohttp) but we need to drain the task
        with patch("aiohttp.ClientSession"):
            await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_handle_join(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(),
        )

        event = {
            "type": "Join",
            "actor": {"displayName": "Admin", "id": "bots/hashed"},
            "object": {"token": "room5", "type": "room"},
        }
        await channel._handle_join(event)

        # Join no longer creates sessions - channel just logs
        assert channel._rooms == {}

    @pytest.mark.asyncio
    async def test_handle_leave(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(),
        )
        channel._rooms["myroom"] = uuid.uuid4()

        event = {
            "type": "Leave",
            "actor": {"displayName": "Admin"},
            "object": {"token": "myroom"},
        }
        await channel._handle_leave(event)

        assert "myroom" not in channel._rooms

    @pytest.mark.asyncio
    async def test_handle_like_reaction(self, agent, caplog):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(),
        )
        caplog.set_level(logging.DEBUG)

        # Add a matching sent part so the reaction can match
        channel._sent_parts = [{"content": "+1", "tool_call_id": None, "session_uuid": uuid.uuid4()}]

        event = {
            "type": "Like",
            "actor": {"displayName": "User1"},
            "content": "+1",
        }
        await channel._handle_reaction(event)
        assert "Reaction" in caplog.text or "+1" in caplog.text

    @pytest.mark.asyncio
    async def test_handle_undo_reaction(self, agent, caplog):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(),
        )
        caplog.set_level(logging.INFO)

        event = {
            "type": "Undo",
            "actor": {"displayName": "User1"},
            "object": {"content": "+1"},
        }
        await channel._handle_reaction_undo(event)
        assert "Reaction removed '+1'" in caplog.text

    @pytest.mark.asyncio
    async def test_empty_message_ignored(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(),
        )

        event = {
            "type": "Create",
            "actor": {"displayName": "User"},
            "object": {"token": "c1", "content": json.dumps({"message": ""})},
        }
        await channel._handle_message(event)

        agent.create_session.assert_not_called()


class TestSessionRouting:
    """Test session lookup and creation."""

    @pytest.mark.asyncio
    async def test_find_session_existing(self, agent):
        existing_uuid = uuid.uuid4()
        existing_session = MagicMock(uuid=existing_uuid)
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(),
        )
        channel._rooms["tok1"] = existing_uuid
        agent.get_session = MagicMock(return_value=existing_session)

        result = await channel._find_session("tok1")

        assert result is existing_session

    @pytest.mark.asyncio
    async def test_find_session_not_registered_returns_none(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(),
        )

        result = await channel._find_session("unknown")

        assert result is None

    @pytest.mark.asyncio
    async def test_register_room(self, agent):
        session_uuid = uuid.uuid4()
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(),
        )
        # Mock subscribe_to_session to avoid session creation
        channel.subscribe_to_session = MagicMock()

        await channel.register_room(session_uuid, "myroom")

        assert channel._rooms["myroom"] == session_uuid
        assert channel._session_conversations[session_uuid] == "myroom"
        channel.subscribe_to_session.assert_called_once_with(session_uuid)


class TestSend:
    """Test send method."""

    @pytest.mark.asyncio
    async def test_send_no_active_session(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(),
        )
        # No session_uuid means no-op
        await channel.send(Message(role="assistant", content=[ContentPart(part_type="text", text="hello")]))  # Should not raise

    @pytest.mark.asyncio
    async def test_send_no_conversation_mapping(self, agent):
        test_uuid = uuid.uuid4()
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(),
        )
        # session_uuid present but no mapping in _session_conversations
        await channel.send(Message(role="assistant", content=[ContentPart(part_type="text", text="hello")]), session_uuid=test_uuid)  # Should not raise

    @pytest.mark.asyncio
    async def test_send_calls_nextcloud_api(self, agent):
        test_uuid = uuid.uuid4()
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(),
        )
        channel._session_conversations[test_uuid] = "convtoken"

        mock_resp_obj = MagicMock(status=201, text="ok")

        mock_post_ctx = MagicMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp_obj)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=None)

        # The code does: async with aiohttp.ClientSession() as session:
        #                  async with session.post(...) as resp:
        # So we need proper mock chains for both context managers
        mock_session_obj = MagicMock()
        mock_session_obj.post = MagicMock(return_value=mock_post_ctx)
        mock_session_obj.__aenter__ = AsyncMock(return_value=mock_session_obj)
        mock_session_obj.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session_obj)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            await channel.send(Message(
                role="assistant",
                content=[ContentPart(part_type="text", text="Hello!")],
            ), session_uuid=test_uuid)
            await asyncio.sleep(0.05)

            mock_session_obj.post.assert_called_once()
            call_args = mock_session_obj.post.call_args
            body = json.loads(call_args[1]["data"])
            assert body["message"] == "Hello!"
            assert body["replyTo"] == ""
            assert body["referenceId"] is not None
            assert body["silent"] is False  # text parts are not silent
            assert "convtoken" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_send_multiple_parts_sequentially(self, agent):
        """Test that multiple content parts are sent in order."""
        test_uuid = uuid.uuid4()
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(),
        )
        channel._session_conversations[test_uuid] = "convtoken"

        mock_resp_obj = MagicMock(status=201, text="ok")

        mock_post_ctx = MagicMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp_obj)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_session_obj = MagicMock()
        mock_session_obj.post = MagicMock(return_value=mock_post_ctx)
        mock_session_obj.__aenter__ = AsyncMock(return_value=mock_session_obj)
        mock_session_obj.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session_obj)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            await channel.send(Message(
                role="assistant",
                content=[
                    ContentPart(part_type="reasoning", reasoning="Thinking..."),
                    ContentPart(part_type="text", text="Hello!"),
                ],
            ), session_uuid=test_uuid)
            await asyncio.sleep(0.05)

            mock_session_obj.post.assert_called()
            call_args_list = mock_session_obj.post.call_args_list
            first_msg = json.loads(call_args_list[0][1]["data"])["message"]
            second_msg = json.loads(call_args_list[1][1]["data"])["message"]
            assert first_msg == "> _Thinking..._"
            assert second_msg == "Hello!"


class TestStartStop:
    """Test server lifecycle."""

    @pytest.mark.asyncio
    async def test_start_returns_server_url(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(port=0),
        )

        url = await channel.start()

        assert "http://" in url
        assert "/nextcloud-talk-webhook" in url
        assert channel._running is True
        assert channel._app is not None
        await channel.stop()

    @pytest.mark.asyncio
    async def test_stop_cleans_up(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(port=0),
        )

        await channel.start()
        assert channel._running is True
        await channel.stop()
        assert channel._running is False
