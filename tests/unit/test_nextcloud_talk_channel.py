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
from peteos.rolemanager import RoleManager


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
    role_manager = RoleManager()
    role_manager.register_role(Role(name="test", description="Test"))
    chatbot_manager = MagicMock()
    tool_manager = MagicMock()
    return MagicMock(
        role_manager=role_manager,
        create_session=MagicMock(),
        get_session=MagicMock(),
        register_channel=MagicMock(),
    )


def _make_signature(body: str, secret: str, random_nonce: str) -> str:
    return hmac.new(
        secret.encode(),
        (random_nonce + body).encode(),
        hashlib.sha256,
    ).hexdigest()


class TestNextcloudTalkChannelInit:
    """Test NextcloudTalkChannel initialization."""

    def test_channel_creation(self, agent):
        role_manager = RoleManager()
        role_manager.register_role(Role(name="test", description="Test"))
        agent = MagicMock(create_session=MagicMock(), get_session=MagicMock())

        channel = NextcloudTalkChannel(
            name="nextcloud",
            agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc123",
            bot_secret="secret",
            host="0.0.0.0",
            port=9999,
        )

        assert channel.name == "nextcloud"
        assert channel._nextcloud_url == "https://cloud.example.com"
        assert channel._bot_id == "abc123"
        assert channel._bot_secret == "secret"
        assert channel._running is False
        assert channel._rooms == {}
        assert channel._session_conversations == {}

    def test_channel_registered_with_agent(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc123", bot_secret="secret",
        )

        agent.register_channel.assert_called_with(channel)


class TestSignatureVerification:
    """Test HMAC-SHA256 webhook signature verification."""

    def test_valid_signature(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc123", bot_secret="mysecret",
        )

        body = '{"type":"Create"}'
        random_nonce = "a" * 64
        signature = _make_signature(body, "mysecret", random_nonce)

        assert channel._verify_signature(body.encode(), random_nonce, signature) is True

    def test_invalid_signature(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc123", bot_secret="mysecret",
        )

        body = '{"type":"Create"}'
        random_nonce = "b" * 64

        assert channel._verify_signature(body.encode(), random_nonce, "totallywrong") is False

    def test_signature_with_wrong_secret(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc123", bot_secret="secret_a",
        )

        body = '{"type":"Create"}'
        random_nonce = "c" * 64
        signature = _make_signature(body, "secret_a", random_nonce)

        # Channel with different secret should reject
        channel._bot_secret = "secret_b"
        assert channel._verify_signature(body.encode(), random_nonce, signature) is False


class TestEventDispatch:
    """Test webhook event dispatch."""

    @pytest.mark.asyncio
    async def test_handle_create_message(self, agent):
        agent.role_manager = RoleManager()
        agent.role_manager.register_role(Role(name="test", description="Test"))

        session_uuid = uuid.uuid4()
        session_mock = MagicMock(uuid=session_uuid)
        session_mock.queue_message = AsyncMock(return_value=None)
        agent.create_session = MagicMock(return_value=session_mock)
        agent.get_session = MagicMock(return_value=session_mock)

        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc", bot_secret="secret",
        )

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

        agent.create_session.assert_called_once_with("test")
        assert channel._active_session_uuid == session_uuid
        assert channel._rooms["conv1"] == session_uuid
        assert channel._session_conversations[session_uuid] == "conv1"

    @pytest.mark.asyncio
    async def test_handle_join(self, agent):
        agent.role_manager = RoleManager()
        agent.role_manager.register_role(Role(name="test", description="Test"))

        session_uuid = uuid.uuid4()
        agent.create_session = MagicMock(return_value=MagicMock(uuid=session_uuid))

        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc", bot_secret="secret",
        )

        event = {
            "type": "Join",
            "actor": {"displayName": "Admin", "id": "bots/hashed"},
            "object": {"token": "room5", "type": "room"},
        }
        await channel._handle_join(event)

        assert channel._rooms["room5"] == session_uuid
        assert channel._active_session_uuid == session_uuid

    @pytest.mark.asyncio
    async def test_handle_leave(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc", bot_secret="secret",
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
            nextcloud_url="https://cloud.example.com",
            bot_id="abc", bot_secret="secret",
        )
        caplog.set_level(logging.INFO)

        event = {
            "type": "Like",
            "actor": {"displayName": "User1"},
            "content": ":thumbsup:",
        }
        await channel._handle_reaction(event)
        assert "Reaction ':thumbsup:'" in caplog.text

    @pytest.mark.asyncio
    async def test_handle_undo_reaction(self, agent, caplog):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc", bot_secret="secret",
        )
        caplog.set_level(logging.INFO)

        event = {
            "type": "Undo",
            "actor": {"displayName": "User1"},
            "object": {"content": ":thumbsup:"},
        }
        await channel._handle_reaction_undo(event)
        assert "Reaction removed ':thumbsup:'" in caplog.text

    @pytest.mark.asyncio
    async def test_empty_message_ignored(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc", bot_secret="secret",
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

    def test_existing_session_reused(self, agent):
        existing_uuid = uuid.uuid4()
        existing_session = MagicMock(uuid=existing_uuid)
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc", bot_secret="secret",
        )
        channel._rooms["tok1"] = existing_uuid
        agent.get_session = MagicMock(return_value=existing_session)

        result = channel._get_or_create_session("tok1")

        assert result is existing_session
        agent.create_session.assert_not_called()

    def test_new_session_created(self, agent):
        new_uuid = uuid.uuid4()
        new_session = MagicMock(uuid=new_uuid)
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc", bot_secret="secret",
        )
        agent.create_session = MagicMock(return_value=new_session)

        result = channel._get_or_create_session("newroom")

        assert result is new_session
        agent.create_session.assert_called_once_with("test")
        assert channel._rooms["newroom"] == new_uuid
        assert channel._session_conversations[new_uuid] == "newroom"

    def test_failed_session_creation_returns_none(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc", bot_secret="secret",
        )
        agent.create_session = AsyncMock(side_effect=RuntimeError("failed"))

        result = channel._get_or_create_session("badroom")

        assert result is None
        assert "badroom" not in channel._rooms


class TestReceive:
    """Test receive method."""

    def test_receive_returns_none(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc", bot_secret="secret",
        )
        assert channel.receive() is None


class TestSend:
    """Test send method."""

    def test_send_no_active_session(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc", bot_secret="secret",
        )
        # _active_session_uuid is None by default
        channel.send(Message(role="assistant", content=[ContentPart(part_type="text", text="hello")]))  # Should not raise

    def test_send_no_conversation_mapping(self, agent):
        test_uuid = uuid.uuid4()
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc", bot_secret="secret",
        )
        channel._active_session_uuid = test_uuid
        # No mapping in _session_conversations
        channel.send(Message(role="assistant", content=[ContentPart(part_type="text", text="hello")]))  # Should not raise

    @pytest.mark.asyncio
    async def test_send_calls_nextcloud_api(self, agent):
        test_uuid = uuid.uuid4()
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc", bot_secret="secret",
        )
        channel._active_session_uuid = test_uuid
        channel._session_conversations[test_uuid] = "convtoken"

        mock_response = AsyncMock(
            __aenter__=AsyncMock(return_value=MagicMock(status=201, text=AsyncMock(return_value="ok"))),
            __aexit__=AsyncMock(return_value=None),
        )
        mock_post_ctx = AsyncMock(return_value=mock_response)

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session_obj = MagicMock(post=mock_post_ctx)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session_obj)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            channel.send(Message(
                role="assistant",
                content=[ContentPart(part_type="text", text="Hello!")],
            ))
            await asyncio.sleep(0.05)

        mock_post_ctx.assert_called_once()
        body = json.loads(mock_post_ctx.call_args[1]["data"])
        assert body["message"] == "Hello!"
        assert body["replyTo"] == ""
        assert body["referenceId"] is not None
        assert body["silent"] is True
        assert "convtoken" in mock_post_ctx.call_args[0][0]

    @pytest.mark.asyncio
    async def test_send_multiple_parts_sequentially(self, agent):
        """Test that multiple content parts are sent in order."""
        test_uuid = uuid.uuid4()
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc", bot_secret="secret",
        )
        channel._active_session_uuid = test_uuid
        channel._session_conversations[test_uuid] = "convtoken"

        mock_response = AsyncMock(
            __aenter__=AsyncMock(return_value=MagicMock(status=201, text=AsyncMock(return_value="ok"))),
            __aexit__=AsyncMock(return_value=None),
        )
        mock_post_ctx = AsyncMock(return_value=mock_response)

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session_obj = MagicMock(post=mock_post_ctx)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session_obj)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            channel.send(Message(
                role="assistant",
                content=[
                    ContentPart(part_type="reasoning", reasoning="Thinking..."),
                    ContentPart(part_type="text", text="Hello!"),
                ],
            ))
            await asyncio.sleep(0.05)

        assert mock_post_ctx.call_count == 2
        first_msg = json.loads(mock_post_ctx.call_args_list[0][1]["data"])["message"]
        second_msg = json.loads(mock_post_ctx.call_args_list[1][1]["data"])["message"]
        assert first_msg == "> _Thinking..._"
        assert second_msg == "Hello!"


class TestStartStop:
    """Test server lifecycle."""

    @pytest.mark.asyncio
    async def test_start_returns_server_url(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            nextcloud_url="https://cloud.example.com",
            bot_id="abc", bot_secret="secret",
            host="0.0.0.0", port=0,
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
            nextcloud_url="https://cloud.example.com",
            bot_id="abc", bot_secret="secret",
            host="0.0.0.0", port=0,
        )

        await channel.start()
        assert channel._running is True
        await channel.stop()
        assert channel._running is False
