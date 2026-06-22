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

    def test_config_defaults(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(),
        )
        assert channel._config["host"] == "0.0.0.0"
        assert channel._config["default_role"] == "test"
        assert channel._config["show_reasoning"] is True
        assert channel._config["show_tool_calls"] is True
        assert channel._config["show_tool_results"] is True
        assert channel._config["prefix_actor_names"] is False


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
        await channel.send(Message(role="assistant", content=[ContentPart(part_type="text", text="hello")]))

    @pytest.mark.asyncio
    async def test_send_no_conversation_mapping(self, agent):
        test_uuid = uuid.uuid4()
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(),
        )
        await channel.send(Message(role="assistant", content=[ContentPart(part_type="text", text="hello")]), session_uuid=test_uuid)

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


class TestShouldSendPart:
    """Test _should_send_part filtering logic."""

    def _make_channel(self, **overrides):
        agent_mock = MagicMock()
        config = _make_config(**overrides)
        return NextcloudTalkChannel(name="nextcloud", agent=agent_mock, config=config)

    def test_text_always_sends(self):
        ch = self._make_channel()
        part = ContentPart(part_type="text", text="Hello")
        assert ch._should_send_part(part) is True

    def test_reasoning_respects_config(self):
        ch = self._make_channel(show_reasoning=True)
        part = ContentPart(part_type="reasoning", reasoning="Thinking...")
        assert ch._should_send_part(part) is True

        ch._config["show_reasoning"] = False
        assert ch._should_send_part(part) is False

    def test_tool_use_respects_config(self):
        ch = self._make_channel(show_tool_calls=True)
        part = ContentPart(part_type="tool_use", data={"id": "1", "name": "x", "arguments": "{}"})
        assert ch._should_send_part(part) is True

        ch._config["show_tool_calls"] = False
        assert ch._should_send_part(part) is False

    def test_tool_result_respects_config(self):
        ch = self._make_channel(show_tool_results=True)
        part = ContentPart(part_type="tool_result", data={"content": "result"})
        assert ch._should_send_part(part) is True

        ch._config["show_tool_results"] = False
        assert ch._should_send_part(part) is False

    def test_image_always_sends(self):
        ch = self._make_channel()
        part = ContentPart(part_type="image", source={"type": "url", "url": "http://img.png"})
        assert ch._should_send_part(part) is True

    def test_video_always_sends(self):
        ch = self._make_channel()
        part = ContentPart(part_type="video", source={"type": "url", "url": "http://vid.mp4"})
        assert ch._should_send_part(part) is True


class TestFormatForNextcloud:
    """Test _format_for_nextcloud for each part type."""

    def _make_channel(self, **overrides):
        agent_mock = MagicMock()
        config = _make_config(**overrides)
        return NextcloudTalkChannel(name="nextcloud", agent=agent_mock, config=config)

    def test_format_text(self):
        ch = self._make_channel()
        part = ContentPart(part_type="text", text="Hello")
        payload = ch._format_for_nextcloud(part, "ref-1")
        assert payload["message"] == "Hello"
        assert payload["silent"] is False
        assert payload["referenceId"] == "ref-1"

    def test_format_reasoning(self):
        ch = self._make_channel()
        part = ContentPart(part_type="reasoning", reasoning="Thinking...")
        payload = ch._format_for_nextcloud(part, "ref-1")
        assert payload["message"] == "> _Thinking..._"
        assert payload["silent"] is True

    def test_format_tool_use_dict(self):
        ch = self._make_channel()
        part = ContentPart(part_type="tool_use", data={
            "id": "tc1", "name": "calculate", "arguments": "{'x': 1}",
        })
        payload = ch._format_for_nextcloud(part, "ref-1")
        assert "calculate({'x': 1})" in payload["message"]
        assert "/* id: tc1 */" in payload["message"]

    def test_format_tool_use_list(self):
        ch = self._make_channel()
        part = ContentPart(part_type="tool_calls", data={
            "tool_calls": [
                {"id": "1", "name": "a", "arguments": "{}"},
                {"id": "2", "name": "b", "arguments": "{}"},
            ],
        })
        payload = ch._format_for_nextcloud(part, "ref-1")
        assert "a({})" in payload["message"]
        assert "b({})" in payload["message"]

    def test_format_tool_result_string(self):
        ch = self._make_channel()
        part = ContentPart(part_type="tool_result", data={"content": "result text"})
        payload = ch._format_for_nextcloud(part, "ref-1")
        assert payload["message"] == "result text"

    def test_format_tool_result_dict(self):
        ch = self._make_channel()
        part = ContentPart(part_type="tool_result", data={"content": {"key": "value"}})
        payload = ch._format_for_nextcloud(part, "ref-1")
        assert "```json" in payload["message"]
        assert '"key": "value"' in payload["message"]

    def test_format_image_url(self):
        ch = self._make_channel()
        part = ContentPart(part_type="image", source={"type": "url", "url": "http://img.png"})
        payload = ch._format_for_nextcloud(part, "ref-1")
        assert payload["message"] == "![image](http://img.png)"

    def test_format_image_base64(self):
        ch = self._make_channel()
        part = ContentPart(part_type="image", source={"type": "base64", "media_type": "image/png"})
        payload = ch._format_for_nextcloud(part, "ref-1")
        assert payload["message"] == "[Image] (base64 encoded, image/png)"

    def test_format_video_url(self):
        ch = self._make_channel()
        part = ContentPart(part_type="video", source={"type": "url", "url": "http://vid.mp4"})
        payload = ch._format_for_nextcloud(part, "ref-1")
        assert payload["message"] == "📹 [Video](http://vid.mp4)"

    def test_format_pdf_url(self):
        ch = self._make_channel()
        part = ContentPart(part_type="pdf", source={"type": "url", "url": "http://doc.pdf"})
        payload = ch._format_for_nextcloud(part, "ref-1")
        assert payload["message"] == "📄 [Document](http://doc.pdf)"

    def test_format_pdf_base64(self):
        ch = self._make_channel()
        part = ContentPart(part_type="pdf", source={"type": "base64", "media_type": "application/pdf"})
        payload = ch._format_for_nextcloud(part, "ref-1")
        assert payload["message"] == "[Document] (base64 encoded, application/pdf)"


class TestWebhookHandling:
    """Test _handle_webhook dispatch."""

    @pytest.mark.asyncio
    async def test_handle_webhook_create(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(bot_secret="secret"),
        )

        body = json.dumps({"type": "Create"})
        random_nonce = "a" * 64
        signature = _make_signature(body, "secret", random_nonce)

        mock_request = MagicMock()
        mock_request.read = AsyncMock(return_value=body.encode())
        mock_request.headers = {
            "X-NEXTCLOUD-TALK-RANDOM": random_nonce,
            "X-NEXTCLOUD-TALK-SIGNATURE": signature,
        }
        mock_request.remote = "127.0.0.1"

        response = await channel._handle_webhook(mock_request)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_handle_webhook_invalid_signature(self, agent):
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(bot_secret="secret"),
        )

        mock_request = MagicMock()
        mock_request.read = AsyncMock(return_value=b'{}')
        mock_request.headers = {
            "X-NEXTCLOUD-TALK-RANDOM": "nonce",
            "X-NEXTCLOUD-TALK-SIGNATURE": "bad",
        }
        mock_request.remote = "127.0.0.1"

        response = await channel._handle_webhook(mock_request)
        assert response.status == 401

    @pytest.mark.asyncio
    async def test_handle_webhook_unknown_event_type(self, agent, caplog):
        caplog.set_level(logging.WARNING)
        channel = NextcloudTalkChannel(
            name="nextcloud", agent=agent,
            config=_make_config(bot_secret="secret"),
        )

        body = json.dumps({"type": "UnknownEvent"})
        random_nonce = "x" * 64
        signature = _make_signature(body, "secret", random_nonce)

        mock_request = MagicMock()
        mock_request.read = AsyncMock(return_value=body.encode())
        mock_request.headers = {
            "X-NEXTCLOUD-TALK-RANDOM": random_nonce,
            "X-NEXTCLOUD-TALK-SIGNATURE": signature,
        }
        mock_request.remote = "127.0.0.1"

        response = await channel._handle_webhook(mock_request)
        assert response.status == 200
        assert "Unknown event type" in caplog.text
