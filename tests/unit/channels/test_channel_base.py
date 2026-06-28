"""Unit tests for Channel base class."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from peteos.engine.channel import Channel, NotificationEvent
from peteos.chatbot import Message, ContentPart


# Helper for testing abstract Channel (not collected by pytest)
class _TestChannel(Channel):
    """Test implementation of abstract Channel."""

    def __init__(self, name: str, runner):
        """Initialize test channel."""
        super().__init__(name, runner)
        self._sent_messages = []

    async def send(self, message: str, session_uuid: uuid.UUID | None = None) -> None:
        """Send message by storing it."""
        self._sent_messages.append(message)


class TestChannelRegistry:
    """Test channel registry functionality."""

    def test_channel_registry_adds_channels(self):
        """Test that channels are added to registry on creation."""
        mock_runner = MagicMock()
        ch = _TestChannel("test1", mock_runner)
        assert ch.name in Channel._registry
        assert Channel._registry[ch.name] == ch
        mock_runner.subscribe.assert_called_once_with(ch)
        # Clean up
        Channel.deregister_all()

    def test_channel_registry_overwrites_duplicate_name(self):
        """Test that registering a channel with same name overwrites."""
        mock_runner = MagicMock()
        ch1 = _TestChannel("test1", mock_runner)
        ch2 = _TestChannel("test1", mock_runner)
        assert ch2 in Channel._registry.values()
        assert ch1 not in Channel._registry.values()
        # Clean up
        Channel.deregister_all()

    def test_channel_get_by_name(self):
        """Test get_by_name retrieves correct channel."""
        mock_runner = MagicMock()
        ch1 = _TestChannel("test1", mock_runner)
        ch2 = _TestChannel("test2", mock_runner)
        assert Channel.get_by_name("test1") == ch1
        assert Channel.get_by_name("test2") == ch2
        assert Channel.get_by_name("nonexistent") is None
        # Clean up
        Channel.deregister_all()

    def test_channel_list_all(self):
        """Test list_all returns all registered channels."""
        mock_runner = MagicMock()
        ch1 = _TestChannel("test1", mock_runner)
        ch2 = _TestChannel("test2", mock_runner)
        channels = Channel.list_all()
        assert len(channels) >= 2
        assert "test1" in channels
        assert "test2" in channels
        # Clean up
        Channel.deregister_all()

    def test_channel_deregister_all(self):
        """Test deregister_all removes all channels."""
        mock_runner = MagicMock()
        ch1 = _TestChannel("test1", mock_runner)
        ch2 = _TestChannel("test2", mock_runner)
        Channel.deregister_all()
        assert len(Channel._registry) == 0


class TestChannelMethods:
    """Test Channel instance methods."""

    @pytest.mark.asyncio
    async def test_channel_send(self):
        """Test send stores message."""
        mock_runner = MagicMock()
        ch = _TestChannel("test", mock_runner)
        await ch.send("Hello, World!")
        assert len(ch._sent_messages) == 1
        assert ch._sent_messages[0] == "Hello, World!"
        # Clean up
        if ch.name in Channel._registry:
            del Channel._registry[ch.name]


class TestChannelFilterToggles:
    """Test enable/disable reasoning, tool_calls, tool_results."""

    def setup_method(self):
        self.runner = MagicMock()

    def test_reasoning_default_on(self):
        ch = _TestChannel("t", self.runner)
        assert ch._show_reasoning is True

    def test_tool_calls_default_on(self):
        ch = _TestChannel("t", self.runner)
        assert ch._show_tool_calls is True

    def test_tool_results_default_on(self):
        ch = _TestChannel("t", self.runner)
        assert ch._show_tool_results is True

    def test_enable_reasoning_off(self):
        ch = _TestChannel("t", self.runner)
        ch.enable_reasoning(False)
        assert ch._show_reasoning is False

    def test_enable_tool_calls_off(self):
        ch = _TestChannel("t", self.runner)
        ch.enable_tool_calls(False)
        assert ch._show_tool_calls is False

    def test_enable_tool_results_off(self):
        ch = _TestChannel("t", self.runner)
        ch.enable_tool_results(False)
        assert ch._show_tool_results is False


class TestChannelSessionSubscription:
    """Test subscribe_to_session and unsubscribe_from_session.

    The channel is already registered with its runner via the constructor.
    These methods just store/unstore the session UUID for routing.
    """

    def setup_method(self):
        self.runner = MagicMock()
        self.session_uuid = uuid.uuid4()

    def test_subscribe_to_session(self):
        ch = _TestChannel("t", self.runner)
        result = ch.subscribe_to_session(self.session_uuid)
        assert result is True
        assert ch._session_uuid == self.session_uuid

    def test_subscribe_to_different_session_returns_false(self):
        ch = _TestChannel("t", self.runner)
        ch.subscribe_to_session(self.session_uuid)
        new_uuid = uuid.uuid4()
        result = ch.subscribe_to_session(new_uuid)
        assert result is False

    def test_unsubscribe_from_session(self):
        ch = _TestChannel("t", self.runner)
        ch.subscribe_to_session(self.session_uuid)
        result = ch.unsubscribe_from_session(self.session_uuid)
        assert result is True
        assert ch._session_uuid is None

    def test_unsubscribe_when_not_subscribed_returns_false(self):
        ch = _TestChannel("t", self.runner)
        result = ch.unsubscribe_from_session(self.session_uuid)
        assert result is False


def _iter_side_effect(items):
    """Generator-based side_effect that yields items then returns StopAsyncIteration."""
    for item in items:
        yield item


class TestChannelRunLoop:
    """Test Channel.run() notification loop with filtering."""

    @pytest.mark.asyncio
    async def test_run_delivers_plain_message(self):
        mock_runner = MagicMock()
        ch = _TestChannel("t", mock_runner)
        test_msg = Message.create(
            role="user",
            content_parts=[ContentPart.create_text("hi")],
        )

        ch._wait = AsyncMock(side_effect=_iter_side_effect([test_msg, None]))
        ch._running = True
        ch._session_uuid = None

        await ch.run()

        assert len(ch._sent_messages) == 1

    @pytest.mark.asyncio
    async def test_run_delivers_notification_event(self):
        mock_runner = MagicMock()
        ch = _TestChannel("t", mock_runner)
        test_msg = Message.create(
            role="user",
            content_parts=[ContentPart.create_text("hi")],
        )
        event = NotificationEvent(session_uuid=uuid.uuid4(), message=test_msg)

        ch._wait = AsyncMock(side_effect=_iter_side_effect([event, None]))
        ch._running = True
        ch._session_uuid = event.session_uuid

        await ch.run()

        assert len(ch._sent_messages) == 1

    @pytest.mark.asyncio
    async def test_run_skips_reasoning_when_disabled(self):
        mock_runner = MagicMock()
        ch = _TestChannel("t", mock_runner)
        reasoning_msg = Message.create(
            role="reasoning",
            content_parts=[ContentPart.create_thinking("thinking")],
        )
        user_msg = Message.create(
            role="user",
            content_parts=[ContentPart.create_text("hi")],
        )

        ch._wait = AsyncMock(side_effect=_iter_side_effect([reasoning_msg, user_msg, None]))
        ch._running = True
        ch._show_reasoning = False
        ch._session_uuid = None

        await ch.run()

        # Only the non-reasoning message delivered
        assert len(ch._sent_messages) == 1

    @pytest.mark.asyncio
    async def test_run_skips_tool_when_disabled(self):
        mock_runner = MagicMock()
        ch = _TestChannel("t", mock_runner)
        tool_msg = Message.create(role="tool", content_parts=[])
        user_msg = Message.create(
            role="user",
            content_parts=[ContentPart.create_text("hi")],
        )

        ch._wait = AsyncMock(side_effect=_iter_side_effect([tool_msg, user_msg, None]))
        ch._running = True
        ch._show_tool_calls = False
        ch._session_uuid = None

        await ch.run()

        assert len(ch._sent_messages) == 1

    @pytest.mark.asyncio
    async def test_run_skips_tool_result_when_disabled(self):
        mock_runner = MagicMock()
        ch = _TestChannel("t", mock_runner)
        tool_result_msg = Message.create(role="tool_result", content_parts=[])
        user_msg = Message.create(
            role="user",
            content_parts=[ContentPart.create_text("hi")],
        )

        ch._wait = AsyncMock(side_effect=[tool_result_msg, user_msg, None])
        ch._running = True
        ch._show_tool_results = False
        ch._session_uuid = None

        await ch.run()

        assert len(ch._sent_messages) == 1

    @pytest.mark.asyncio
    async def test_run_stops_on_none_event(self):
        mock_runner = MagicMock()
        ch = _TestChannel("t", mock_runner)
        ch._wait = AsyncMock(return_value=None)
        ch._running = True

        await ch.run()

        assert ch._running is False

    @pytest.mark.asyncio
    async def test_run_stops_on_none_message(self):
        mock_runner = MagicMock()
        ch = _TestChannel("t", mock_runner)
        event = NotificationEvent(session_uuid=uuid.uuid4(), message=None)
        ch._wait = AsyncMock(return_value=event)
        ch._running = True
        ch.send = AsyncMock()

        await ch.run()

        assert ch._running is False
