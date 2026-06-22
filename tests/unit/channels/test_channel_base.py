"""Unit tests for Channel base class."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from peteos.persona.channel import Channel, NotificationEvent
from peteos.chatbot import Message, ContentPart


# Helper for testing abstract Channel (not collected by pytest)
class _TestChannel(Channel):
    """Test implementation of abstract Channel."""

    def __init__(self, name: str, agent):
        """Initialize test channel."""
        super().__init__(name, agent)
        self._sent_messages = []

    async def send(self, message: str, session_uuid: uuid.UUID | None = None) -> None:
        """Send message by storing it."""
        self._sent_messages.append(message)


class TestChannelRegistry:
    """Test channel registry functionality."""

    def test_channel_registry_adds_channels(self):
        """Test that channels are added to registry on creation."""
        mock_agent = MagicMock()
        ch = _TestChannel("test1", mock_agent)
        assert ch.name in Channel._registry
        assert Channel._registry[ch.name] == ch
        # Clean up
        Channel.deregister_all()

    def test_channel_registry_overwrites_duplicate_name(self):
        """Test that registering a channel with same name overwrites."""
        mock_agent = MagicMock()
        ch1 = _TestChannel("test1", mock_agent)
        ch2 = _TestChannel("test1", mock_agent)
        assert ch2 in Channel._registry.values()
        assert ch1 not in Channel._registry.values()
        # Clean up
        Channel.deregister_all()

    def test_channel_get_by_name(self):
        """Test get_by_name retrieves correct channel."""
        mock_agent = MagicMock()
        ch1 = _TestChannel("test1", mock_agent)
        ch2 = _TestChannel("test2", mock_agent)
        assert Channel.get_by_name("test1") == ch1
        assert Channel.get_by_name("test2") == ch2
        assert Channel.get_by_name("nonexistent") is None
        # Clean up
        Channel.deregister_all()

    def test_channel_list_all(self):
        """Test list_all returns all registered channels."""
        mock_agent = MagicMock()
        ch1 = _TestChannel("test1", mock_agent)
        ch2 = _TestChannel("test2", mock_agent)
        channels = Channel.list_all()
        assert len(channels) >= 2
        assert "test1" in channels
        assert "test2" in channels
        # Clean up
        Channel.deregister_all()

    def test_channel_deregister_all(self):
        """Test deregister_all removes all channels."""
        mock_agent = MagicMock()
        ch1 = _TestChannel("test1", mock_agent)
        ch2 = _TestChannel("test2", mock_agent)
        Channel.deregister_all()
        assert len(Channel._registry) == 0


class TestChannelMethods:
    """Test Channel instance methods."""

    @pytest.mark.asyncio
    async def test_channel_send(self):
        """Test send stores message."""
        mock_agent = MagicMock()
        ch = _TestChannel("test", mock_agent)
        await ch.send("Hello, World!")
        assert len(ch._sent_messages) == 1
        assert ch._sent_messages[0] == "Hello, World!"
        # Clean up
        if ch.name in Channel._registry:
            del Channel._registry[ch.name]


class TestChannelFilterToggles:
    """Test enable/disable reasoning, tool_calls, tool_results."""

    def setup_method(self):
        self.agent = MagicMock()

    def test_reasoning_default_on(self):
        ch = _TestChannel("t", self.agent)
        assert ch._show_reasoning is True

    def test_tool_calls_default_on(self):
        ch = _TestChannel("t", self.agent)
        assert ch._show_tool_calls is True

    def test_tool_results_default_on(self):
        ch = _TestChannel("t", self.agent)
        assert ch._show_tool_results is True

    def test_enable_reasoning_off(self):
        ch = _TestChannel("t", self.agent)
        ch.enable_reasoning(False)
        assert ch._show_reasoning is False

    def test_enable_tool_calls_off(self):
        ch = _TestChannel("t", self.agent)
        ch.enable_tool_calls(False)
        assert ch._show_tool_calls is False

    def test_enable_tool_results_off(self):
        ch = _TestChannel("t", self.agent)
        ch.enable_tool_results(False)
        assert ch._show_tool_results is False


class TestChannelSessionSubscription:
    """Test subscribe_to_session and unsubscribe_from_session."""

    def setup_method(self):
        self.agent = MagicMock()
        self.agent._sessions = {}
        self.session_uuid = uuid.uuid4()
        self.session_mock = MagicMock()
        self.session_mock.subscribe.return_value = True
        self.session_mock.unsubscribe.return_value = True
        self.agent.get_session = MagicMock(return_value=self.session_mock)

    def test_subscribe_to_valid_session(self):
        self.agent._sessions[self.session_uuid] = self.session_mock
        ch = _TestChannel("t", self.agent)
        result = ch.subscribe_to_session(self.session_uuid)
        assert result is True
        assert ch._session_uuid == self.session_uuid
        self.session_mock.subscribe.assert_called_once_with(ch)

    def test_subscribe_to_invalid_session_returns_false(self):
        ch = _TestChannel("t", self.agent)
        fake_uuid = uuid.uuid4()
        result = ch.subscribe_to_session(fake_uuid)
        assert result is False

    def test_subscribe_to_different_session_returns_false(self):
        ch = _TestChannel("t", self.agent)
        self.agent._sessions[self.session_uuid] = self.session_mock
        ch.subscribe_to_session(self.session_uuid)
        new_uuid = uuid.uuid4()
        self.agent._sessions[new_uuid] = MagicMock()
        result = ch.subscribe_to_session(new_uuid)
        assert result is False

    def test_unsubscribe_from_valid_session(self):
        self.agent._sessions[self.session_uuid] = self.session_mock
        ch = _TestChannel("t", self.agent)
        ch.subscribe_to_session(self.session_uuid)
        result = ch.unsubscribe_from_session(self.session_uuid)
        assert result is True
        assert ch._session_uuid is None
        self.session_mock.unsubscribe.assert_called_once_with(ch)

    def test_unsubscribe_from_invalid_session_returns_false(self):
        ch = _TestChannel("t", self.agent)
        fake_uuid = uuid.uuid4()
        result = ch.unsubscribe_from_session(fake_uuid)
        assert result is False


class TestChannelRunLoop:
    """Test Channel.run() notification loop with filtering."""

    @pytest.mark.asyncio
    async def test_run_delivers_plain_message(self):
        agent = MagicMock()
        ch = _TestChannel("t", agent)
        test_msg = Message(role="user", content=[ContentPart(part_type="text", text="hi")])

        ch._wait = AsyncMock(side_effect=[test_msg, None])
        ch._running = True
        ch._session_uuid = None

        await ch.run()

        assert len(ch._sent_messages) == 1

    @pytest.mark.asyncio
    async def test_run_delivers_notification_event(self):
        agent = MagicMock()
        ch = _TestChannel("t", agent)
        test_msg = Message(role="user", content=[ContentPart(part_type="text", text="hi")])
        event = NotificationEvent(session_uuid=uuid.uuid4(), message=test_msg)

        ch._wait = AsyncMock(side_effect=[event, None])
        ch._running = True
        ch._session_uuid = event.session_uuid

        await ch.run()

        assert len(ch._sent_messages) == 1

    @pytest.mark.asyncio
    async def test_run_skips_reasoning_when_disabled(self):
        agent = MagicMock()
        ch = _TestChannel("t", agent)
        reasoning_msg = Message(role="reasoning", content=[ContentPart(part_type="reasoning", text="thinking")])
        user_msg = Message(role="user", content=[ContentPart(part_type="text", text="hi")])

        ch._wait = AsyncMock(side_effect=[reasoning_msg, user_msg])
        ch._running = True
        ch._show_reasoning = False
        ch._session_uuid = None

        await ch.run()

        # Only the non-reasoning message delivered
        assert len(ch._sent_messages) == 1

    @pytest.mark.asyncio
    async def test_run_skips_tool_when_disabled(self):
        agent = MagicMock()
        ch = _TestChannel("t", agent)
        tool_msg = Message(role="tool", content=[])
        user_msg = Message(role="user", content=[ContentPart(part_type="text", text="hi")])

        ch._wait = AsyncMock(side_effect=[tool_msg, user_msg])
        ch._running = True
        ch._show_tool_calls = False
        ch._session_uuid = None

        await ch.run()

        assert len(ch._sent_messages) == 1

    @pytest.mark.asyncio
    async def test_run_skips_tool_result_when_disabled(self):
        agent = MagicMock()
        ch = _TestChannel("t", agent)
        tool_result_msg = Message(role="tool_result", content=[])
        user_msg = Message(role="user", content=[ContentPart(part_type="text", text="hi")])

        ch._wait = AsyncMock(side_effect=[tool_result_msg, user_msg])
        ch._running = True
        ch._show_tool_results = False
        ch._session_uuid = None

        await ch.run()

        assert len(ch._sent_messages) == 1

    @pytest.mark.asyncio
    async def test_run_stops_on_none_event(self):
        agent = MagicMock()
        ch = _TestChannel("t", agent)
        ch._wait = AsyncMock(return_value=None)
        ch._running = True

        await ch.run()

        assert ch._running is False

    @pytest.mark.asyncio
    async def test_run_stops_on_none_message(self):
        agent = MagicMock()
        ch = _TestChannel("t", agent)
        ch._wait = AsyncMock(return_value=Message(role="user", content=[]))
        ch._running = True
        ch.send = AsyncMock()

        await ch.run()

        assert ch._running is False
