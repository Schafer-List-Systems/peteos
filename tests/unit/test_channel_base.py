"""Unit tests for Channel base class."""

import uuid
from unittest.mock import MagicMock

import pytest

from peteos.channel import Channel


# Helper for testing abstract Channel (not collected by pytest)
class _TestChannel(Channel):
    """Test implementation of abstract Channel."""

    def __init__(self, name: str, agent):
        """Initialize test channel."""
        super().__init__(name, agent)
        self._sent_messages = []

    def send(self, message: str) -> None:
        """Send message by storing it."""
        self._sent_messages.append(message)

    def receive(self) -> str | None:
        """Return None for testing."""
        return None


class TestChannelRegistry:
    """Test channel registry functionality."""

    def test_channel_registry_adds_channels(self):
        """Test that channels are added to registry on creation."""
        mock_agent = MagicMock()
        ch = _TestChannel("test1", mock_agent)
        assert ch.name in Channel._registry
        assert Channel._registry[ch.name] == ch
        # Clean up
        if ch.name in Channel._registry:
            del Channel._registry[ch.name]

    def test_channel_registry_overwrites_duplicate_name(self):
        """Test that registering a channel with same name overwrites."""
        mock_agent = MagicMock()
        ch1 = _TestChannel("test1", mock_agent)
        ch2 = _TestChannel("test1", mock_agent)
        assert ch2 in Channel._registry.values()
        assert ch1 not in Channel._registry.values()
        # Clean up
        if "test1" in Channel._registry:
            del Channel._registry["test1"]

    def test_channel_get_by_name(self):
        """Test get_by_name retrieves correct channel."""
        mock_agent = MagicMock()
        ch1 = _TestChannel("test1", mock_agent)
        ch2 = _TestChannel("test2", mock_agent)
        assert Channel.get_by_name("test1") == ch1
        assert Channel.get_by_name("test2") == ch2
        assert Channel.get_by_name("nonexistent") is None
        # Clean up
        for name in ["test1", "test2"]:
            if name in Channel._registry:
                del Channel._registry[name]

    def test_channel_list_all(self):
        """Test list_all returns all registered channels."""
        mock_agent = MagicMock()
        ch1 = _TestChannel("test1", mock_agent)
        ch2 = _TestChannel("test2", mock_agent)
        channels = Channel.list_all()
        assert len(channels) == 2
        assert "test1" in channels
        assert "test2" in channels
        # Clean up
        for name in ["test1", "test2"]:
            if name in Channel._registry:
                del Channel._registry[name]

    def test_channel_deregister_all(self):
        """Test deregister_all removes all channels."""
        mock_agent = MagicMock()
        ch1 = _TestChannel("test1", mock_agent)
        ch2 = _TestChannel("test2", mock_agent)
        Channel.deregister_all()
        assert len(Channel._registry) == 0


class TestChannelMethods:
    """Test Channel instance methods."""

    def test_channel_select_session(self):
        """Test select_session sets active session."""
        mock_agent = MagicMock()
        ch = _TestChannel("test", mock_agent)
        session_uuid = uuid.UUID("810fb120-e4e5-4e32-9718-88bbcaf7641a")
        ch.select_session(session_uuid)
        assert ch.active_session_uuid == session_uuid
        # Clean up
        if ch.name in Channel._registry:
            del Channel._registry[ch.name]

    def test_channel_send(self):
        """Test send stores message."""
        mock_agent = MagicMock()
        ch = _TestChannel("test", mock_agent)
        ch.send("Hello, World!")
        assert len(ch._sent_messages) == 1
        assert ch._sent_messages[0] == "Hello, World!"
        # Clean up
        if ch.name in Channel._registry:
            del Channel._registry[ch.name]
