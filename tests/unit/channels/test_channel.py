"""Unit tests for Channel base class."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from peteos.engine.channel import Channel
from peteos.engine.events import MessageEvent
from peteos.chatbot import Message, ContentPart


class _TestChannel(Channel):
    """Test implementation of abstract Channel."""

    def __init__(self):
        super().__init__()
        self._sent_messages: list[Message] = []

    async def on_outgoing(self, message: Message) -> None:
        self._sent_messages.append(message)


class TestChannelAttachDetach:
    """Test channel attach and detach lifecycle."""

    @pytest.mark.asyncio
    async def test_attach_subscribes_runner(self):
        """Test that attach subscribes the channel to the runner."""
        mock_runner = MagicMock()
        ch = _TestChannel()
        await ch.attach(mock_runner)
        mock_runner.subscribe.assert_called_once_with(ch)
        assert ch._runner is mock_runner

    @pytest.mark.asyncio
    async def test_attach_with_greeting_queues_message(self):
        """Test that attach with greeting queues a greeting message."""
        mock_runner = MagicMock()
        mock_runner.queue_message = AsyncMock()
        ch = _TestChannel()
        await ch.attach(mock_runner, greeting="Hello!")
        mock_runner.queue_message.assert_awaited_once()
        msg = mock_runner.queue_message.call_args[0][0]
        assert msg.role == "assistant"
        assert msg.content[0].text == "Hello!"

    @pytest.mark.asyncio
    async def test_detach_unsubscribes_runner(self):
        """Test that detach unsubscribes the channel from the runner."""
        mock_runner = MagicMock()
        ch = _TestChannel()
        await ch.attach(mock_runner)
        await ch.detach()
        mock_runner.unsubscribe.assert_called_once_with(ch)
        assert ch._runner is None

    @pytest.mark.asyncio
    async def test_detach_with_farewell_queues_message(self):
        """Test that detach with farewell queues a farewell message."""
        mock_runner = MagicMock()
        mock_runner.queue_message = AsyncMock()
        ch = _TestChannel()
        await ch.attach(mock_runner)
        await ch.detach(farewell="Goodbye!")
        mock_runner.queue_message.assert_awaited_once()
        msg = mock_runner.queue_message.call_args[0][0]
        assert msg.role == "assistant"
        assert msg.content[0].text == "Goodbye!"


class TestChannelNotify:
    """Test _notify internal handler."""

    @pytest.mark.asyncio
    async def test_notify_calls_on_outgoing(self):
        """Test that _notify calls on_outgoing with the message."""
        ch = _TestChannel()
        test_msg = Message.create(
            role="user",
            content_parts=[ContentPart.create_text("hi")],
        )
        event = MessageEvent(message=test_msg)
        await ch._notify(event)
        assert len(ch._sent_messages) == 1
        assert ch._sent_messages[0] == MessageEvent(message=test_msg)

    @pytest.mark.asyncio
    async def test_notify_skips_disabled_role(self):
        """Test that _notify skips messages with a disabled role."""
        ch = _TestChannel()
        ch.enable_role("reasoning", on=False)
        reasoning_msg = Message.create(
            role="reasoning",
            content_parts=[ContentPart.create_thinking("thinking")],
        )
        user_msg = Message.create(
            role="user",
            content_parts=[ContentPart.create_text("hi")],
        )
        await ch._notify(MessageEvent(message=reasoning_msg))
        await ch._notify(MessageEvent(message=user_msg))
        assert len(ch._sent_messages) == 1
        assert ch._sent_messages[0] == MessageEvent(message=user_msg)

    @pytest.mark.asyncio
    async def test_notify_skips_disabled_content_type(self):
        """Test that _notify skips messages with a disabled content type."""
        ch = _TestChannel()
        ch.enable_content_type("tool_use", on=False)
        tool_msg = Message.create(
            role="assistant",
            content_parts=[
                ContentPart.create_tool_use("c1", "test_tool", "{}"),
            ],
        )
        user_msg = Message.create(
            role="user",
            content_parts=[ContentPart.create_text("hi")],
        )
        await ch._notify(MessageEvent(message=tool_msg))
        await ch._notify(MessageEvent(message=user_msg))
        assert len(ch._sent_messages) == 1
        assert ch._sent_messages[0] == MessageEvent(message=user_msg)

    @pytest.mark.asyncio
    async def test_notify_raises_on_unknown_event_type(self):
        """Test that _notify raises TypeError for unknown event types."""
        ch = _TestChannel()
        not_an_event = "not an event"
        with pytest.raises(TypeError, match="Unknown event type"):
            await ch._notify(not_an_event)
        assert len(ch._sent_messages) == 0

    @pytest.mark.asyncio
    async def test_notify_does_nothing_on_none_message(self):
        """Test that _notify ignores MessageEvent with None message."""
        ch = _TestChannel()
        await ch._notify(MessageEvent(message=None))
        assert len(ch._sent_messages) == 0


class TestChannelEnqueue:
    """Test enqueue method."""

    @pytest.mark.asyncio
    async def test_enqueue_calls_runner_queue_message(self):
        """Test that enqueue calls runner.queue_message."""
        mock_runner = MagicMock()
        mock_runner.queue_message = AsyncMock()
        ch = _TestChannel()
        await ch.attach(mock_runner)
        test_msg = Message.create(
            role="user",
            content_parts=[ContentPart.create_text("hello")],
        )
        await ch.enqueue(test_msg)
        mock_runner.queue_message.assert_awaited_once_with(test_msg)

    @pytest.mark.asyncio
    async def test_enqueue_raises_when_not_attached(self):
        """Test that enqueue raises RuntimeError when not attached."""
        ch = _TestChannel()
        test_msg = Message.create(
            role="user",
            content_parts=[ContentPart.create_text("hello")],
        )
        with pytest.raises(RuntimeError, match="not attached"):
            await ch.enqueue(test_msg)


class TestChannelFilters:
    """Test enable_role and enable_content_type filters."""

    def test_role_filter_default_all_enabled(self):
        """Test that all roles are enabled by default."""
        ch = _TestChannel()
        assert "reasoning" not in ch._disabled_roles
        assert "tool" not in ch._disabled_roles
        assert "tool_result" not in ch._disabled_roles

    def test_enable_role_disable(self):
        """Test that enable_role can disable a role."""
        ch = _TestChannel()
        ch.enable_role("reasoning", on=False)
        assert "reasoning" in ch._disabled_roles

    def test_enable_role_re_enable(self):
        """Test that enable_role can re-enable a role."""
        ch = _TestChannel()
        ch.enable_role("reasoning", on=False)
        ch.enable_role("reasoning", on=True)
        assert "reasoning" not in ch._disabled_roles

    def test_content_type_filter_default_all_enabled(self):
        """Test that all content types are enabled by default."""
        ch = _TestChannel()
        assert "text" not in ch._disabled_content_types
        assert "tool_use" not in ch._disabled_content_types

    def test_enable_content_type_disable(self):
        """Test that enable_content_type can disable a type."""
        ch = _TestChannel()
        ch.enable_content_type("tool_use", on=False)
        assert "tool_use" in ch._disabled_content_types

    def test_enable_content_type_re_enable(self):
        """Test that enable_content_type can re-enable a type."""
        ch = _TestChannel()
        ch.enable_content_type("tool_use", on=False)
        ch.enable_content_type("tool_use", on=True)
        assert "tool_use" not in ch._disabled_content_types
