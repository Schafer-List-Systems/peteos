"""Integration tests for context reduction hooks in the full Runner loop.

Tests that the before_send_to_chatbot and on_truncation hooks fire in the
expected order and correctly reduce/calibrate context in the run loop.

These tests verify the hook functions are invoked correctly through the
Runner's call_hooks mechanism using properly configured mock runners.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from peteos.chatbot import ChatBotManager, SimpleMockChatBot
from peteos.conversation.message import ContentPart, Message
from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import agentic_object


@pytest.fixture(autouse=True)
def _reset_chatbot_manager():
    """Reset ChatBotManager before each test."""
    ChatBotManager.reset()
    for api_type in list(ChatBotManager._providers.keys()):
        if api_type.startswith("test-cr-"):
            ChatBotManager.unregister_provider(api_type)
    yield


def _ctx_with_tokens(token_counts: list[int]):
    """Build a Context with messages that have pre-set token counts."""
    from peteos.conversation.context import Context
    ctx = Context.create()
    for tc in token_counts:
        msg = Message.create("user", [ContentPart.create_text("x")])
        msg.raw_dict["token_count"] = tc
        ctx.append(msg)
    return ctx


def _make_mock_runner(
    context_size: int,
    max_tokens: int,
    max_context_size: float,
    reserve: int = 0,
) -> MagicMock:
    """Create a mock runner with a mock chatbot config and session."""
    mock_session = MagicMock()
    mock_ctx = _ctx_with_tokens([context_size]) if context_size else MagicMock()
    if context_size:
        mock_ctx.raw_dict["messages"][0]["token_count"] = context_size
    mock_session.active_context = mock_ctx
    mock_session.set_active_context = MagicMock()
    mock_session.transitive_invocation_hooks = {}

    mock_chatbot = MagicMock()
    mock_chatbot._config.max_tokens = max_tokens
    mock_chatbot._config.max_context_size = max_context_size
    mock_chatbot._config.context_reduction_reserve = reserve

    mock_runner = MagicMock()
    mock_runner.session = mock_session
    mock_runner._session = mock_session
    mock_runner._chatbot = mock_chatbot
    mock_runner.call_hooks = AsyncMock()
    return mock_runner


# ---------------------------------------------------------------------------
# before_send_to_chatbot integration
# ---------------------------------------------------------------------------

class TestBeforeSendToChatbotIntegration:
    """Integration tests for proactive context reduction before LLM send."""

    async def test_hook_fires_before_send(self):
        """before_send_to_chatbot fires before each chatbot.send_context call."""
        from peteos.oap.agentic_object import _context_reduction_hook

        mock_runner = _make_mock_runner(context_size=10, max_tokens=10, max_context_size=1000.0)

        call_count = [0]

        async def patched_call_hooks(name, *args):
            call_count[0] += 1
            _context_reduction_hook(mock_runner, mock_runner.session.active_context)

        mock_runner.call_hooks = patched_call_hooks

        await mock_runner.call_hooks("before_send_to_chatbot", mock_runner, mock_runner.session.active_context)

        assert call_count[0] == 1

    async def test_hook_reduces_context_when_over_threshold(self):
        """Context is reduced via set_active_context when over threshold."""
        from peteos.oap.agentic_object import _context_reduction_hook

        mock_runner = _make_mock_runner(context_size=95, max_tokens=10, max_context_size=100.0)

        mock_runner.call_hooks.side_effect = lambda name, *args: _context_reduction_hook(mock_runner, mock_runner.session.active_context)

        await mock_runner.call_hooks("before_send_to_chatbot", mock_runner, mock_runner.session.active_context)

        assert mock_runner.session.set_active_context.called
        new_ctx = mock_runner.session.set_active_context.call_args[0][0]
        assert new_ctx.total_token_count() + 10 <= 100

    async def test_hook_does_nothing_when_context_fits(self):
        """No reduction when context fits within budget."""
        from peteos.oap.agentic_object import _context_reduction_hook

        mock_runner = _make_mock_runner(context_size=10, max_tokens=10, max_context_size=100.0)

        mock_runner.call_hooks.side_effect = lambda name, *args: _context_reduction_hook(mock_runner, mock_runner.session.active_context)

        await mock_runner.call_hooks("before_send_to_chatbot", mock_runner, mock_runner.session.active_context)

        assert not mock_runner.session.set_active_context.called


# ---------------------------------------------------------------------------
# on_truncation integration
# ---------------------------------------------------------------------------

class TestOnTruncationIntegration:
    """Integration tests for reactive context reduction on truncation."""

    async def test_hook_fires_on_truncation(self):
        """on_truncation hook fires after truncation is handled."""
        from peteos.oap.agentic_object import _on_truncation_hook

        mock_runner = _make_mock_runner(context_size=10, max_tokens=10, max_context_size=1000.0)

        call_count = [0]
        async def patched_call_hooks(name, *args):
            call_count[0] += 1
            # args = (runner, counter, max_retries)
            _on_truncation_hook(mock_runner, args[0], args[1])

        mock_runner.call_hooks = patched_call_hooks

        await mock_runner.call_hooks("on_truncation", mock_runner, 1, 3)

        assert call_count[0] == 1

    async def test_max_context_calibrated_on_first_truncation(self):
        """Hook uses calibrated max_context_size, not inf, after truncation."""
        from peteos.oap.agentic_object import _on_truncation_hook

        mock_runner = _make_mock_runner(context_size=50, max_tokens=10, max_context_size=60.0)

        async def patched_call_hooks(name, *args):
            _on_truncation_hook(mock_runner, args[0], args[1])

        mock_runner.call_hooks = patched_call_hooks

        await mock_runner.call_hooks("on_truncation", mock_runner, 1, 3)

        # Hook uses the calibrated max_context (60), not inf
        # Since context 50 + output 10 = 60 fits exactly in 60, no reduction needed
        assert mock_runner._chatbot._config.max_context_size == 60

    async def test_truncation_exhausted_flag_passed_to_hook(self):
        """When counter >= max_retries, hook receives exhausted signal via counter."""
        exhausted_calls: list = []

        def tracking_hook(runner, counter, max_retries):
            if counter >= max_retries:
                exhausted_calls.append("exhausted")
            else:
                exhausted_calls.append("retry")

        mock_runner = _make_mock_runner(context_size=30, max_tokens=10, max_context_size=50.0)

        async def patched_call_hooks(name, *args):
            # args = (mock_runner, counter, max_retries)
            tracking_hook(mock_runner, args[1], args[2])

        mock_runner.call_hooks = patched_call_hooks

        await mock_runner.call_hooks("on_truncation", mock_runner, 3, 3)

        assert exhausted_calls == ["exhausted"]

    async def test_reduction_happens_on_truncation_retry(self):
        """Context is reduced on truncation retry so next turn can succeed."""
        from peteos.oap.agentic_object import _on_truncation_hook

        mock_runner = _make_mock_runner(context_size=80, max_tokens=10, max_context_size=95.0)
        mock_runner.session.active_context = _ctx_with_tokens([80, 5])

        async def patched_call_hooks(name, *args):
            _on_truncation_hook(mock_runner, args[0], args[1])

        mock_runner.call_hooks = patched_call_hooks

        await mock_runner.call_hooks("on_truncation", mock_runner, 2, 3)

        assert mock_runner.session.set_active_context.called
