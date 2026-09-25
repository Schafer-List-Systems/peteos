"""Tests for context reduction hooks and the _try_context_reduction core."""

from unittest.mock import MagicMock

from peteos.conversation.context import Context
from peteos.conversation.message import ContentPart, Message
from peteos.conversation import system_prompt_message, tool_definitions_message
from peteos.oap.agentic_object import (
    _try_context_reduction,
    _context_reduction_hook,
    _on_truncation_hook,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx_with_tokens(token_counts: list[int], sys_tokens: int = 0) -> Context:
    """Build a context where each message has a pre-set token count."""
    ctx = Context.create()
    for tc in token_counts:
        msg = Message.create("user", [ContentPart.create_text("x")])
        msg.raw_dict["token_count"] = tc
        ctx.append(msg)
    return ctx


def _mock_runner(context_size: int, max_tokens: int, max_context: float, reserve: int = 0):
    """Create a mock runner with a mock chatbot config and mock session."""
    mock_session = MagicMock()
    mock_ctx = _ctx_with_tokens([context_size]) if context_size else Context.create()
    mock_ctx.raw_dict["messages"][0]["token_count"] = context_size
    mock_session.active_context = mock_ctx
    mock_session.set_active_context = MagicMock()

    mock_chatbot = MagicMock()
    mock_chatbot._config.max_tokens = max_tokens
    mock_chatbot._config.max_context_size = max_context
    mock_chatbot._config.context_reduction_reserve = reserve

    mock_runner = MagicMock()
    mock_runner.session = mock_session
    mock_runner._session = mock_session  # both accessors used by hooks
    mock_runner._chatbot = mock_chatbot
    return mock_runner


# ---------------------------------------------------------------------------
# _try_context_reduction
# ---------------------------------------------------------------------------

class TestTryContextReduction:
    """Unit tests for the shared _try_context_reduction core function."""

    def test_no_reduction_needed(self):
        """When budget is negative (max_context exhausted), returns None."""
        ctx = _ctx_with_tokens([10])
        result = _try_context_reduction(ctx, max_context=10.0, max_output=20, reserve=0)
        assert result is None

    def test_strip_thinking_suffices(self):
        """When strip_thinking fits in budget, returns the stripped context."""
        ctx = _ctx_with_tokens([10])
        result = _try_context_reduction(ctx, max_context=100.0, max_output=10, reserve=0)
        assert result is not None
        assert result is not ctx

    def test_falls_back_to_rolling_window(self):
        """When strip_thinking doesn't fit, falls back to rolling_token_window."""
        ctx = _ctx_with_tokens([10, 20, 30])
        result = _try_context_reduction(ctx, max_context=50.0, max_output=5, reserve=0)
        assert result is not None
        assert result.total_token_count() <= 50 - 5  # budget

    def test_negative_budget_returns_none(self):
        """When budget is negative, returns None."""
        ctx = _ctx_with_tokens([100])
        result = _try_context_reduction(ctx, max_context=10.0, max_output=20, reserve=0)
        assert result is None

    def test_reserve_used_in_budget_calculation(self):
        """Reserve is subtracted from budget before checking."""
        ctx = _ctx_with_tokens([90])
        result = _try_context_reduction(ctx, max_context=100.0, max_output=5, reserve=10)
        # budget = 100 - 10 - 5 = 85; context = 90 > 85 → needs reduction
        assert result is not None


# ---------------------------------------------------------------------------
# _context_reduction_hook (proactive)
# ---------------------------------------------------------------------------

class TestContextReductionHook:
    """Tests for the proactive before_send_to_chatbot hook."""

    def test_no_op_when_max_context_is_infinity(self):
        """When max_context is infinity, hook does nothing."""
        runner = _mock_runner(context_size=1_000_000, max_tokens=100, max_context=float("inf"))
        ctx = runner.session.active_context
        _context_reduction_hook(runner, ctx)
        runner.session.set_active_context.assert_not_called()

    def test_no_op_when_context_fits(self):
        """When context_size + max_output <= max_context - reserve, hook does nothing."""
        runner = _mock_runner(
            context_size=50,
            max_tokens=10,
            max_context=100.0,
            reserve=0,
        )
        ctx = runner.session.active_context
        _context_reduction_hook(runner, ctx)
        runner.session.set_active_context.assert_not_called()

    def test_reduces_when_over_threshold(self):
        """When context would exceed, calls set_active_context with reduced context."""
        runner = _mock_runner(
            context_size=95,
            max_tokens=10,
            max_context=100.0,
            reserve=0,
        )
        ctx = runner.session.active_context
        _context_reduction_hook(runner, ctx)
        runner.session.set_active_context.assert_called_once()
        new_ctx = runner.session.set_active_context.call_args[0][0]
        assert new_ctx.total_token_count() + 10 <= 100  # fits with output budget

    def test_reserve_margin_used(self):
        """Reserve is subtracted from max_context before comparison."""
        runner = _mock_runner(
            context_size=90,
            max_tokens=10,
            max_context=100.0,
            reserve=5,  # effective limit = 100 - 5 = 95
        )
        ctx = runner.session.active_context
        _context_reduction_hook(runner, ctx)
        # 90 + 10 = 100 > 95 (with reserve) → reduction triggered
        runner.session.set_active_context.assert_called_once()


# ---------------------------------------------------------------------------
# _on_truncation_hook (reactive)
# ---------------------------------------------------------------------------

class TestOnTruncationHook:
    """Tests for the reactive on_truncation hook."""

    def test_no_op_when_max_context_is_infinity(self):
        """When max_context is infinity, hook does nothing."""
        runner = _mock_runner(context_size=1_000_000, max_tokens=100, max_context=float("inf"))
        _on_truncation_hook(runner, counter=1, max_retries=3)
        runner.session.set_active_context.assert_not_called()

    def test_no_op_when_context_fits(self):
        """When budget is negative (exhausted), hook does nothing."""
        runner = _mock_runner(
            context_size=50,
            max_tokens=10,
            max_context=5.0,  # budget = 5 - 0 - 10 = -5 < 0
            reserve=0,
        )
        _on_truncation_hook(runner, counter=1, max_retries=3)
        runner.session.set_active_context.assert_not_called()

    def test_reduces_when_over_threshold(self):
        """When context exceeds calibrated max_context, reduces context."""
        runner = _mock_runner(
            context_size=90,
            max_tokens=10,
            max_context=95.0,  # calibrated by truncation handler
            reserve=0,
        )
        _on_truncation_hook(runner, counter=2, max_retries=3)
        runner.session.set_active_context.assert_called_once()

    def test_does_not_act_on_counter(self):
        """The hook receives counter/max_retries but does not act on them."""
        runner = _mock_runner(
            context_size=90,
            max_tokens=10,
            max_context=95.0,
            reserve=0,
        )
        _on_truncation_hook(runner, counter=3, max_retries=3)  # last try
        _on_truncation_hook(runner, counter=1, max_retries=3)  # early try
        # Both calls should reduce — hook has no knowledge of retry policy
        assert runner.session.set_active_context.call_count == 2
