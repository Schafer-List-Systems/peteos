"""Tests for rolling_token_window on Context and Session."""

import pytest

from peteos.conversation.context import Context
from peteos.conversation.message import ContentPart, Message
from peteos.conversation.session import Session
from peteos.conversation.system_prompt_message import SystemPromptMessage
from peteos.conversation.tool_definitions_message import ToolDefinitionsMessage


class TestContextRollingTokenWindow:
    """Tests for Context.rolling_token_window."""

    def _build_ctx_with_tokens(
        self,
        non_special_tokens: list[int],
        sys_tokens: int = 0,
        tool_tokens: int = 0,
    ) -> Context:
        """Build a context with optional special messages and
        non-special messages whose token counts are given."""
        sys_msg = SystemPromptMessage.create("You are helpful.") if sys_tokens else None
        tools_msg = ToolDefinitionsMessage() if tool_tokens else None

        if sys_msg:
            sys_msg.raw_dict["token_count"] = sys_tokens
        if tools_msg:
            tools_msg.raw_dict["token_count"] = tool_tokens

        ctx = Context.create(
            system_prompt_message=sys_msg,
            tool_definitions_message=tools_msg,
        )

        for i, tc in enumerate(non_special_tokens):
            msg = Message.create(
                "user",
                [ContentPart.create_text(f"msg{i}")],
            )
            msg.raw_dict["token_count"] = tc
            ctx.append(msg)

        return ctx

    def _count_ctx_tokens(self, ctx: Context) -> int:
        """Sum token counts across all messages in ctx.messages."""
        return sum(msg.count_tokens() for msg in ctx.messages)

    def test_fork_when_already_under_max(self):
        ctx = self._build_ctx_with_tokens([10, 20, 30], sys_tokens=5, tool_tokens=5)
        total = self._count_ctx_tokens(ctx)
        result = ctx.rolling_token_window(100)
        assert result is not None
        assert self._count_ctx_tokens(result) == total
        assert len(result.messages) == len(ctx.messages)

    def test_discards_messages_to_fit_under_max(self):
        # special(5+5=10) + non-special(10+20+30+40=100) = 110
        ctx = self._build_ctx_with_tokens([10, 20, 30, 40], sys_tokens=5, tool_tokens=5)
        total = self._count_ctx_tokens(ctx)  # 110
        result = ctx.rolling_token_window(60)
        assert result is not None
        # Dropping from end: remove msg3(40) -> 70, still > 60
        # remove msg2(30) -> 40 <= 60, stop
        # Kept: 10 + 20 + special(10) = 40
        assert self._count_ctx_tokens(result) <= 60
        assert len(result.messages) == 4  # 2 non-special + 2 special

    def test_discards_only_one_message(self):
        # special(5+5=10) + non-special(10+20+30+40=100) = 110
        ctx = self._build_ctx_with_tokens([10, 20, 30, 40], sys_tokens=5, tool_tokens=5)
        result = ctx.rolling_token_window(80)
        assert result is not None
        # Remove msg3(40) -> 60 <= 80, stop
        assert self._count_ctx_tokens(result) == 60
        assert len(result.messages) == 4  # 3 non-special + special

    def test_empty_context_returns_fork(self):
        ctx = Context.create()
        result = ctx.rolling_token_window(100)
        assert result is not None
        assert len(result.messages) == 0

    def test_no_non_special_messages_only_special(self):
        ctx = self._build_ctx_with_tokens([], sys_tokens=5, tool_tokens=5)
        result = ctx.rolling_token_window(100)
        assert result is not None
        assert self._count_ctx_tokens(result) == 10
        assert len(result.messages) == 2

    def test_no_non_special_messages_no_special(self):
        ctx = Context.create()
        result = ctx.rolling_token_window(100)
        assert result is not None
        assert len(result.messages) == 0


class TestSessionRollingTokenWindow:
    """Tests for Session rolling window forwarding."""

    def _make_session(self) -> Session:
        """Create a session with a context containing a few messages."""
        session = Session.create(parent_dir="/tmp/test_session")
        return session

    def _count_ctx_tokens(self, ctx: Context) -> int:
        return sum(msg.count_tokens() for msg in ctx.messages)

    def test_rolling_sequence_window_forwards_to_context(self):
        session = self._make_session()
        for i in range(5):
            msg = Message.create("user", [ContentPart.create_text(f"msg{i}")])
            session.active_context.append(msg)

        result = session.rolling_sequence_window(2)
        assert result is not None
        assert session.active_context is result
        assert len(result.messages) == 2

    def test_rolling_token_window_forwards_to_context(self):
        session = self._make_session()
        for tc in [10, 20, 30, 40]:
            msg = Message.create("user", [ContentPart.create_text("x")])
            msg.raw_dict["token_count"] = tc
            session.active_context.append(msg)

        result = session.rolling_token_window(55)
        assert result is not None
        assert session.active_context is result
        assert self._count_ctx_tokens(result) <= 55

    def test_no_active_context_returns_none(self):
        session = Session("/tmp/test_nocx")
        assert session.rolling_sequence_window(2) is None
        assert session.rolling_token_window(100) is None
