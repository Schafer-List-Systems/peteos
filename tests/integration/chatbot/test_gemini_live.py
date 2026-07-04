"""Integration tests for Gemini ChatBot — live backend.

Tests every GeminiChatBot API surface and operation mode with a real Gemini backend.
Run with:
    CHATBOT_TEST_BACKEND_URL=https://generativelanguage.googleapis.com \\
    CHATBOT_TEST_BACKEND_NAME=gemini \\
    CHATBOT_TEST_API_KEY=your-gemini-key \\
    CHATBOT_TEST_MODEL=gemini-2.0-flash \\
    pytest tests/integration/chatbot/test_gemini_live.py -v -m chatbot_integration
"""

import os

import pytest

skip_reason = "Set CHATBOT_TEST_BACKEND_URL, CHATBOT_TEST_BACKEND_NAME, CHATBOT_TEST_API_KEY, CHATBOT_TEST_MODEL to run"


def _skip_if_not_configured():
    """Skip the test if env vars are not set."""
    required = [
        "CHATBOT_TEST_BACKEND_URL",
        "CHATBOT_TEST_BACKEND_NAME",
        "CHATBOT_TEST_API_KEY",
        "CHATBOT_TEST_MODEL",
    ]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        pytest.skip(f"Missing env vars: {', '.join(missing)}")


class TestGeminiLiveStreaming:
    """Test live Gemini ChatBot streaming mode."""

    @pytest.mark.asyncio
    async def test_send_context_streaming_text(self, live_gemini_chatbot):
        """Send a text-only context in streaming mode and verify response."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        user_msg = Message.create("user", [ContentPart.create_text("Say hello")])
        ctx = Context.create()
        ctx.append(user_msg)

        response = await live_gemini_chatbot.send_context(ctx, streaming=True)
        chunks = []
        async for chunk in response:
            chunks.append(chunk)

        assert "role" in response.data
        assert response.data["role"] == "model"
        assert "content" in response.data
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_send_context_streaming_with_system_prompt(self, live_gemini_chatbot):
        """Send a context with system prompt in streaming mode."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        messages = [
            Message.create("system", [ContentPart.create_text("You are a translator. Always reply in uppercase.")]),
            Message.create("user", [ContentPart.create_text("hello")]),
        ]
        ctx = Context.create()
        for msg in messages:
            ctx.append(msg)

        response = await live_gemini_chatbot.send_context(ctx, streaming=True)
        async for _ in response:
            pass

        assert "role" in response.data
        assert "content" in response.data
        assert response.data["role"] == "model"

    @pytest.mark.asyncio
    async def test_send_context_streaming_with_temperature(self, live_gemini_chatbot):
        """Send a context with custom temperature in streaming mode."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        user_msg = Message.create("user", [ContentPart.create_text("Say exactly 'test'")])
        ctx = Context.create()
        ctx.append(user_msg)

        response = await live_gemini_chatbot.send_context(
            ctx,
            streaming=True,
            generation_config={"temperature": 0.0},
        )
        async for _ in response:
            pass

        assert "role" in response.data
        assert "content" in response.data

    @pytest.mark.asyncio
    async def test_send_context_streaming_multi_turn(self, live_gemini_chatbot):
        """Send a multi-turn conversation in streaming mode."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        messages = [
            Message.create("user", [ContentPart.create_text("First message")]),
            Message.create("assistant", [ContentPart.create_text("First response")]),
            Message.create("user", [ContentPart.create_text("Second message")]),
        ]
        ctx = Context.create()
        for msg in messages:
            ctx.append(msg)

        response = await live_gemini_chatbot.send_context(ctx, streaming=True)
        async for _ in response:
            pass

        assert "role" in response.data
        assert "content" in response.data

    @pytest.mark.asyncio
    async def test_send_context_streaming_empty_context(self, live_gemini_chatbot):
        """Empty context triggers an API error (Gemini expects at least one turn)."""
        _skip_if_not_configured()

        from peteos.conversation.context import Context

        ctx = Context.create()

        response = await live_gemini_chatbot.send_context(ctx, streaming=True)
        async for _ in response:
            pass

        # The LLM rejects empty contexts — error is captured in response.data
        assert "error" in response.data


class TestGeminiLiveNonStreaming:
    """Test live Gemini ChatBot non-streaming mode."""

    @pytest.mark.asyncio
    async def test_send_context_non_streaming_text(self, live_gemini_chatbot):
        """Send a text-only context in non-streaming mode."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        user_msg = Message.create("user", [ContentPart.create_text("Say hello")])
        ctx = Context.create()
        ctx.append(user_msg)

        response = await live_gemini_chatbot.send_context(ctx, streaming=False)
        async for _ in response:
            pass

        assert "role" in response.data
        assert response.data["role"] == "model"
        assert "content" in response.data

    @pytest.mark.asyncio
    async def test_send_context_non_streaming_with_system_prompt(self, live_gemini_chatbot):
        """Send a context with system prompt in non-streaming mode."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        messages = [
            Message.create("system", [ContentPart.create_text("You are helpful.")]),
            Message.create("user", [ContentPart.create_text("What day is today?")]),
        ]
        ctx = Context.create()
        for msg in messages:
            ctx.append(msg)

        response = await live_gemini_chatbot.send_context(ctx, streaming=False)
        async for _ in response:
            pass

        assert "role" in response.data
        assert "content" in response.data

    @pytest.mark.asyncio
    async def test_send_context_non_streaming_generation_config(self, live_gemini_chatbot):
        """Send a context with multiple generation config params."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        user_msg = Message.create("user", [ContentPart.create_text("Say 'test'")])
        ctx = Context.create()
        ctx.append(user_msg)

        response = await live_gemini_chatbot.send_context(
            ctx,
            streaming=False,
            generation_config={"temperature": 0.0, "max_tokens": 50},
        )
        async for _ in response:
            pass

        assert "role" in response.data
        assert "content" in response.data
