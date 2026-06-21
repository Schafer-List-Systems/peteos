"""Integration tests for Anthropic ChatBot — live backend.

Tests every AnthropicChatBot API surface and operation mode with a real LLM backend.
Run with:
    CHATBOT_TEST_BACKEND_URL=http://localhost:8000 \\
    CHATBOT_TEST_BACKEND_NAME=test \\
    CHATBOT_TEST_API_KEY=sk-ant-... \\
    CHATBOT_TEST_MODEL=claude-3-haiku \\
    pytest tests/integration/chatbot/test_anthropic_live.py -v -m chatbot_integration
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


class TestAnthropicLiveStreaming:
    """Test live Anthropic ChatBot streaming mode."""

    @pytest.mark.asyncio
    async def test_send_context_streaming_text(self, live_anthropic_chatbot):
        """Send a text-only context in streaming mode and verify response."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        user_msg = Message.create("user", [ContentPart.create_text("Say hello")])
        ctx = Context({})
        ctx.append(user_msg)

        response = await live_anthropic_chatbot.send_context(ctx, streaming=True)
        chunks = []
        async for chunk in response:
            chunks.append(chunk)

        assert "role" in response.data
        assert response.data["role"] == "assistant"
        assert "content" in response.data
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_send_context_streaming_with_thinking(self, live_anthropic_chatbot):
        """Send a context with thinking content in streaming mode."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        messages = [
            Message.create("system", [ContentPart.create_text("Think out loud and then answer.")]),
            Message.create("user", [ContentPart.create_text("What is 2+2? Answer in one sentence.")]),
        ]
        ctx = Context({})
        for msg in messages:
            ctx.append(msg)

        response = await live_anthropic_chatbot.send_context(ctx, streaming=True)
        chunks = []
        async for chunk in response:
            chunks.append(chunk)

        assert "role" in response.data
        assert response.data["role"] == "assistant"
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_send_context_streaming_with_tool_choice(self, live_anthropic_chatbot):
        """Send a context with tool_choice in streaming mode."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        user_msg = Message.create("user", [ContentPart.create_text("What is 1+1?")])
        ctx = Context({})
        ctx.append(user_msg)

        response = await live_anthropic_chatbot.send_context(
            ctx,
            streaming=True,
            generation_config={"tool_choice": {"type": "auto"}},
        )
        async for _ in response:
            pass

        assert "role" in response.data
        assert "content" in response.data

    @pytest.mark.asyncio
    async def test_send_context_streaming_with_temperature(self, live_anthropic_chatbot):
        """Send a context with custom temperature in streaming mode."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        user_msg = Message.create("user", [ContentPart.create_text("Say exactly 'test'")])
        ctx = Context({})
        ctx.append(user_msg)

        response = await live_anthropic_chatbot.send_context(
            ctx,
            streaming=True,
            generation_config={"temperature": 0.0},
        )
        async for _ in response:
            pass

        assert "role" in response.data
        assert "content" in response.data

    @pytest.mark.asyncio
    async def test_send_context_streaming_multi_turn(self, live_anthropic_chatbot):
        """Send a multi-turn conversation in streaming mode."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        messages = [
            Message.create("user", [ContentPart.create_text("First message")]),
            Message.create("assistant", [ContentPart.create_text("First response")]),
            Message.create("user", [ContentPart.create_text("Second message")]),
        ]
        ctx = Context({})
        for msg in messages:
            ctx.append(msg)

        response = await live_anthropic_chatbot.send_context(ctx, streaming=True)
        async for _ in response:
            pass

        assert "role" in response.data
        assert "content" in response.data

    @pytest.mark.asyncio
    async def test_send_context_streaming_empty_context(self, live_anthropic_chatbot):
        """Empty context triggers an API error (LLM expects at least one message)."""
        _skip_if_not_configured()

        from peteos.conversation.context import Context

        ctx = Context({})

        response = await live_anthropic_chatbot.send_context(ctx, streaming=True)
        async for _ in response:
            pass

        # The LLM rejects empty contexts — error is captured in response.data
        assert "error" in response.data

    @pytest.mark.asyncio
    async def test_send_context_streaming_system_prompt(self, live_anthropic_chatbot):
        """Send a context with system prompt in streaming mode."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        messages = [
            Message.create("system", [ContentPart.create_text("You are a translator. Always reply in uppercase.")]),
            Message.create("user", [ContentPart.create_text("hello")]),
        ]
        ctx = Context({})
        for msg in messages:
            ctx.append(msg)

        response = await live_anthropic_chatbot.send_context(ctx, streaming=True)
        async for _ in response:
            pass

        assert "role" in response.data
        assert "content" in response.data
        assert response.data["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_send_context_streaming_with_stop_reason(self, live_anthropic_chatbot):
        """Send a context that completes and verify stop_reason is captured."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        user_msg = Message.create("user", [ContentPart.create_text("Say hi")])
        ctx = Context({})
        ctx.append(user_msg)

        response = await live_anthropic_chatbot.send_context(ctx, streaming=True)
        async for _ in response:
            pass

        # Anthropic always sets a stop_reason (end_turn or tool_use)
        assert "stop_reason" in response.data

    @pytest.mark.asyncio
    async def test_send_context_streaming_max_tokens_override(self, live_anthropic_chatbot):
        """Send a context with max_tokens override in streaming mode."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        user_msg = Message.create("user", [ContentPart.create_text("Say something brief")])
        ctx = Context({})
        ctx.append(user_msg)

        response = await live_anthropic_chatbot.send_context(
            ctx,
            streaming=True,
            generation_config={"max_tokens": 50},
        )
        async for _ in response:
            pass

        assert "role" in response.data
        assert "content" in response.data


class TestAnthropicLiveNonStreaming:
    """Test live Anthropic ChatBot non-streaming mode."""

    @pytest.mark.asyncio
    async def test_send_context_non_streaming_text(self, live_anthropic_chatbot):
        """Send a text-only context in non-streaming mode."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        user_msg = Message.create("user", [ContentPart.create_text("Say hello")])
        ctx = Context({})
        ctx.append(user_msg)

        response = await live_anthropic_chatbot.send_context(ctx, streaming=False)
        async for _ in response:
            pass

        assert "role" in response.data
        assert response.data["role"] == "assistant"
        assert "content" in response.data
        # Qwen may return thinking as the first block instead of text
        assert response.data["content"][0]["type"] in ("text", "thinking")

    @pytest.mark.asyncio
    async def test_send_context_non_streaming_with_system_prompt(self, live_anthropic_chatbot):
        """Send a context with system prompt in non-streaming mode."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        messages = [
            Message.create("system", [ContentPart.create_text("You are helpful.")]),
            Message.create("user", [ContentPart.create_text("What day is today?")]),
        ]
        ctx = Context({})
        for msg in messages:
            ctx.append(msg)

        response = await live_anthropic_chatbot.send_context(ctx, streaming=False)
        async for _ in response:
            pass

        assert "role" in response.data
        assert "content" in response.data

    @pytest.mark.asyncio
    async def test_send_context_non_streaming_tool_choice(self, live_anthropic_chatbot):
        """Send a context with tool_choice in non-streaming mode."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        user_msg = Message.create("user", [ContentPart.create_text("What is 1+1?")])
        ctx = Context({})
        ctx.append(user_msg)

        response = await live_anthropic_chatbot.send_context(
            ctx,
            streaming=False,
            generation_config={"tool_choice": {"type": "auto"}},
        )
        async for _ in response:
            pass

        assert "role" in response.data
        assert "content" in response.data

    @pytest.mark.asyncio
    async def test_send_context_non_streaming_generation_config(self, live_anthropic_chatbot):
        """Send a context with multiple generation config params."""
        _skip_if_not_configured()

        from peteos.conversation.message import Message, ContentPart
        from peteos.conversation.context import Context

        user_msg = Message.create("user", [ContentPart.create_text("Say 'test'")])
        ctx = Context({})
        ctx.append(user_msg)

        response = await live_anthropic_chatbot.send_context(
            ctx,
            streaming=False,
            generation_config={"temperature": 0.0, "max_tokens": 50},
        )
        async for _ in response:
            pass

        assert "role" in response.data
        assert "content" in response.data


class TestAnthropicLiveModelProperties:
    """Test live Anthropic ChatBot model property access."""

    @pytest.mark.asyncio
    async def test_model_property_returns_config_model(self, live_anthropic_chatbot, chatbot_config):
        """Model property returns the configured model name."""
        _skip_if_not_configured()

        assert live_anthropic_chatbot.model == chatbot_config["model"]

    @pytest.mark.asyncio
    async def test_model_setter_changes_config(self, live_anthropic_chatbot):
        """Model setter updates the config."""
        _skip_if_not_configured()

        old_model = live_anthropic_chatbot.model
        live_anthropic_chatbot.model = "test-model-override"
        assert live_anthropic_chatbot._config.model == "test-model-override"
        live_anthropic_chatbot.model = old_model

    @pytest.mark.asyncio
    async def test_list_available_models(self, live_anthropic_chatbot):
        """list_available_models returns at least the configured model."""
        _skip_if_not_configured()

        models = live_anthropic_chatbot.list_available_models()
        assert isinstance(models, list)
        assert len(models) >= 1
        assert live_anthropic_chatbot.model in models
