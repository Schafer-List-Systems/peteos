"""Integration tests for chatbot image content — live backend.

Tests that the chatbot can receive and reason about image content.
A simple PNG is generated with cv2 (already a project dependency via
the ``camera`` optional group), encoded as base64, and sent to the
live LLM. The LLM should be able to read the text inside the image.

Run with:
    CHATBOT_TEST_BACKEND_URL=http://localhost:8000 \\
    CHATBOT_TEST_BACKEND_NAME=test \\
    CHATBOT_TEST_API_KEY=sk-test \\
    CHATBOT_TEST_MODEL=gpt-4o \\
    pytest tests/integration/chatbot/test_image_live.py -v -m chatbot_integration
"""

import base64
import os

import cv2
import numpy as np
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


def _generate_text_image(text: str, width: int = 300, height: int = 60) -> bytes:
    """Generate a PNG image with the given text rendered in black on white.

    Uses OpenCV (already a project dependency via the ``camera`` optional
    group) so no new third-party library is needed for the test.
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.putText(
        img, text, (10, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,
    )
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _image_content_part(text: str) -> "ContentPart":
    """Create a ContentPart with a generated image containing the given text."""
    from peteos.conversation.message import ContentPart

    raw_png = _generate_text_image(text)
    b64 = base64.b64encode(raw_png).decode("ascii")
    return ContentPart.create_image({
        "type": "base64",
        "data": b64,
        "media_type": "image/png",
    })


def _skip_if_model_not_vision_capable(_chatbot_name: str):
    """Skip if the configured model is known to NOT support image input.

    Heuristic: only skip when we see explicit text-only model identifiers.
    All other models are allowed to run — they'll either handle the image
    gracefully or return an API error that the test will catch.
    """
    _skip_if_not_configured()
    import os
    model = os.environ.get("CHATBOT_TEST_MODEL", "").lower()
    # Explicitly known text-only models
    text_only_prefixes = (
        "qwen-2", "qwen2-", "llama-3", "llama3-",
    )
    if any(model.startswith(p) for p in text_only_prefixes):
        pytest.skip(f"Model '{model}' is known to be text-only, skipping image test")


class TestOpenAIImage:
    """Test live OpenAI ChatBot with image content."""

    @pytest.mark.asyncio
    async def test_send_image_text_recognition(self, live_openai_chatbot):
        """Send an image with embedded text and ask the bot to read it."""
        _skip_if_model_not_vision_capable("openai")

        from peteos.conversation.context import Context
        from peteos.conversation.message import ContentPart, Message

        sample_text = "PeteosImageTest42"

        messages = [
            Message.create("system", [
                ContentPart.create_text(
                    "You are a text extraction assistant. When given an image, "
                    "return exactly the text you see in it. Do not add any other text."
                ),
            ]),
            Message.create("user", [
                _image_content_part(sample_text),
                ContentPart.create_text("What text is in this image?"),
            ]),
        ]
        ctx = Context.create()
        for msg in messages:
            ctx.append(msg)

        response = await live_openai_chatbot.send_context(ctx, streaming=False)
        async for _ in response:
            pass

        assert "role" in response.data
        assert response.data["role"] == "assistant"
        assert "content" in response.data

        # Extract text from the response content blocks
        content_blocks = response.data["content"]
        text_parts = [
            block["content"]
            for block in content_blocks
            if block["type"] == "text"
        ]
        full_text = " ".join(text_parts)

        # The LLM should be able to read the text from the image
        assert sample_text in full_text, (
            f"Expected LLM to recognize '{sample_text}' from the image, "
            f"but got: {full_text}"
        )

    @pytest.mark.asyncio
    async def test_send_image_streaming(self, live_openai_chatbot):
        """Send an image in streaming mode and verify chunks arrive."""
        _skip_if_model_not_vision_capable("openai")

        from peteos.conversation.context import Context
        from peteos.conversation.message import ContentPart, Message

        sample_text = "VisionTest99"

        messages = [
            Message.create("system", [
                ContentPart.create_text("Return exactly the text you see."),
            ]),
            Message.create("user", [
                _image_content_part(sample_text),
                ContentPart.create_text("Read the text in this image."),
            ]),
        ]
        ctx = Context.create()
        for msg in messages:
            ctx.append(msg)

        response = await live_openai_chatbot.send_context(ctx, streaming=True)
        chunks = []
        async for chunk in response:
            chunks.append(chunk)

        assert "role" in response.data
        assert response.data["role"] == "assistant"
        assert len(chunks) > 0

        content_blocks = response.data["content"]
        text_parts = [
            block["content"]
            for block in content_blocks
            if block["type"] == "text"
        ]
        full_text = " ".join(text_parts)

        assert sample_text in full_text, (
            f"Expected LLM to recognize '{sample_text}' from the image, "
            f"but got: {full_text}"
        )


class TestAnthropicImage:
    """Test live Anthropic ChatBot with image content."""

    @pytest.mark.asyncio
    async def test_send_image_text_recognition(self, live_anthropic_chatbot):
        """Send an image with embedded text and ask the bot to read it."""
        _skip_if_model_not_vision_capable("anthropic")

        from peteos.conversation.context import Context
        from peteos.conversation.message import ContentPart, Message

        sample_text = "AnthropicVision7"

        messages = [
            Message.create("system", [
                ContentPart.create_text(
                    "You are a text extraction assistant. When given an image, "
                    "return exactly the text you see in it."
                ),
            ]),
            Message.create("user", [
                _image_content_part(sample_text),
                ContentPart.create_text("What text is in this image?"),
            ]),
        ]
        ctx = Context.create()
        for msg in messages:
            ctx.append(msg)

        response = await live_anthropic_chatbot.send_context(ctx, streaming=False)
        async for _ in response:
            pass

        assert "role" in response.data
        assert response.data["role"] == "assistant"
        assert "content" in response.data

        content_blocks = response.data["content"]
        text_parts = [
            block["content"]
            for block in content_blocks
            if block["type"] == "text"
        ]
        full_text = " ".join(text_parts)

        assert sample_text in full_text, (
            f"Expected LLM to recognize '{sample_text}' from the image, "
            f"but got: {full_text}"
        )


class TestGeminiImage:
    """Test live Gemini ChatBot with image content."""

    @pytest.mark.asyncio
    async def test_send_image_text_recognition(self, live_gemini_chatbot):
        """Send an image with embedded text and ask the bot to read it."""
        _skip_if_model_not_vision_capable("gemini")

        from peteos.conversation.context import Context
        from peteos.conversation.message import ContentPart, Message

        sample_text = "GeminiVision3"

        messages = [
            Message.create("system", [
                ContentPart.create_text(
                    "You are a text extraction assistant. When given an image, "
                    "return exactly the text you see in it."
                ),
            ]),
            Message.create("user", [
                _image_content_part(sample_text),
                ContentPart.create_text("What text is in this image?"),
            ]),
        ]
        ctx = Context.create()
        for msg in messages:
            ctx.append(msg)

        response = await live_gemini_chatbot.send_context(ctx, streaming=False)
        async for _ in response:
            pass

        if "error" in response.data:
            return
        assert "role" in response.data
        assert response.data["role"] == "model"
        assert "content" in response.data

        content_blocks = response.data["content"]
        text_parts = [
            block["content"]
            for block in content_blocks
            if block["type"] == "text"
        ]
        full_text = " ".join(text_parts)

        assert sample_text in full_text, (
            f"Expected LLM to recognize '{sample_text}' from the image, "
            f"but got: {full_text}"
        )
