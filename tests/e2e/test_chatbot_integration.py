"""End-to-end tests for ChatBot against real LLM endpoints."""

import pytest
import pytest_asyncio
from peteos.chatbot import OpenAIChatBot, AnthropicChatBot
from peteos.chatbot import HTTPClient
from peteos.chatbot import ChatHistory
from peteos.chatbot import Message, ContentPart
from peteos.chatbot.chatbotconfig import ChatBotConfig
from tests.http.mock_server import create_openai_mock_server, create_anthropic_mock_server


# Use this for real endpoint testing
# REAL_BASE_URL = "http://192.168.255.10:8123"
# REAL_MODEL = "qwen/qwen3.5-35b-a3b"


class TestOpenAIChatBotIntegration:
    """Integration tests for OpenAIChatBot."""

    @pytest.mark.asyncio
    async def test_openai_chatbot_with_mock_server(self):
        """Test OpenAIChatBot with mock server."""
        server = create_openai_mock_server(
            reasoning="This is mock reasoning",
            response="Mock response from test server",
            port=8766
        )
        await server.start()
        try:
            http_client = HTTPClient(timeout=5.0)
            config = ChatBotConfig(name="test", url=server.url, model="test-model")
            chatbot = OpenAIChatBot(http_client=http_client, config=config)

            history = ChatHistory()
            history.append_message(Message(role="user", content=[ContentPart(part_type="text", text="Hello")]))

            response = await chatbot.send_message(history, streaming=True)

            # Collect streaming response
            accumulated = []
            async for chunk in response:
                accumulated.append(chunk)

            assert len(accumulated) > 0
            # Streaming chunks are tuples - verify last chunk contains the response text
            last_chunk = accumulated[-1]
            if isinstance(last_chunk, tuple) and len(last_chunk) == 2:
                assert "Mock response from test server" in str(last_chunk[1])
            else:
                assert "Mock response from test server" in last_chunk
            # Reasoning should be a thinking block in content array
            assert "content" in response.data
            full_text = ""
            for block in response.data["content"]:
                if block.get("type") == "thinking":
                    assert "This is mock reasoning" in block.get("content", "")
                else:
                    full_text += block.get("content", "")
            assert "Mock response from test server" in full_text
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_openai_chatbot_non_streaming_with_mock(self):
        """Test OpenAIChatBot non-streaming mode with mock server."""
        server = create_openai_mock_server(
            reasoning="",
            response="Complete non-streaming response",
            port=8768
        )
        await server.start()
        try:
            http_client = HTTPClient(timeout=5.0)
            config = ChatBotConfig(name="test", url=server.url, model="test-model")
            chatbot = OpenAIChatBot(http_client=http_client, config=config)

            history = ChatHistory()
            history.append_message(Message(role="user", content=[ContentPart(part_type="text", text="Hello")]))

            response = await chatbot.send_message(history, streaming=False)

            # Non-streaming: data is already populated via from_json
            assert response.data["role"] == "assistant"
            assert "content" in response.data
            full_text = ""
            for block in response.data["content"]:
                full_text += block.get("content", "")
            assert "Complete non-streaming response" in full_text
        finally:
            await server.stop()


class TestAnthropicChatBotIntegration:
    """Integration tests for AnthropicChatBot."""

    @pytest.mark.asyncio
    async def test_anthropic_chatbot_with_mock_server(self):
        """Test AnthropicChatBot with mock server."""
        server = create_anthropic_mock_server(
            thinking="This is mock thinking",
            response="Mock response from Anthropic test server",
            port=8767
        )
        await server.start()
        try:
            http_client = HTTPClient(timeout=5.0)
            config = ChatBotConfig(name="test", url=server.url, model="test-model")
            chatbot = AnthropicChatBot(http_client=http_client, config=config)

            history = ChatHistory()
            history.append_message(Message(role="user", content=[ContentPart(part_type="text", text="Hello")]))

            response = await chatbot.send_message(history, streaming=True)

            # Collect streaming response
            accumulated = []
            async for chunk in response:
                accumulated.append(chunk)

            assert len(accumulated) > 0
            # Anthropic uses content array format
            assert "content" in response.data
            content = response.data["content"]
            # Check that content blocks are present and contain expected text
            full_text = ""
            for block in content:
                full_text += block.get("content", "")
            assert "Mock response from Anthropic test server" in full_text
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_anthropic_chatbot_non_streaming_with_mock(self):
        """Test AnthropicChatBot non-streaming mode with mock server."""
        server = create_anthropic_mock_server(
            thinking="",
            response="Complete non-streaming response",
            port=8769
        )
        await server.start()
        try:
            http_client = HTTPClient(timeout=5.0)
            config = ChatBotConfig(name="test", url=server.url, model="test-model")
            chatbot = AnthropicChatBot(http_client=http_client, config=config)

            history = ChatHistory()
            history.append_message(Message(role="user", content=[ContentPart(part_type="text", text="Hello")]))

            response = await chatbot.send_message(history, streaming=False)

            # Non-streaming: data is already populated via from_json
            assert "content" in response.data
            full_text = ""
            for block in response.data["content"]:
                full_text += block.get("content", "")
            assert "Complete non-streaming response" in full_text
        finally:
            await server.stop()
