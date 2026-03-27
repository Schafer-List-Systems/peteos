"""End-to-end tests for ChatBot against real LLM endpoints."""

import pytest
import pytest_asyncio
from peteos.chatbot import OpenAIChatBot, AnthropicChatBot
from peteos.httpclient import HTTPClient
from peteos.chathistory import ChatHistory
from peteos.message import Message
from tests.http.mock_server import create_openai_mock_server, create_anthropic_mock_server


# Use this for real endpoint testing
# REAL_BASE_URL = "http://192.168.255.10:8123"
# REAL_MODEL = "qwen/qwen3.5-35b-a3b"


class TestOpenAIChatBotIntegration:
    """Integration tests for OpenAIChatBot."""

    @pytest_asyncio.fixture
    async def mock_server(self):
        """Start a mock server for testing."""
        server = create_openai_mock_server(
            reasoning="This is mock reasoning",
            response="Mock response from test server",
            port=8766
        )
        await server.start()
        yield server
        await server.stop()

    @pytest.mark.asyncio
    async def test_openai_chatbot_with_mock_server(self, mock_server):
        """Test OpenAIChatBot with mock server."""
        http_client = HTTPClient(timeout=5.0)
        chatbot = OpenAIChatBot(
            http_client=http_client,
            model="test-model",
            base_url=mock_server.url
        )

        history = ChatHistory()
        history.append_message(Message(content={"role": "user", "content": "Hello"}))

        response = await chatbot.send_message(history, streaming=True)

        # Collect streaming response
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        assert len(accumulated) > 0
        assert "Mock response from test server" in accumulated[-1]
        assert "This is mock reasoning" in response.thinking_content

    @pytest.mark.asyncio
    async def test_openai_chatbot_non_streaming_with_mock(self, mock_server):
        """Test OpenAIChatBot non-streaming mode with mock server."""
        http_client = HTTPClient(timeout=5.0)
        chatbot = OpenAIChatBot(
            http_client=http_client,
            model="test-model",
            base_url=mock_server.url
        )

        history = ChatHistory()
        history.append_message(Message(content={"role": "user", "content": "Hello"}))

        # For non-streaming, we need to create a different mock
        server = create_openai_mock_server(
            reasoning="",
            response="Complete non-streaming response",
            port=8768
        )
        await server.start()

        chatbot_non_streaming = OpenAIChatBot(
            http_client=http_client,
            model="test-model",
            base_url=server.url
        )

        response = await chatbot_non_streaming.send_message(history, streaming=False)

        # Collect all chunks
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        assert len(accumulated) >= 1
        assert "Complete non-streaming response" in accumulated[-1]
        await server.stop()


class TestAnthropicChatBotIntegration:
    """Integration tests for AnthropicChatBot."""

    @pytest_asyncio.fixture
    async def mock_server(self):
        """Start a mock server for testing."""
        server = create_anthropic_mock_server(
            thinking="This is mock thinking",
            response="Mock response from Anthropic test server",
            port=8767
        )
        await server.start()
        yield server
        await server.stop()

    @pytest.mark.asyncio
    async def test_anthropic_chatbot_with_mock_server(self, mock_server):
        """Test AnthropicChatBot with mock server."""
        http_client = HTTPClient(timeout=5.0)
        chatbot = AnthropicChatBot(
            http_client=http_client,
            model="test-model",
            base_url=mock_server.url
        )

        history = ChatHistory()
        history.append_message(Message(content={"role": "user", "content": "Hello"}))

        response = await chatbot.send_message(history, streaming=True)

        # Collect streaming response
        accumulated = []
        async for chunk in response:
            accumulated.append(chunk)

        assert len(accumulated) > 0
        assert "Mock response from Anthropic test server" in accumulated[-1]
        assert "This is mock thinking" in response.thinking_content


# Uncomment to test against real endpoint
# class TestRealEndpoint:
#     """Tests against the real LLM endpoint at 192.168.255.10:8123."""
#
#     @pytest.mark.asyncio
#     async def test_openai_chatbot_with_real_endpoint(self):
#         """Test OpenAIChatBot against real Qwen model."""
#         http_client = HTTPClient(timeout=30.0)
#         chatbot = OpenAIChatBot(
#             http_client=http_client,
#             model="qwen/qwen3.5-35b-a3b",
#             base_url=REAL_BASE_URL
#         )
#
#         history = ChatHistory()
#         history.append_message(Message(content={"role": "user", "content": "Say hello"}))
#
#         response = await chatbot.send_message(history, streaming=True)
#
#         accumulated = []
#         async for chunk in response:
#             accumulated.append(chunk)
#
#         assert len(accumulated) > 0
#         # Response should contain some greeting
#         assert any(word in " ".join(accumulated).lower()
#                    for word in ["hello", "hi", "hey"])
