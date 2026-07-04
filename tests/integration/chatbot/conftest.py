"""Integration test configuration for the chatbot package.

Environment variables (all required for integration tests to run):
    CHATBOT_TEST_BACKEND_URL   - Base URL of the LLM backend (e.g., http://localhost:8000)
    CHATBOT_TEST_BACKEND_NAME  - Backend name (e.g., "test-backend")
    CHATBOT_TEST_API_KEY       - API key for the backend
    CHATBOT_TEST_MODEL         - Model name to use (e.g., "gpt-4", "claude-3-haiku")
    CHATBOT_TEST_API_TYPE      - "openai" or "anthropic" (optional, defaults to "openai")

Run integration tests with:
    CHATBOT_TEST_BACKEND_URL=http://localhost:8000 \
    CHATBOT_TEST_BACKEND_NAME=test \
    CHATBOT_TEST_API_KEY=sk-test \
    CHATBOT_TEST_MODEL=gpt-4 \
    pytest tests/integration/chatbot/ -v -m chatbot_integration

Without env vars, tests are skipped with a message explaining what's needed.
"""

import os
from typing import Any, Dict, Optional

import pytest
import pytest_asyncio
from aiohttp import ClientSession, ClientTimeout


def _get_env(name):
    """Get an environment variable, returning None if not set."""
    return os.environ.get(name)


def _get_config():
    """Get the full test configuration from env vars."""
    return {
        "url": _get_env("CHATBOT_TEST_BACKEND_URL"),
        "name": _get_env("CHATBOT_TEST_BACKEND_NAME"),
        "api_key": _get_env("CHATBOT_TEST_API_KEY"),
        "model": _get_env("CHATBOT_TEST_MODEL"),
        "api_type": _get_env("CHATBOT_TEST_API_TYPE") or "openai",
    }


@pytest.fixture
def chatbot_config():
    """Return the test configuration dict from env vars."""
    return _get_config()


def _session_http_client_factory(session, base_url):
    """Create a SessionHTTPClient class that captures the session."""

    class SessionHTTPClient:
        def __init__(self, sess):
            self._session = sess
            self._base_url = base_url

        async def post(self, url, body, headers: Optional[Dict[str, str]] = None):
            # The full URL is passed by ChatBot; strip base_url to get the relative path
            relative_path = url[len(self._base_url):] if url.startswith(self._base_url) else url
            kwargs: dict[str, Any] = {"json": body}
            if headers:
                kwargs["headers"] = headers
            async with self._session.post(relative_path, **kwargs) as resp:
                return await resp.json()

        async def stream_post(self, url, body, headers: Optional[Dict[str, str]] = None):
            """Yield raw SSE lines from the response."""
            relative_path = url[len(self._base_url):] if url.startswith(self._base_url) else url
            async with self._session.post(relative_path, json=body) as resp:
                async for line in resp.content:
                    decoded = line.decode("utf-8")
                    if decoded.strip():
                        yield decoded

    return SessionHTTPClient


@pytest_asyncio.fixture
async def openai_client():
    """Create an OpenAI-compatible aiohttp ClientSession."""
    cfg = _get_config()
    url = cfg["url"]
    api_key = cfg["api_key"]

    async with ClientSession(
        base_url=url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=ClientTimeout(total=120),
    ) as client:
        yield client


@pytest_asyncio.fixture
async def anthropic_client():
    """Create an Anthropic-compatible aiohttp ClientSession."""
    cfg = _get_config()
    url = cfg["url"]
    api_key = cfg["api_key"]

    async with ClientSession(
        base_url=url,
        headers={
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        timeout=ClientTimeout(total=120),
    ) as client:
        yield client


@pytest_asyncio.fixture
async def live_openai_chatbot(openai_client):
    """Create a live OpenAI ChatBot instance connected to the test backend."""
    cfg = _get_config()

    from peteos.chatbot.openaichatbot import OpenAIChatBot
    from peteos.chatbot.chatbotconfig import ChatBotConfig

    config = ChatBotConfig(
        name=cfg["name"],
        url=cfg["url"],
        api_type="openai",
        model=cfg["model"],
        streaming=False,
    )

    SessionHTTPClient = _session_http_client_factory(openai_client, cfg["url"])
    http_client = SessionHTTPClient(openai_client)
    chatbot = OpenAIChatBot(http_client, config)
    yield chatbot


@pytest_asyncio.fixture
async def live_anthropic_chatbot(anthropic_client):
    """Create a live Anthropic ChatBot instance connected to the test backend."""
    cfg = _get_config()

    from peteos.chatbot.anthropicchatbot import AnthropicChatBot
    from peteos.chatbot.chatbotconfig import ChatBotConfig

    config = ChatBotConfig(
        name=cfg["name"],
        url=cfg["url"],
        api_type="anthropic",
        model=cfg["model"],
        streaming=False,
    )

    SessionHTTPClient = _session_http_client_factory(anthropic_client, cfg["url"])
    http_client = SessionHTTPClient(anthropic_client)
    chatbot = AnthropicChatBot(http_client, config)
    yield chatbot


# ---------------------------------------------------------------------------
# Helper for building contexts with the conversation package API
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def gemini_client():
    """Create a Gemini-compatible aiohttp ClientSession.

    Gemini uses ?key=API_KEY query param for auth (not a header).
    The GeminiChatBot appends it to the URL, so the fixture needs no special auth.
    """
    cfg = _get_config()
    url = cfg["url"]

    async with ClientSession(
        base_url=url,
        headers={"Content-Type": "application/json"},
        timeout=ClientTimeout(total=120),
    ) as client:
        yield client


@pytest_asyncio.fixture
async def live_gemini_chatbot(gemini_client):
    """Create a live Gemini ChatBot instance connected to the test backend."""
    cfg = _get_config()

    from peteos.chatbot.geminichatbot import GeminiChatBot
    from peteos.chatbot.chatbotconfig import ChatBotConfig

    config = ChatBotConfig(
        name=cfg["name"],
        url=cfg["url"],
        api_type="gemini",
        model=cfg["model"],
        streaming=False,
        api_key=cfg["api_key"],
    )

    SessionHTTPClient = _session_http_client_factory(gemini_client, cfg["url"])
    http_client = SessionHTTPClient(gemini_client)
    chatbot = GeminiChatBot(http_client, config)
    yield chatbot


# ---------------------------------------------------------------------------
# Helper for building contexts with the conversation package API
# ---------------------------------------------------------------------------

def _make_context(messages):
    """Build a Context from Message instances."""
    from peteos.conversation.context import Context

    ctx = Context({})
    for msg in messages:
        ctx.append(msg)
    return ctx
