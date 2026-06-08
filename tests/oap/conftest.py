"""Session-scoped ChatBotManager fixture for OAP benchmarks."""

import os

import pytest

from peteos.chatbot import ChatBotManager


@pytest.fixture(autouse=True)
async def _oap_backend():
    """Configure ChatBotManager with a backend for OAP benchmarks.

    Uses OAP_BACKEND_URL environment variable (required).
    Resets manager first to ensure clean state across test runs.
    """
    url = os.environ["OAP_BACKEND_URL"]
    ChatBotManager.reset()
    await ChatBotManager.add_backend(
        "local", url, api_type="anthropic", streaming=False
    )
    yield
    ChatBotManager.reset()
