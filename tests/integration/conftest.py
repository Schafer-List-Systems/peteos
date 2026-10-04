"""Integration test configuration for the chatbot package.

Integration tests use tests/integration/mock_backend.py as the backend — no API keys needed.
"""

import os

import pytest


def _get_env(name):
    return os.environ.get(name)


def _get_config():
    return {
        "url": _get_env("CHATBOT_TEST_BACKEND_URL") or "http://localhost:18888",
        "name": _get_env("CHATBOT_TEST_BACKEND_NAME") or "test",
        "api_key": _get_env("CHATBOT_TEST_API_KEY") or "test",
        "model": _get_env("CHATBOT_TEST_MODEL") or "test-model",
        "api_type": _get_env("CHATBOT_TEST_API_TYPE") or "openai",
    }


@pytest.fixture
def chatbot_config():
    return _get_config()
