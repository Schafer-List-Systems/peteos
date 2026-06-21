"""Unit tests for HTTPClient."""

import pytest

from peteos.chatbot.httpclient import HTTPClient


class TestHTTPClient:
    """Tests for HTTPClient initialization and configuration."""

    def test_init_with_timeout(self):
        """HTTPClient stores timeout."""
        client = HTTPClient(timeout=30.0)
        assert client._timeout == 30.0

    def test_init_without_timeout(self):
        """HTTPClient accepts None for no timeout."""
        client = HTTPClient(timeout=None)
        assert client._timeout is None

    def test_init_default_retry_delays(self):
        """HTTPClient defaults to [0, 1, 3] retry delays."""
        client = HTTPClient(timeout=5.0)
        assert client._retry_delays == [0, 1, 3]

    def test_init_custom_retry_delays(self):
        """HTTPClient accepts custom retry delays."""
        client = HTTPClient(timeout=5.0, retry_delays=[1, 2, 3, 4])
        assert client._retry_delays == [1, 2, 3, 4]

    def test_max_retries_calculation(self):
        """_max_retries is len(retry_delays)."""
        client = HTTPClient(timeout=5.0)
        assert client._max_retries == 3  # len([0, 1, 3])

    def test_max_retries_custom(self):
        """_max_retries reflects custom retry_delays length."""
        client = HTTPClient(timeout=5.0, retry_delays=[1])
        assert client._max_retries == 1

    def test_get_and_post_methods_exist(self):
        """HTTPClient has get and post methods."""
        client = HTTPClient(timeout=5.0)
        assert hasattr(client, "get")
        assert hasattr(client, "post")
        assert hasattr(client, "stream_post")
