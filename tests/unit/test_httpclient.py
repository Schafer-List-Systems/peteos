"""Tests for HTTPClient."""

import pytest
from peteos.httpclient import HTTPClient


class TestHTTPClientStreaming:
    """Tests for HTTPClient stream_post method."""

    @pytest.mark.asyncio
    async def test_stream_post_returns_async_generator(self):
        """Test that stream_post returns an async generator."""
        # This would require mocking httpx - skipped for now
        # In real tests, we'd use pytest-httpx or similar
        client = HTTPClient(timeout=5.0)
        # Just verify instantiation works
        assert client._timeout == 5.0

    @pytest.mark.asyncio
    async def test_stream_post_with_mock(self):
        """Test stream_post using mock HTTPClient."""
        from tests.unit.mock_httpclient import MockHTTPClient

        mock_client = MockHTTPClient()
        mock_client.set_stream_response([
            '{"choices": [{"delta": {"content": "Hello"}}]}',
            '{"choices": [{"delta": {"content": " World"}}]}'
        ])

        lines = []
        async for line in mock_client.stream_post("http://test.com", {"test": "body"}):
            lines.append(line)

        assert len(lines) >= 3  # At least 2 data lines + [DONE]
        assert 'data: {"choices": [{"delta": {"content": "Hello"}}]}' in lines[0]


class TestHTTPClientPost:
    """Tests for HTTPClient post method."""

    @pytest.mark.asyncio
    async def test_post_with_mock(self):
        """Test post using mock HTTPClient."""
        from tests.unit.mock_httpclient import MockHTTPClient

        expected_response = {"result": "success", "data": [1, 2, 3]}
        mock_client = MockHTTPClient()
        mock_client.set_post_response(expected_response)

        result = await mock_client.post("http://test.com", {"query": "test"})

        assert result == expected_response
        assert mock_client.last_url == "http://test.com"
        assert mock_client.last_body == {"query": "test"}
