"""Mock HTTP client for testing ChatBot implementations."""

from typing import AsyncGenerator, Dict, Any, List


class MockHTTPClient:
    """Mock HTTP client for testing.

    Allows injecting predefined responses without making real API calls.
    """

    def __init__(self):
        """Initialize MockHTTPClient."""
        self._stream_responses: List[str] = []
        self._post_response: Dict[str, Any] = {}
        self._last_url: str = ""
        self._last_body: Dict[str, Any] = {}

    async def stream_post(
        self,
        url: str,
        body: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Simulate streaming POST response.

        Args:
            url: The request URL (stored for verification).
            body: The request body (stored for verification).

        Yields:
            Predefined SSE lines.
        """
        self._last_url = url
        self._last_body = body

        for line in self._stream_responses:
            yield line

    async def post(self, url: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate non-streaming POST response.

        Args:
            url: The request URL (stored for verification).
            body: The request body (stored for verification).

        Returns:
            Predefined response.
        """
        self._last_url = url
        self._last_body = body
        return self._post_response

    def set_stream_response(self, lines: List[str]) -> None:
        """
        Set the responses for streaming requests.

        Args:
            lines: List of SSE lines to yield (without "data: " prefix).
        """
        self._stream_responses = [f"data: {line}" if not line.startswith("data: ") else line for line in lines]
        self._stream_responses.append("[DONE]")

    def set_post_response(self, response: Dict[str, Any]) -> None:
        """
        Set the response for non-streaming POST requests.

        Args:
            response: JSON response to return.
        """
        self._post_response = response

    @property
    def last_url(self) -> str:
        """Get the last URL called."""
        return self._last_url

    @property
    def last_body(self) -> Dict[str, Any]:
        """Get the last request body."""
        return self._last_body
