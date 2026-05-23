"""HTTP client for communicating with LLM APIs."""

import httpx
from typing import AsyncGenerator


class HTTPClient:
    """Async HTTP client with streaming support for SSE."""

    def __init__(self, timeout: float):
        """
        Initialize HTTPClient.

        Args:
            timeout: Request timeout in seconds.
        """
        self._timeout = timeout

    async def stream_post(
        self,
        url: str,
        body: dict
    ) -> AsyncGenerator[str, None]:
        """
        Send a POST request and stream SSE events.

        Args:
            url: Endpoint URL.
            body: Request body.

        Yields:
            Raw SSE lines as strings.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", url, json=body) as response:
                async for line in response.aiter_lines():
                    if line:
                        yield line

    async def get(self, url: str) -> dict:
        """
        Send a GET request and return the parsed JSON response.

        Args:
            url: Endpoint URL.

        Returns:
            Parsed JSON response.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url)
            return response.json()

    async def post(self, url: str, body: dict) -> dict:
        """
        Send a POST request and return the parsed JSON response.

        Args:
            url: Endpoint URL.
            body: Request body.

        Returns:
            Parsed JSON response.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=body)
            return response.json()
