"""HTTP client for communicating with LLM APIs."""

import httpx
from typing import AsyncGenerator, Optional


class HTTPClient:
    """Async HTTP client with streaming support for SSE."""

    def __init__(self, timeout: Optional[float]):
        """
        Initialize HTTPClient.

        Args:
            timeout: Request timeout in seconds. Pass None for no timeout.
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

        Raises:
            RuntimeError: If the response status code is not 2xx.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", url, json=body) as response:
                if response.status_code >= 300:
                    error_text = await response.aread()
                    raise RuntimeError(
                        f"HTTP {response.status_code} from {url}: {error_text.decode(errors='replace')}"
                    )
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

        Raises:
            RuntimeError: If the response status code is not 2xx.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url)
            if response.status_code >= 300:
                error_text = response.content.decode(errors='replace')
                raise RuntimeError(
                    f"HTTP {response.status_code} from {url}: {error_text}"
                )
            return response.json()

    async def post(self, url: str, body: dict) -> dict:
        """
        Send a POST request and return the parsed JSON response.

        Args:
            url: Endpoint URL.
            body: Request body.

        Returns:
            Parsed JSON response.

        Raises:
            RuntimeError: If the response status code is not 2xx.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=body)
            if response.status_code >= 300:
                error_text = response.content.decode(errors='replace')
                raise RuntimeError(
                    f"HTTP {response.status_code} from {url}: {error_text}"
                )
            return response.json()
