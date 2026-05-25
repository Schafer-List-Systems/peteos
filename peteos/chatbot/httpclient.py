"""HTTP client for communicating with LLM APIs."""

import asyncio
import httpx
from typing import AsyncGenerator, Optional

from peteos.logger import get_logger

_logger = get_logger(__name__)


class HTTPClient:
    """Async HTTP client with streaming support for SSE."""

    def __init__(
        self,
        timeout: Optional[float],
        retry_delays: Optional[list[float]] = None,
    ):
        """
        Initialize HTTPClient.

        Args:
            timeout: Request timeout in seconds. Pass None for no timeout.
            retry_delays: Delay in seconds before each retry attempt.
                Defaults to [0, 1, 3] (immediate, then 1s, then 3s).
                The length of this list defines the max number of retries.
        """
        self._timeout = timeout
        self._retry_delays = retry_delays or [0, 1, 3]

    @property
    def _max_retries(self) -> int:
        return len(self._retry_delays)

    async def _ensure_response(self, client: httpx.AsyncClient, method: str, url: str, json_body: Optional[dict] = None) -> httpx.Response:
        """Make an HTTP request with retry for connection-level failures.

        Retries on HTTP errors (3xx, 5xx) and network errors.
        GeneratorExit is always propagated without retry.

        Args:
            client: The httpx.AsyncClient to use.
            method: HTTP method ("GET" or "POST").
            url: Target URL.
            json_body: Optional JSON body for POST requests.

        Returns:
            The httpx.Response object.

        Raises:
            GeneratorExit: Always re-raised without retry.
            RuntimeError: On HTTP error after all retries exhausted.
            Exception: On network error after all retries exhausted.
        """
        for attempt in range(self._max_retries + 1):
            try:
                if method == "GET":
                    return await client.get(url)
                else:
                    return await client.post(url, json=json_body)
            except GeneratorExit:
                raise
            except Exception:
                if attempt < self._max_retries:
                    delay = self._retry_delays[attempt]
                    if delay > 0:
                        _logger.warning("HTTP %s to %s failed (attempt %d/%d), retrying in %ss", method, url, attempt + 1, self._max_retries, delay)
                    await asyncio.sleep(delay)
                    continue
                raise

    async def stream_post(
        self,
        url: str,
        body: dict
    ) -> AsyncGenerator[str, None]:
        """
        Send a POST request and stream SSE events.

        The initial connection is retried on failure. Once streaming
        begins, mid-stream failures are not retried.

        Args:
            url: Endpoint URL.
            body: Request body.

        Yields:
            Raw SSE lines as strings.

        Raises:
            RuntimeError: If the response status code is not 2xx.
        """
        for attempt in range(self._max_retries + 1):
            client = httpx.AsyncClient(timeout=self._timeout)
            try:
                async with client:
                    response = await self._ensure_response(client, "POST", url, body)
                    if response.status_code >= 300:
                        error_text = await response.aread()
                        error_str = error_text.decode(errors='replace')
                        _logger.error("HTTP %d from %s: %s", response.status_code, url, error_str)
                        raise RuntimeError(
                            f"HTTP {response.status_code} from {url}: {error_str}"
                        )

                    try:
                        async for line in response.aiter_lines():
                            if line:
                                yield line
                    finally:
                        await response.aclose()
                    return
            except GeneratorExit:
                await client.aclose()
                raise
            except Exception:
                await client.aclose()
                if attempt < self._max_retries:
                    delay = self._retry_delays[attempt]
                    if delay > 0:
                        _logger.warning("HTTP POST to %s failed (attempt %d/%d), retrying in %ss", url, attempt + 1, self._max_retries, delay)
                    await asyncio.sleep(delay)
                    continue
                raise

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
            response = await self._ensure_response(client, "GET", url)
            if response.status_code >= 300:
                error_text = response.content.decode(errors='replace')
                _logger.error("HTTP %d from %s: %s", response.status_code, url, error_text)
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
            response = await self._ensure_response(client, "POST", url, body)
            if response.status_code >= 300:
                error_text = response.content.decode(errors='replace')
                _logger.error("HTTP %d from %s: %s", response.status_code, url, error_text)
                raise RuntimeError(
                    f"HTTP {response.status_code} from {url}: {error_text}"
                )
            return response.json()
