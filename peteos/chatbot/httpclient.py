"""HTTP client for communicating with LLM APIs."""

import asyncio
import httpx
from typing import AsyncGenerator, Callable, Optional

from peteos.utils import get_logger

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

    async def _ensure_response(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        json_body: Optional[dict] = None,
        hooks: Optional[dict[str, list[Callable]]] = None,
    ) -> httpx.Response:
        """Make an HTTP request with retry for network errors and HTTP error status codes.

        Retries on:
        - Network errors (connection refused, timeout, DNS failure)
        - HTTP error status codes (3xx, 4xx, 5xx)

        GeneratorExit is always propagated without retry.
        on_http_error hooks are called on each HTTP error; if a hook raises,
        the exception propagates and the retry loop is broken.

        Args:
            client: The httpx.AsyncClient to use.
            method: HTTP method ("GET" or "POST").
            url: Target URL.
            json_body: Optional JSON body for POST requests.
            hooks: Optional dict of hook name -> list of callables.
                "on_http_error" hooks receive {"url", "status_code", "body", "error_text"}
                and may raise to abort the retry loop.

        Returns:
            The httpx.Response object (2xx status).

        Raises:
            GeneratorExit: Always re-raised without retry.
            RuntimeError: On HTTP error after all retries exhausted.
            Exception: On network error after all retries exhausted.
             Exception: Re-raised if an on_http_error hook raises.
        """
        # Prepare the error context once; re-used across retries with attempt metadata attached on each failure.
        error_context: dict | None = None
        for attempt in range(self._max_retries + 1):
            try:
                # Attempt one HTTP request against the target.
                response = await client.get(url) if method == "GET" else await client.post(url, json=json_body)

                # Surface non-success status codes as errors that will be caught below.
                if response.status_code >= 300:
                    error_text = response.content.decode(errors='replace')
                    error_context = {
                        "url": url,
                        "status_code": response.status_code,
                        "body": json_body,
                        "error_text": error_text,
                    }
                    _logger.error("HTTP %d from %s (attempt %d/%d): %s", response.status_code, url, attempt + 1, self._max_retries + 1, error_text)
                    raise RuntimeError(
                        f"HTTP {response.status_code} from {url}: {error_text}"
                    )

                # 2xx — return the successful response.
                return response

            # GeneratorExit — raised by the caller cancelling the async generator
            # (e.g. the session was interrupted). Always propagate without retry.
            except GeneratorExit:
                raise

            # Any other exception — attach retry metadata, fire hooks, then decide whether to retry.
            except Exception as e:
                if error_context is not None:
                    error_context["attempt"] = attempt + 1
                    error_context["max_retries"] = self._max_retries
                    error_context["exception"] = e

                # Notify listeners of the error; a listener may raise to abort retries.
                for hook in (hooks or {}).get("on_http_error", []):
                    result = hook(error_context)
                    if asyncio.iscoroutine(result):
                        await result

                # Check whether any retries remain for this attempt loop.
                if attempt < self._max_retries:
                    delay = self._retry_delays[attempt]
                    if delay > 0:
                        _logger.warning("HTTP %s to %s failed (attempt %d/%d), retrying in %ss", method, url, attempt + 1, self._max_retries, delay)
                    await asyncio.sleep(delay)
                    continue

                # No retries remain — surface the original error.
                raise

    async def stream_post(
        self,
        url: str,
        body: dict,
        headers: Optional[dict] = None,
        hooks: Optional[dict[str, list[Callable]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Send a POST request and stream SSE events.

        The initial connection and HTTP error responses are retried.
        Once streaming begins, mid-stream failures are not retried.
        on_http_error hooks fire on each HTTP-level error and may raise to abort retry.

        Args:
            url: Endpoint URL.
            body: Request body.
            headers: Optional HTTP headers to include.
            hooks: Optional hook dict passed to _ensure_response.
                See _ensure_response for hook semantics.

        Yields:
            Raw SSE lines as strings.

        Raises:
            RuntimeError: If the response status code is not 2xx after all retries.
            Exception: Re-raised if an on_http_error hook raises.
        """
        client = httpx.AsyncClient(timeout=self._timeout, headers=headers or {})
        try:
            async with client:
                response = await self._ensure_response(client, "POST", url, body, hooks=hooks)

                try:
                    async for line in response.aiter_lines():
                        if line:
                            yield line
                finally:
                    await response.aclose()
        finally:
            await client.aclose()

    async def get(
        self,
        url: str,
        headers: Optional[dict] = None,
        hooks: Optional[dict[str, list[Callable]]] = None,
    ) -> dict:
        """
        Send a GET request and return the parsed JSON response.

        Args:
            url: Target URL.
            headers: Optional HTTP headers to include.
            hooks: Optional hook dict passed to _ensure_response.
                See _ensure_response for hook semantics.

        Returns:
            Parsed JSON response.

        Raises:
            RuntimeError: If the response status code is not 2xx after all retries.
            Exception: Re-raised if an on_http_error hook raises.
        """
        async with httpx.AsyncClient(timeout=self._timeout, headers=headers or {}) as client:
            response = await self._ensure_response(client, "GET", url, hooks=hooks)
            return response.json()

    async def post(
        self,
        url: str,
        body: dict,
        headers: Optional[dict] = None,
        hooks: Optional[dict[str, list[Callable]]] = None,
    ) -> dict:
        """
        Send a POST request and return the parsed JSON response.

        Args:
            url: Endpoint URL.
            body: Request body.
            headers: Optional HTTP headers to include.
            hooks: Optional hook dict passed to _ensure_response.
                See _ensure_response for hook semantics.

        Returns:
            Parsed JSON response.

        Raises:
            RuntimeError: If the response status code is not 2xx after all retries.
            Exception: Re-raised if an on_http_error hook raises.
        """
        async with httpx.AsyncClient(timeout=self._timeout, headers=headers or {}) as client:
            response = await self._ensure_response(client, "POST", url, body, hooks=hooks)
            return response.json()
