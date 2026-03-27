"""HTTP server for mocking LLM backends."""

import asyncio
import json
from aiohttp import web


class MockLLMServer:
    """HTTP server for mocking LLM backends."""

    def __init__(self, responses: list[str], host: str = "127.0.0.1", port: int = 8765):
        """
        Initialize mock server.

        Args:
            responses: List of JSON response strings to send (without data: prefix)
            host: Server host
            port: Server port
        """
        self.responses = [f"data: {resp}\n" for resp in responses if resp != "[DONE]"]
        self.responses.append("[DONE]\n")
        self.host = host
        self.port = port
        self._app: web.Application = None
        self._runner: web.AppRunner = None
        self._site: web.TCPSite = None
        self._url: str = ""

    async def start(self) -> str:
        """Start the mock server and return the base URL."""
        self._app = web.Application()
        self._app.router.add_post("/v1/chat/completions", self._handle_completions)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        self._url = f"http://{self.host}:{self.port}"
        return self._url

    async def stop(self):
        """Stop the mock server."""
        if self._runner:
            await self._runner.cleanup()

    async def _handle_completions(self, request: web.Request) -> web.Response:
        """Handle streaming completions requests."""
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        await response.prepare(request)

        for data in self.responses:
            await response.write(data.encode())
            await asyncio.sleep(0.01)

        await response.write_eof()
        return response

    @property
    def url(self) -> str:
        """Get the server URL."""
        return self._url


def create_openai_mock_server(reasoning: str = "", response: str = "Hello from mock LLM", port: int = 8765):
    """Create a mock server for OpenAI-compatible responses."""
    responses = []
    if reasoning:
        responses.append(json.dumps({"choices": [{"delta": {"reasoning": reasoning, "content": ""}}]}))
    responses.append(json.dumps({"choices": [{"delta": {"content": response}}]}))
    responses.append("[DONE]")

    return MockLLMServer(responses, port=port)


def create_anthropic_mock_server(thinking: str = "Thinking step by step", response: str = "Hello", port: int = 8765):
    """Create a mock server for Anthropic-compatible responses."""
    responses = [
        json.dumps({
            "type": "content_block_start",
            "content_block": {"type": "thinking", "thinking": ""}
        }),
        json.dumps({
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": thinking}
        }),
        json.dumps({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": response}
        }),
        json.dumps({
            "type": "content_block_stop",
            "content_block": {"type": "text"}
        }),
        "[DONE]"
    ]
    return MockLLMServer(responses, port=port)
