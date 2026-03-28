"""HTTP server for mocking LLM backends."""

import asyncio
import json
from aiohttp import web


class MockLLMServer:
    """HTTP server for mocking LLM backends."""

    def __init__(self, responses: list[str], host: str = "127.0.0.1", port: int = 8765, endpoint: str = "/chat/completions", enable_streaming: bool = True):
        """
        Initialize mock server.

        Args:
            responses: List of JSON response strings to send (without data: prefix)
            host: Server host
            port: Server port
            endpoint: API endpoint to handle (/chat/completions for OpenAI, /messages for Anthropic)
            enable_streaming: If False, returns JSON instead of SSE (for non-streaming mode)
        """
        self.responses = [f"data: {resp}\n" for resp in responses if resp != "[DONE]"]
        self.responses.append("[DONE]\n")
        self.json_response = [resp for resp in responses if resp != "[DONE]"]
        self.host = host
        self.port = port
        self.endpoint = endpoint
        self.enable_streaming = enable_streaming
        self._app: web.Application = None
        self._runner: web.AppRunner = None
        self._site: web.TCPSite = None
        self._url: str = ""

    async def start(self) -> str:
        """Start the mock server and return the base URL."""
        self._app = web.Application()
        self._app.router.add_post(self.endpoint, self._handle_completions)

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
        # Check request body for stream parameter
        body = await request.json()
        is_streaming = body.get("stream", True)

        if is_streaming:
            # Return SSE stream
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
        else:
            # Return JSON (non-streaming)
            if "/v1/messages" in self.endpoint:
                # Anthropic format
                full_content = ""
                for resp in self.json_response:
                    data = json.loads(resp)
                    if "delta" in data:
                        full_content += data["delta"].get("text", "")
                return web.json_response({
                    "id": "msg-123",
                    "model": "test-model",
                    "role": "assistant",
                    "content": [{"type": "text", "text": full_content}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "type": "message",
                    "usage": {"input_tokens": 10, "output_tokens": 20}
                })
            else:
                # OpenAI format
                full_content = ""
                for resp in self.json_response:
                    data = json.loads(resp)
                    if "choices" in data:
                        for choice in data["choices"]:
                            if "delta" in choice:
                                full_content += choice["delta"].get("content", "")
                return web.json_response({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": full_content
                        }
                    }]
                })

    @property
    def url(self) -> str:
        """Get the server URL."""
        return self._url


def create_openai_mock_server(reasoning: str = "", response: str = "Hello from mock LLM", port: int = 8765, enable_streaming: bool = True):
    """Create a mock server for OpenAI-compatible responses."""
    responses = []
    if reasoning:
        responses.append(json.dumps({"choices": [{"delta": {"reasoning": reasoning, "content": ""}}]}))
    responses.append(json.dumps({"choices": [{"delta": {"content": response}}]}))
    responses.append("[DONE]")

    return MockLLMServer(responses, port=port, endpoint="/v1/chat/completions", enable_streaming=enable_streaming)


def create_anthropic_mock_server(thinking: str = "Thinking step by step", response: str = "Hello", port: int = 8765, enable_streaming: bool = True):
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
    return MockLLMServer(responses, port=port, endpoint="/v1/messages", enable_streaming=enable_streaming)
