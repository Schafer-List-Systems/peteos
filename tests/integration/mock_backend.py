"""Mock HTTP backend for testing the agent end-to-end.

This simple HTTP server simulates a chatbot backend that:
1. Accepts POST /v1/models requests and returns a list of models
2. Accepts POST /v1/chat/completions requests and returns a predefined response
"""

import json
import asyncio
from aiohttp import web


async def handle_models(request: web.Request) -> web.Response:
    """Return list of available models."""
    models = [
        {"id": "test-model", "name": "Test Model", "object": "model"}
    ]
    return web.json_response({"models": models})


async def handle_chat_completions(request: web.Request) -> web.StreamResponse:
    """Handle chat completion requests.

    Returns SSE-formatted response for proper streaming.
    """
    body = await request.json()
    messages = body.get("messages", [])

    # Echo back the last user message as a simple response
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = f"I received your message: {msg.get('content', '')}"
            break
    else:
        text = "I am a test response"

    async def event_generator():
        yield 'data: {"id": "chatcmpl-test", "object": "chat.completion", "created": 1234567890, "model": "' + body.get("model", "test-model") + '", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "' + text + '"}}], "usage": null}\n'
        yield 'data: [DONE]\n'

    response = web.StreamResponse(status=200)
    response.headers['Content-Type'] = 'text/event-stream'
    response.headers['Cache-Control'] = 'no-cache'
    await response.prepare(request)
    async for chunk in event_generator():
        await response.write(chunk.encode())
    await response.write_eof()
    return response


async def handle_root(request: web.Request) -> web.Response:
    """Root endpoint info."""
    return web.json_response({
        "status": "ok",
        "endpoints": ["/v1/models", "/v1/chat/completions"]
    })


def create_app() -> web.Application:
    """Create the mock backend application."""
    app = web.Application()
    app.router.add_get("/", handle_root)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    return app


async def run_server(port: int = 18888):
    """Run the mock backend server."""
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, "localhost", port)
    await site.start()

    print(f"Mock backend running at http://localhost:{port}")
    print("Endpoints:")
    print(f"  GET  http://localhost:{port}/")
    print(f"  GET  http://localhost:{port}/v1/models")
    print(f"  POST http://localhost:{port}/v1/chat/completions")
    print("\nPress Ctrl+C to stop")

    # Keep running
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18888
    asyncio.run(run_server(port))
