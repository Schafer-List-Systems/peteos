"""Mock backend for OAP examples — returns structured SSE responses."""

import json
from aiohttp import web


async def create_mock_backend(
    responses: dict[str, str],
    port: int = 18889,
) -> tuple[web.AppRunner, dict[str, str]]:
    """Start a mock LLM backend that returns predefined responses.

    Each key in ``responses`` matches against the incoming prompt (case-insensitive).
    The first match wins; a key of ``"*"`` is the default fallback.

    Args:
        responses: Mapping of prompt substrings to expected response strings.
        port: TCP port to bind.

    Returns:
        (runner, responses) so the caller can inspect what was echoed.
    """
    async def handle_models(request: web.Request) -> web.Response:
        return web.json_response({
            "data": [
                {"id": "oap-model", "name": "OAP Model", "object": "model"}
            ]
        })

    async def handle_chat(request: web.Request) -> web.Response:
        body = await request.json()
        messages = body.get("messages", [])

        # Find the first user message content and match it to a response
        user_texts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                user_texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_texts.append(part.get("text", ""))

        # Match the last user text to a response
        response_text = None
        for text in reversed(user_texts):
            for key, value in responses.items():
                if key == "*" or key.lower() in text.lower():
                    response_text = value
                    break
            if response_text is not None:
                break

        if response_text is None:
            response_text = json.dumps({"success": True, "message": "done"})

        event = json.dumps({
            "id": "oap-chat",
            "object": "chat.completion",
            "created": 1716796800,
            "model": "oap-model",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": response_text},
            }],
            "usage": None,
        })

        async def event_generator():
            yield f"data: {event}\n"
            yield "data: [DONE]\n"

        resp = web.StreamResponse(status=200)
        resp.headers["Content-Type"] = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        await resp.prepare(request)
        async for chunk in event_generator():
            await resp.write(chunk.encode())
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", port)
    await site.start()

    return runner, responses