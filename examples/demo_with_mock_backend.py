#!/usr/bin/env python3
"""
Demo with Mock Backend

This example demonstrates a fully working interactive shell using a mock HTTP backend.
The mock backend responds to any message with a simple echo, allowing you to test
the complete agent flow without a real LLM backend.

Usage:
    PYTHONPATH=/home/frygge/projects/private/peteos python examples/demo_with_mock_backend.py
"""

import asyncio
from aiohttp import web
from peteos.agent import Agent
from peteos.channels import InteractiveShellChannel
from peteos.chatbot.manager import ChatBotManager
from peteos.logger import setup_logging
from peteos.role import Role
from peteos.toolmanager import ToolManager


async def create_mock_backend(port: int = 18888) -> tuple[web.AppRunner, int]:
    """Create and start a mock backend server."""
    async def handle_models(request: web.Request) -> web.Response:
        return web.json_response({
            "data": [
                {"id": "demo-model", "name": "Demo Model", "object": "model"}
            ]
        })

    async def handle_chat_completions(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        messages = body.get("messages", [])

        # Echo back the last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                text = f"Demo response to: {msg.get('content', '')}"
                break
        else:
            text = "Demo response"

        async def event_generator():
            yield f'data: {{"id": "demo", "object": "chat.completion", "created": 1234567890, "model": "demo-model", "choices": [{{"index": 0, "delta": {{"role": "assistant", "content": "{text}"}}}}], "usage": null}}\n'
            yield 'data: [DONE]\n'

        response = web.StreamResponse(status=200)
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['Cache-Control'] = 'no-cache'
        await response.prepare(request)
        async for chunk in event_generator():
            await response.write(chunk.encode())
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, "localhost", port)
    await site.start()

    return runner


async def setup_components():
    """Setup all components with mock backend."""
    # Start mock backend
    print("Starting mock backend on http://localhost:18888...")
    runner = await create_mock_backend(18888)

    # Setup role
    role = Role(name="demo", description="Demo role", model="demo-model")

    # Setup chatbot manager with mock backend
    ChatBotManager.reset()
    await ChatBotManager.add_backend("mock", "http://localhost:18888")

    # Setup tool manager with example tools
    tool_manager = ToolManager()

    def get_weather(city: str) -> str:
        return f"Sunny and 25°C in {city}"

    def calculate(expression: str) -> str:
        import ast, operator
        op_map = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
        def eval_expr(node):
            if isinstance(node, (ast.Num, ast.Constant)):
                return node.n if isinstance(node, ast.Num) else node.value
            elif isinstance(node, ast.BinOp):
                return op_map[type(node.op)](eval_expr(node.left), eval_expr(node.right))
            else:
                raise ValueError("Unsupported")
        return str(eval_expr(ast.parse(expression, mode="eval").body))

    tool_manager.register_tool(func=get_weather)
    tool_manager.register_tool(func=calculate)

    return runner, role, tool_manager


async def main():
    """Main entry point."""
    # Configure logging
    setup_logging(level="INFO", debug=True)

    print("=" * 60)
    print("  Peteos Demo with Mock Backend")
    print("=" * 60)
    print()

    # Setup components
    runner, role, tool_manager = await setup_components()
    print("Components ready!")
    print()

    # Create the Agent
    agent = Agent(role, tool_manager)
    await agent.start()
    print("Agent started")
    print()

    # Create the shell channel
    shell = InteractiveShellChannel("shell", runner)

    print("-" * 60)
    print("Commands: /new, /list, /switch, /approve, /deny, /pending, /image (or /file), /quit")
    print("Type any text to send a message to the active session.")
    print("-" * 60)
    print()

    # Run the shell
    await shell.run()

    print()
    print("Goodbye!")

    # Cleanup
    await agent.stop()
    await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
