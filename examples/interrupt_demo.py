"""Example: Detecting REPLExecutionEnvironment running state.

This example demonstrates how to:
1. Check if the REPL is currently running via `is_running` property
2. Request an interrupt from another thread/task
3. Detect whether it's still running or has finished

Requirements:
    Set environment variables:
        BASE_URL: Base URL of API endpoint
        MODEL: Model identifier (e.g., "qwen/qwen3.5-35b")
        CHAT_PROTOCOL: "openai" or "anthropic" (default: "anthropic")
        USE_STREAMING: "true" (default) or "false" for non-streaming mode

Example:
    BASE_URL=http://192.168.255.10:8123 \
    MODEL=qwen/qwen3.5-35b-a3b \
    CHAT_PROTOCOL=anthropic \
    USE_STREAMING=true \
    python examples/interrupt_demo.py
"""

import asyncio
import os
import threading
import time

from peteos.chatbot import OpenAIChatBot, AnthropicChatBot
from peteos.httpclient import HTTPClient
from peteos.chathistory import ChatHistory
from peteos.replexecutionenvironment import REPLExecutionEnvironment
from peteos.toolmanager import ToolManager


def interrupt_after_delay(env: REPLExecutionEnvironment, delay: float):
    """Thread function that requests interrupt after a delay."""
    time.sleep(delay)
    print(f"\n[Thread] Requesting interrupt after {delay} seconds...")
    env.set_interrupt()


async def main():
    # Configuration from environment
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    model = os.getenv("MODEL", "qwen3.5-35b")
    chat_protocol = os.getenv("CHAT_PROTOCOL", "anthropic").lower()
    use_streaming = os.getenv("USE_STREAMING", "true").lower() == "true"

    print(f"Connecting to {base_url} with model {model}")
    print(f"Protocol: {chat_protocol}")
    print(f"Streaming mode: {use_streaming}")
    print()

    # Create HTTP client with timeout
    http_client = HTTPClient(timeout=60.0)

    # Create ChatBot instance based on protocol
    if chat_protocol == "openai":
        chatbot = OpenAIChatBot(
            http_client=http_client,
            model=model,
            base_url=base_url
        )
    else:  # default to anthropic
        chatbot = AnthropicChatBot(
            http_client=http_client,
            model=model,
            base_url=base_url
        )

    # Create chat history and tool manager
    chat_history = ChatHistory()
    tool_manager = ToolManager()

    # Create execution environment
    env = REPLExecutionEnvironment(
        chatbot=chatbot,
        chat_history=chat_history,
        tool_manager=tool_manager
    )

    # Add a long-running question that should take time to process
    question = """Please write a very detailed, lengthy explanation of
quantum mechanics. Include history, key experiments, mathematical
formulations, interpretations, and applications. Be as comprehensive
as possible."""

    # Add user question
    from peteos.message import Message
    chat_history.append_message(Message(content={"role": "user", "content": question}))

    print(f"Starting REPL loop with interrupt after 3 seconds...\n")

    # Start interrupt thread
    interrupt_thread = threading.Thread(
        target=interrupt_after_delay,
        args=(env, 3),
        daemon=True
    )
    interrupt_thread.start()

    # Check running state before starting
    print(f"Before start: is_running={env.is_running}")

    # Run the REPL loop
    await env.run()

    # Check running state after completion
    print(f"After completion: is_running={env.is_running}")

    # Check interrupt flag
    print(f"Interrupt requested: {env._interrupt}")

    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
