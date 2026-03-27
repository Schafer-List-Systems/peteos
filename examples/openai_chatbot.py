"""Example: Using OpenAIChatBot with a streaming response.

This example demonstrates how to use OpenAIChatBot to send a message
and receive a streaming response with reasoning/thinking content.

Requirements:
    Set environment variables:
        OPENAI_COMPATIBLE_BASE_URL: Base URL of OpenAI-compatible API endpoint
        OPENAI_COMPATIBLE_MODEL: Model identifier (e.g., "qwen3.5-35b")

Example with local endpoint:
    OPENAI_COMPATIBLE_BASE_URL=http://192.168.255.10:8123 \\
    OPENAI_COMPATIBLE_MODEL=qwen/qwen3.5-35b-a3b \\
    python examples/openai_chatbot.py

API Protocol: OpenAI-compatible
    Responses use "reasoning" key for thinking/reasoning content
"""

import asyncio
import os

from peteos.chatbot import OpenAIChatBot
from peteos.httpclient import HTTPClient
from peteos.chathistory import ChatHistory
from peteos.message import Message


async def main():
    # Configuration from environment
    base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000")
    model = os.getenv("OPENAI_COMPATIBLE_MODEL", "qwen3.5-35b")

    print(f"Connecting to {base_url} with model {model}")

    # Create HTTP client with timeout
    http_client = HTTPClient(timeout=60.0)

    # Create ChatBot instance
    chatbot = OpenAIChatBot(
        http_client=http_client,
        model=model,
        base_url=base_url
    )

    # Build chat history
    history = ChatHistory()
    history.append_message(Message(content={"role": "user", "content": "Hello! Please think step by step and explain your reasoning."}))

    # Send message with streaming
    print("\nRequesting response...\n")
    response = await chatbot.send_message(history, streaming=True)

    # Collect streaming response
    accumulated = []
    async for chunk in response:
        accumulated.append(chunk)
        print(chunk, end="", flush=True)

    print("\n" + "=" * 60)
    print("Final accumulated response:")
    print(response.text_content)
    print("\nThinking/Reasoning content:")
    print(response.thinking_content)


if __name__ == "__main__":
    asyncio.run(main())
