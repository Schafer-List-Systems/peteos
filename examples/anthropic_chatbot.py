"""Example: Using AnthropicChatBot with a streaming response.

This example demonstrates how to use AnthropicChatBot to send a message
and receive a streaming response with thinking/reasoning content.

Requirements:
    Set environment variables:
        ANTHROPIC_COMPATIBLE_BASE_URL: Base URL of Anthropic-compatible API endpoint
        ANTHROPIC_COMPATIBLE_MODEL: Model identifier (e.g., "claude-3-opus")

Example with local endpoint:
    ANTHROPIC_COMPATIBLE_BASE_URL=http://localhost:8000 \\
    ANTHROPIC_COMPATIBLE_MODEL=claude-3-opus \\
    python examples/anthropic_chatbot.py

API Protocol: Anthropic-compatible
    Responses use "thinking" or "reasoning" keys for thinking/reasoning content
"""

import asyncio
import os

from peteos.chatbot import AnthropicChatBot
from peteos.httpclient import HTTPClient
from peteos.chathistory import ChatHistory
from peteos.message import Message


async def main():
    # Configuration from environment
    base_url = os.getenv("ANTHROPIC_COMPATIBLE_BASE_URL", "http://localhost:8000")
    model = os.getenv("ANTHROPIC_COMPATIBLE_MODEL", "claude-3-opus")

    print(f"Connecting to {base_url} with model {model}")

    # Create HTTP client with timeout
    http_client = HTTPClient(timeout=60.0)

    # Create ChatBot instance
    chatbot = AnthropicChatBot(
        http_client=http_client,
        model=model,
        base_url=base_url
    )

    # Build chat history with system prompt
    history = ChatHistory()
    history.append_message(Message(content={"role": "system", "content": "You are a helpful assistant that shows your thinking process."}))
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
