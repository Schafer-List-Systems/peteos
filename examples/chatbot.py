"""Example: ChatBot with streaming response.

This example demonstrates how to use ChatBot to send a message
and receive a streaming response with reasoning/thinking content.

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
    python examples/chatbot.py

API Protocol: Configurable (OpenAI or Anthropic)
    Responses use "reasoning"/"thinking" key for thinking content
"""

import asyncio
import os

from peteos.chatbot import OpenAIChatBot, AnthropicChatBot
from peteos.httpclient import HTTPClient
from peteos.chathistory import ChatHistory
from peteos.message import Message


async def main():
    # Configuration from environment
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    model = os.getenv("MODEL", "qwen3.5-35b")
    chat_protocol = os.getenv("CHAT_PROTOCOL", "anthropic").lower()
    use_streaming = os.getenv("USE_STREAMING", "true").lower() == "true"

    print(f"Connecting to {base_url} with model {model}")
    print(f"Protocol: {chat_protocol}")
    print(f"Streaming mode: {use_streaming}")
    if use_streaming:
        print("Note: Models that stream reasoning may take 30+ seconds before text appears.")

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

    # Build chat history with system prompt
    history = ChatHistory()
    history.append_message(Message(content={"role": "system", "content": "You are a helpful assistant."}))
    history.append_message(Message(content={"role": "user", "content": "Explain how photosynthesis works step by step."}))

    # Send message
    streaming_mode = "streaming" if use_streaming else "non-streaming"
    print(f"\nRequesting response... ({streaming_mode})\n")
    response = await chatbot.send_message(history, streaming=use_streaming)

    # Collect streaming response - yields (key, chunk) tuples
    async for key, chunk in response:
        # Print only text chunks (skip reasoning)
        if key == "text_content":
            print(chunk, end="", flush=True)

    print("\n" + "=" * 60)
    print("Final accumulated response:")
    print(response.data.get("text_content", ""))
    print("\nThinking/Reasoning content:")
    print(response.data.get("thinking_content", ""))


if __name__ == "__main__":
    asyncio.run(main())
