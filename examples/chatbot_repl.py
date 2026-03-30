"""Example: ChatBot REPL with multi-turn conversation.

This example demonstrates how to use REPLExecutionEnvironment for a multi-turn
conversation where the model remembers previous responses.

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
    python examples/chatbot_repl.py

Note:
    This example uses hardcoded questions to verify memory:
    1. "What is the color of a leaf?"
    2. "What is the result when mixing it with blue?"
"""

import asyncio
import os

from peteos.chatbot import OpenAIChatBot, AnthropicChatBot
from peteos.httpclient import HTTPClient
from peteos.chathistory import ChatHistory
from peteos.replexecutionenvironment import REPLExecutionEnvironment
from peteos.toolmanager import ToolManager


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

    # Define the multi-turn conversation
    questions = [
        "What is the color of a leaf?",
        "What is the result when mixing it with blue?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Turn {i}: {question}")
        print('='*60)

        # Add user question to history
        from peteos.message import Message
        chat_history.append_message(Message(content={"role": "user", "content": question}))

        # Run the REPL loop
        await env.run()

        # Print accumulated response
        from peteos.message import Message
        # Find the last assistant message in chat history
        last_msg = chat_history.messages[-1]
        if last_msg.content.get("role") == "assistant":
            text = last_msg.content.get("text", "")
            print(text)

        # Clear interrupt flag for next turn
        env.clear_interrupt()

        # Reset interrupt flag
        env._interrupt = False


if __name__ == "__main__":
    asyncio.run(main())
