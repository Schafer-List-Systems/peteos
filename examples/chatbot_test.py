#!/usr/bin/env python3
"""
Simple ChatBot Test Script

This script tests the chatbot functionality in isolation, without the
complex agent and channel infrastructure.

Usage:
    PYTHONPATH=/home/frygge/projects/private/peteos python examples/chatbot_test.py
"""

import asyncio
import json
import logging
import os
import random
from peteos.chatbot.manager import ChatBotManager
# GenericChatBot removed - subclasses derive from ChatBot directly
from peteos.chatbot.chathistory import ChatHistory
from peteos.chatbot.message import Message
from peteos.chatbot.contentpart import ContentPart
from peteos.logger import setup_logging
from peteos.toolmanager import Tool, ToolManager

# Configure logging for this script
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def setup_chatbot_manager(config_file: str = "config/chatbot_config.json"):
    """Setup ChatBotManager from configuration file."""
    ChatBotManager.reset()
    try:
        await ChatBotManager.load_from_file(config_file)
        print(f"Loaded backend configuration from {config_file}")
    except FileNotFoundError:
        print(f"Note: Config file {config_file} not found.")
        print("Creating empty ChatBotManager...")


def setup_tool_manager() -> ToolManager:
    """Setup ToolManager with a random tool for dice rolling."""
    tool_manager = ToolManager()

    def random_number(min_val: int = 1, max_val: int = 10) -> int:
        """Generate a random number between min_val and max_val (inclusive).

        Args:
            min_val: Minimum value (default: 1)
            max_val: Maximum value (default: 10)

        Returns:
            Random integer between min_val and max_val
        """
        return random.randint(min_val, max_val)

    tool_manager.register_tool(Tool.from_callable(random_number))
    return tool_manager


async def main():
    """Main entry point."""
    setup_logging(level="INFO", debug=True)

    print("=" * 60)
    print("  ChatBot Test Script")
    print("=" * 60)
    print()

    # Setup chatbot manager
    await setup_chatbot_manager()

    # Get a ChatBot from the first available backend/model
    if not ChatBotManager._backends:
        print("No backends configured. Exiting.")
        return

    # Get backend - support TEST_BACKEND environment variable to specify which backend to test
    backend_name = os.environ.get("TEST_BACKEND")
    if backend_name and backend_name in ChatBotManager._backends:
        print(f"Testing backend: {backend_name}")
    elif not backend_name:
        backend_name = list(ChatBotManager._backends.keys())[0]
        print(f"No TEST_BACKEND specified, using first backend: {backend_name}")
    else:
        print(f"Backend '{backend_name}' not found. Available backends: {list(ChatBotManager._backends.keys())}")
        return

    backend_info = ChatBotManager._backends[backend_name]

    # Get first model from backend
    model_name = list(backend_info.models.keys())[0]
    chatbot = backend_info.models[model_name]

    print(f"Using backend: {backend_name} ({backend_info.api_type})")
    print(f"Model: {model_name}")
    print()

    print("ChatBot initialized")
    print()

    # Setup tool manager with random tool
    tool_manager = setup_tool_manager()

    # Create chat history with tool definitions
    chat_history = ChatHistory()
    chat_history.append_message(Message(
        role="system",
        content=[ContentPart(part_type="text", text="You are a helpful assistant.")]
    ))

    # Add tool definitions to chat history
    tool_list = tool_manager.get_tool_list()
    for tool in tool_list:
        chat_history.append_message(Message(
            role="tool",
            content=[ContentPart(
                part_type="tool",
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters
            )]
        ))

    # Interactive loop
    print("Type your messages. Type '/quit' to exit.")
    print("-" * 60)
    print()

    while True:
        # Read input
        user_input = input("You: ").strip()

        if user_input.lower() == "/quit":
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Add user message to history
        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text=user_input)]
        ))

        # Send to chatbot
        print()
        print("Assistant:", end=" ")
        print("-" * 60)

        # Accumulate response
        response = await chatbot.send_message(chat_history, streaming=True)

        # Consume the response generator to accumulate data
        async for _ in response:
            pass

        # Get accumulated data from response
        response_data = response.data
        reasoning = response_data.get("reasoning", "")
        text = response_data.get("text", "")

        # Extract tool calls from uniform content array
        # Tool calls are stored in content array with type="tool_use"
        content_array = response_data.get("content", [])
        tool_calls = [
            item for item in content_array
            if isinstance(item, dict) and item.get("type") == "tool_use"
        ]

        # Print reasoning if present
        if reasoning:
            print(f"[Reasoning: {reasoning}...]")

        # Print text
        if text:
            print(text)
            print()

        # Print tool calls if present
        if tool_calls:
            print(f"[Tool calls: {json.dumps(tool_calls, indent=2)}]")
            print()

        print()

        # Add assistant response to history
        content_parts = []
        if text:
            content_parts.append(ContentPart(part_type="text", text=text))
        if reasoning:
            content_parts.append(ContentPart(part_type="reasoning", reasoning=reasoning))
        if tool_calls:
            # Convert uniform tool call format back to API-specific format if needed
            content_parts.append(ContentPart(part_type="tool_calls", tool_calls=tool_calls))

        chat_history.append_message(Message(
            role="assistant",
            content=content_parts
        ))

        print("-" * 60)
        print()


if __name__ == "__main__":
    asyncio.run(main())
