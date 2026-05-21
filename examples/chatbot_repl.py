"""Example: ChatBot REPL with multi-turn conversation.

This example demonstrates how to use Session for a multi-turn
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

from peteos.agent import Agent
from peteos.chatbot.manager import ChatBotManager
from peteos.chatbot import Message, ContentPart
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager


async def main():
    # Configuration from environment
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    model = os.getenv("MODEL", "qwen3.5-35b")
    chat_protocol = os.getenv("CHAT_PROTOCOL", "anthropic").lower()

    print(f"Connecting to {base_url} with model {model}")
    print(f"Protocol: {chat_protocol}")
    print()

    # Create role manager with a role
    role_manager = RoleManager()
    role_manager.register_role(
        Role(name="test", description="Test role", model=model)
    )

    # Create chatbot manager
    chatbot_manager = ChatBotManager()
    await chatbot_manager.add_backend(chat_protocol, base_url)

    tool_manager = ToolManager()
    agent = Agent(role_manager, chatbot_manager, tool_manager)

    session = await agent.create_session("test")

    # Define the multi-turn conversation
    questions = [
        "What is the color of a leaf?",
        "What is the result when mixing it with blue?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Turn {i}: {question}")
        print('='*60)

        # Add user question to session
        session.push_event(Message(
            role="user",
            content=[ContentPart(part_type="text", text=question)]
        ))

        # Wait for response
        await asyncio.sleep(1)

        # Print accumulated response
        last_msg = session.chat_history.messages[-1]
        if last_msg.get_role() == "assistant" and last_msg.content:
            print(last_msg.content[0].text)

        session.execution_environment.clear_interrupt()

    await session.stop()


if __name__ == "__main__":
    asyncio.run(main())