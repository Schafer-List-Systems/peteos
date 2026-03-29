"""Example: REPLExecutionEnvironment with tool calls.

This example demonstrates how to use REPLExecutionEnvironment with a tool manager
that provides trigonometric functions. The agent will solve triangles given three values.

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
    python examples/triangle_solver.py

Tool Example:
    Given three values of a triangle (e.g., "two sides and the included angle"),
    the agent will use the available tools to compute all remaining lengths and angles.
"""

import asyncio
import math
import os

from peteos.chatbot import OpenAIChatBot, AnthropicChatBot
from peteos.httpclient import HTTPClient
from peteos.chathistory import ChatHistory
from peteos.replexecutionenvironment import REPLExecutionEnvironment
from peteos.toolmanager import ToolManager


def sine(angle_degrees: float) -> float:
    """Calculate sine of an angle (in degrees)."""
    return math.sin(math.radians(angle_degrees))


def cosine(angle_degrees: float) -> float:
    """Calculate cosine of an angle (in degrees)."""
    return math.cos(math.radians(angle_degrees))


def tangent(angle_degrees: float) -> float:
    """Calculate tangent of an angle (in degrees)."""
    return math.tan(math.radians(angle_degrees))


def asin(value: float) -> float:
    """Calculate arcsine and return result in degrees."""
    return math.degrees(math.asin(value))


def acos(value: float) -> float:
    """Calculate arccosine and return result in degrees."""
    return math.degrees(math.acos(value))


def atan(value: float) -> float:
    """Calculate arctangent and return result in degrees."""
    return math.degrees(math.atan(value))


def main():
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

    # Create tool manager and register trigonometric tools
    tool_manager = ToolManager()

    print("Available tools:")
    print("  - sine(angle_degrees): Calculate sine")
    print("  - cosine(angle_degrees): Calculate cosine")
    print("  - tangent(angle_degrees): Calculate tangent")
    print("  - asin(value): Calculate arcsine (returns degrees)")
    print("  - acos(value): Calculate arccosine (returns degrees)")
    print("  - atan(value): Calculate arctangent (returns degrees)")
    print()

    # Register tools
    tool_manager.register_tool(func=sine)
    tool_manager.register_tool(func=cosine)
    tool_manager.register_tool(func=tangent)
    tool_manager.register_tool(func=asin)
    tool_manager.register_tool(func=acos)
    tool_manager.register_tool(func=atan)

    # Create chat history
    chat_history = ChatHistory()

    # Create execution environment
    env = REPLExecutionEnvironment(
        chatbot=chatbot,
        chat_history=chat_history,
        tool_manager=tool_manager
    )

    # Example problem: Given two sides and the included angle (SAS triangle)
    # Use Law of Cosines to find third side, then Law of Sines for other angles
    problem = """I have a triangle with:
- Side a = 5 units
- Side b = 7 units
- Angle C (between sides a and b) = 60 degrees

Please compute:
1. The length of side c
2. Angle A
3. Angle B

Use the trigonometric tools available to you."""

    print(f"Problem: {problem}\n")
    print("Solving triangle...\n")

    # Add system prompt
    from peteos.message import Message
    chat_history.append_message(Message(content={
        "role": "system",
        "content": "You are a helpful mathematics assistant. You have access to trigonometric tools (sine, cosine, tangent, asin, acos, atan) to compute triangle properties. Show your work step by step."
    }))

    # Add user question
    chat_history.append_message(Message(content={"role": "user", "content": problem}))

    # Run the REPL loop
    asyncio.run(env.run())

    # Print final result from chat history
    print("\n" + "="*60)
    print("Solution from conversation:")
    print("="*60)
    for msg in chat_history.messages:
        if msg.content.get("role") == "assistant":
            print(msg.content.get("content", ""))
            print()
    print("Solution complete!")
    print("="*60)


if __name__ == "__main__":
    main()
