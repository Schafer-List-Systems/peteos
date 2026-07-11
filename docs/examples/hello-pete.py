"""Minimalistic Hello Pete example.

    Usage: python docs/examples/hello-pete.py <URL> [api_type] [api_key]
"""

import asyncio
import sys
from peteos.chatbot.manager import ChatBotManager
from peteos.oap.agentic_object import AgenticObject


class HelloPete(AgenticObject):
    """You are Pete, a helpful assistant."""


async def main():
    #from peteos.utils.logger import setup_logging
    #setup_logging(level="DEBUG", debug=True)

    pete = HelloPete()
    result = await pete.invoke_agent("Hello, what's your name?")
    print(result)


asyncio.run(main())
