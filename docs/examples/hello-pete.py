"""Minimalistic Hello Pete example.

    Usage: python docs/examples/hello-pete.py <URL> [api_type] [api_key]
"""

import asyncio
import sys
from peteos.chatbot.manager import ChatBotManager
from peteos.oap.base import AgenticObjectBase


class HelloPete(AgenticObjectBase):
    """You are Pete, a helpful assistant."""


async def main():
    #from peteos.utils.logger import setup_logging
    #setup_logging(level="DEBUG", debug=True)
    args = sys.argv[1:] + [None] * 3
    url, api_type, api_key = args[:3]
    await ChatBotManager.add_backend("local", url, api_type=api_type, api_key=api_key)
    pete = HelloPete()
    result = await pete.invoke_agent("Hello, what's your name?")
    print(result)


asyncio.run(main())
