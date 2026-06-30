"""Minimalistic Hello Pete example."""

import asyncio
import sys
from peteos.chatbot.manager import ChatBotManager
from peteos.oap.base import AgenticObjectBase


class HelloPete(AgenticObjectBase):
    """You are Pete, a helpful assistant."""


async def main():
    #rom peteos.utils.logger import setup_logging
    #setup_logging(level="DEBUG", debug=True)
    url = sys.argv[1]
    await ChatBotManager.add_backend("local", url)
    pete = HelloPete()
    result = await pete.invoke_agent("Hello, what's your name?")
    print(result)


asyncio.run(main())
