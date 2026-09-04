"""Minimalistic Hello Pete example.

    Usage: python3 examples/00_hello_pete.py <URL> [api_type] [api_key]
"""

import asyncio
from peteos import AgenticObject


class HelloPete(AgenticObject):
    """You are Pete, a helpful assistant."""

async def main():
    
    print("00_hello_pete.py -- Simple example of invoking an agent that tells you their name.")
    pete = HelloPete()
    result = await pete.invoke_agent("Hello, what's your name?")
    print(result)

asyncio.run(main())
