"""Example: FibonacciSeries with sandboxed code execution."""

import asyncio
import sys

from peteos.chatbot.manager import ChatBotManager
from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import agentic_object


@agentic_object(allow_code_execution=True)
class FibonacciSeries(AgenticObjectBase):
    """You are a helpful assistant."""


async def main():
    #from peteos.utils.logger import setup_logging
    #setup_logging(level="DEBUG", debug=True)
    url = sys.argv[1]
    await ChatBotManager.add_backend("local", url)

    seq = FibonacciSeries()
    result = await seq.invoke_agent(
        "Compute the sequence where each element is the sum of the squares of its two predecessors."
        " Start with 0, 1. And compute the 10-th element.",
        output_schema=int,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
