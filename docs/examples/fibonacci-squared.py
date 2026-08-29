"""Example: FibonacciSquared with sandboxed code execution.

Usage: python fibonacci-squared.py <backend-url> [api_type] [api_key]
"""

import asyncio

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import agentic_object


@agentic_object(allow_code_execution=True)
class FibonacciSquared(AgenticObject):
    """
    You are an assistant for mathematical computations.
    - Provide results as precise as possible.
    - If you have problems representing large numbers, then return large numbers as strings!
    """


async def main():
    sq = FibonacciSquared()
    result = await sq.invoke_agent(
        "Compute the sequence where each element is the sum of the squares of its two predecessors."
        " Start with 0, 1. And compute the 10-th element.",
        output_schema=int,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
