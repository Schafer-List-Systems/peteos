"""Example: FibonacciSquared with sandboxed code execution.

Usage: python fibonacci-squared.py <backend-url> [api_type] [api_key]
"""

import asyncio

from peteos.oap.adaptive_object import AdaptiveObject
from peteos.oap.decorators import agentic_object


@agentic_object(allow_code_execution=True)
class FibonacciSquared(AdaptiveObject):
    """
    You are an assistant for mathematical computations.
    - Provide results as precise as possible.
    - Use already existing functions whenever possible.
    """


async def main():
    # from peteos.utils.logger import setup_logging
    # setup_logging(level="DEBUG", debug=True)

    sq = FibonacciSquared()
    print(await sq.invoke_agent(
        "Compute the sequence where each element is the sum of the squares of its two predecessors."
        " Start with 0, 1. Compute the 7-th element. For example: the 6-th element shall be 29.",
        output_schema=int,
        persistent_thread_id='keep'
    ))
    print(await sq.invoke_agent(
        "Compute the sequence where each element is the sum of the squares of its two predecessors."
        " Start with 0, 1. Compute the 8-th element.",
        output_schema=int,
        persistent_thread_id='keep'
    ))
    print(await sq.invoke_agent(
        "Compute the sequence where each element is the sum of the squares of its two predecessors."
        " Start with 0, 1. Compute the 10-th element.",
        output_schema=int
    ))


if __name__ == "__main__":
    asyncio.run(main())
