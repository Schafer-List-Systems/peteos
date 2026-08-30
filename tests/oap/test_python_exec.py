"""Test for OAP sandboxed code execution via FibonacciSquared."""

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import agentic_object


def _fibonacci_squared(n: int) -> int:
    """Compute the n-th element (zero-based) of the Fibonacci-squared sequence.

    Each element is the sum of the squares of its two predecessors,
    starting with 0, 1.
    """
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a * a + b * b
    return a


@agentic_object(allow_code_execution=True)
class FibonacciSquared(AgenticObject):
    """
    You are an assistant for mathematical computations.
    - Provide results as precise as possible.
    - If you have problems representing large numbers, then return large numbers as strings!
    """


async def test_python_exec_6th():
    agent = FibonacciSquared()
    result = await agent.invoke_agent(
        "Compute the sequence where each element is the sum of the squares of its two predecessors: 0:0, 1:1, 2:1, 3:2, ... "
        "Start with 0, 1. And compute the 6-th element.",
        output_schema=int,
    )
    expected = _fibonacci_squared(6)
    assert result == expected, f"6th element: expected {expected}, got {result}"


async def test_python_exec_8th():
    agent = FibonacciSquared()
    result = await agent.invoke_agent(
        "Compute the sequence where each element is the sum of the squares of its two predecessors: 0:0, 1:1, 2:1, 3:2, ... "
        "Start with 0, 1. And compute the 8-th element.",
        output_schema=int,
    )
    expected = _fibonacci_squared(8)
    assert result == expected, f"8th element: expected {expected}, got {result}"


async def test_python_exec_10th():
    agent = FibonacciSquared()
    result = await agent.invoke_agent(
        "Compute the sequence where each element is the sum of the squares of its two predecessors: 0:0, 1:1, 2:1, 3:2, ... "
        "Start with 0, 1. And compute the 10-th element.",
        output_schema=int,
    )
    expected = _fibonacci_squared(10)
    assert result == expected, f"10th element: expected {expected}, got {result}"
