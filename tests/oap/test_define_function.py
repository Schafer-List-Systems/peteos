"""Test for OAP persistent thread via FibonacciSquared (AdaptiveObject)."""

from peteos.oap.adaptive_object import AdaptiveObject
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
class FibonacciSquared(AdaptiveObject):
    """
    You are an assistant for mathematical computations.
    - Provide results as precise as possible.
    - If you have problems representing large numbers, then return large numbers as strings!
    - Use already existing functions whenever possible.
    """


async def test_define_function_7th():
    agent = FibonacciSquared()
    result = await agent.invoke_agent(
        "Compute the sequence where each element is the sum of the squares of its two predecessors: 0:0, 1:1, 2:1, 3:2, ... "
        "Compute the 7-th element.",
        output_schema=int,
        persistent_thread_id="define_func_7",
    )
    expected = _fibonacci_squared(6)
    assert result == expected, f"7th element: expected {expected}, got {result}"


async def test_define_function_8th_persistent():
    """Compute 8th element in the same persistent thread as the 7th."""
    agent = FibonacciSquared()
    result = await agent.invoke_agent(
        "Compute the sequence where each element is the sum of the squares of its two predecessors: 0:0, 1:1, 2:1, 3:2, ... "
        "Compute the 8-th element.",
        output_schema=int,
        persistent_thread_id="define_func_7",
    )
    expected = _fibonacci_squared(7)
    assert result == expected, f"8th element (persistent): expected {expected}, got {result}"


async def test_define_function_10th_new_thread():
    """Compute 10th element in a fresh thread."""
    agent = FibonacciSquared()
    result = await agent.invoke_agent(
        "Compute the sequence where each element is the sum of the squares of its two predecessors: 0:0, 1:1, 2:1, 3:2, ... "
        "Compute the 10-th element.",
        output_schema=int,
    )
    expected = _fibonacci_squared(9)
    assert result == expected, f"10th element: expected {expected}, got {result}"
