"""OAP - Object-Agentic Programming for peteos."""

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.adaptive_object import AdaptiveObject
from peteos.oap.decorators import agentic_object, tool
from peteos.oap.error import Error
from peteos.oap.benchmark import (
    AgenticStringComparator,
    BenchmarkRow,
    BenchmarkReport,
    BenchmarkRunner,
)

__all__ = [
    "AdaptiveObject",
    "AgenticObject",
    "AgenticStringComparator",
    "BenchmarkReport",
    "BenchmarkRow",
    "BenchmarkRunner",
    "Error",
    "agentic_object",
    "tool",
]
