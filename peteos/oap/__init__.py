"""OAP - Object-Agentic Programming for peteos."""

from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import agentic_object, tool
from peteos.oap.error import Error

__all__ = [
    "AgenticObjectBase",
    "Error",
    "agentic_object",
    "tool",
]
