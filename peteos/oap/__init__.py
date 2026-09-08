"""OAP - Object-Agentic Programming for peteos."""

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.adaptive_object import AdaptiveObject
from peteos.oap.decorators import agentic_object, sandbox, tool
from peteos.oap.error import Error

__all__ = [
    "AdaptiveObject",
    "AgenticObject",
    "Error",
    "agentic_object",
    "sandbox",
    "tool",
]
