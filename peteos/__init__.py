# Peteos - Agentic application framework

import asyncio

from peteos.config import ConfigManager

# OAP primitives
from peteos.oap import (
    AgenticObject,
    AdaptiveObject,
    Error,
    agentic_object,
    tool,
)

__all__ = [
    "AgenticObject",
    "AdaptiveObject",
    "Error",
    "agentic_object",
    "tool",
]

# Bootstrap backends and roles from peteos.json at import time.
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(ConfigManager.init())
