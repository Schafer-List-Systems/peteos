# Peteos - Agentic application framework

import asyncio

from peteos.config import ConfigManager

from peteos.conversation import  (
    ContentPart,
    Message,
    Context
)

# OAP primitives
from peteos.oap import (
    AgenticObject,
    AdaptiveObject,
    Error,
    agentic_object,
    tool,
)
from peteos.oap.decorators import sandbox
from peteos.engine import (
    Channel,
    ExecStatus,
    NotificationEvent,
    ToolApprovalStatus,
    ToolExecutionStatus,
)

__all__ = [
    # Conversation
    "ContentPart",
    "Message",
    "Context",

    # OAP primitives
    "AgenticObject",
    "AdaptiveObject",
    
    # invocation and hooks
    "Error",
    "ExecStatus",

    # decorators
    "agentic_object",
    "tool",
    "sandbox",

    # tool execution
    "ToolApprovalStatus",
    "ToolExecutionStatus",

    # channels
    "Channel",
    "NotificationEvent",
]

# Bootstrap backends and roles from peteos.json at import time.
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(ConfigManager.init())
