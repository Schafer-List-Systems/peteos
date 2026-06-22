"""Engine — the runtime layer for agentic execution.

Provides the active-class event loop, message queueing, and execution
environment that drive agents to act.  Wraps a conversation.Session
(data model) with runtime behaviour: processing incoming events,
executing tools, and producing output.
"""

from peteos.utils.activeclass import ActiveClass
from peteos.engine.executionenvironment import ExecutionEnvironment, ToolCallGroup
from peteos.engine.exec_status import ExecStatus
from peteos.engine.runner import (
    AgenticState,
    ApprovalEvent,
    Runner,
    ToolApprovalStatus,
    ToolCallRecord,
    ToolExecutionStatus,
)

# Driver is kept as a deprecated alias for Runner
Driver = Runner

__all__ = [
    "ActiveClass",
    "AgenticState",
    "ApprovalEvent",
    "Driver",  # deprecated alias for Runner
    "ExecStatus",
    "ExecutionEnvironment",
    "Runner",
    "ToolCallGroup",
    "ToolApprovalStatus",
    "ToolCallRecord",
    "ToolExecutionStatus",
]
