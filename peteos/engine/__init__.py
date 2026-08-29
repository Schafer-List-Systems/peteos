"""Engine — the runtime layer for agentic execution.

Provides the active-class event loop, message queueing, and execution
environment that drive agents to act.  Wraps a conversation.Session
(data model) with runtime behaviour: processing incoming events,
executing tools, and producing output.
"""

from peteos.engine.channel import Channel, NotificationEvent
from peteos.utils.activeclass import ActiveClass
from peteos.engine.executionenvironment import (
    ApprovalEvent,
    ExecutionEnvironment,
    ToolApprovalStatus,
    ToolCallRecord,
    ToolExecutionStatus,
)
from peteos.engine.exec_status import ExecStatus
from peteos.conversation.session import SessionState
from peteos.engine.runner import Runner

# Driver is kept as a deprecated alias for Runner
Driver = Runner

__all__ = [
    "ActiveClass",
    "ApprovalEvent",
    "Channel",
    "Driver",  # deprecated alias for Runner
    "ExecStatus",
    "ExecutionEnvironment",
    "NotificationEvent",
    "Runner",
    "SessionState",
    "ToolCallGroup",
    "ToolApprovalStatus",
    "ToolCallRecord",
    "ToolExecutionStatus",
]
