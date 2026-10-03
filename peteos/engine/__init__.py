"""Engine — the runtime layer for agentic execution.

Provides the active-class event loop, message queueing, and execution
environment that drive agents to act.  Wraps a conversation.Session
(data model) with runtime behaviour: processing incoming events,
executing tools, and producing output.
"""

from peteos.engine.channel import Channel
from peteos.engine.events import (
    ActivityEvent,
    ActivityState,
    ApprovalEvent,
    MessageEvent,
    ToolExecutionEvent,
)
from peteos.utils.activeclass import ActiveClass
from peteos.engine.executionenvironment import (
    ExecutionEnvironment,
    RespondHandle,
    ToolApprovalStatus,
    ToolCallRecord,
    ToolExecutionStatus,
)
from peteos.engine.exec_status import ExecStatus
from peteos.conversation.session import SessionState
from peteos.engine.runner import Runner

__all__ = [
    "ActiveClass",
    "ActivityEvent",
    "ActivityState",
    "ApprovalEvent",
    "Channel",
    "ExecStatus",
    "ExecutionEnvironment",
    "MessageEvent",
    "RespondHandle",
    "Runner",
    "SessionState",
    "ToolCallGroup",
    "ToolApprovalStatus",
    "ToolCallRecord",
    "ToolExecutionStatus",
    "ToolExecutionEvent",
]
