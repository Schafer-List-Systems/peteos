"""Domain events for the peteos execution engine.

These events are published to channels so that channel implementations
can track the full lifecycle of tool executions and agent invocations.
"""

from dataclasses import dataclass, field
from enum import Enum

from peteos.conversation.message import ContentPart, Message


class ActivityState(str, Enum):
    """Runner activity state."""
    STOPPED = "stopped"
    SLEEPING = "sleeping"
    WAITING = "waiting"
    RUNNING = "running"


@dataclass(frozen=True)
class ActivityEvent:
    """Runner activity-state transition.

    Fires whenever the runner's activity state changes.
    """
    state: ActivityState


@dataclass(frozen=True)
class MessageEvent:
    """Human-readable message event.

    Triggered by the ``after_message_append`` and ``before_notification_publish`` hooks.
    """
    message: Message


@dataclass(frozen=True)
class ApprovalEvent:
    """Tool call approval decision.

    Triggered by the ``on_tool_call`` hook when a tool needs approval.
    """
    tool_call_id: str = ""
    tool_call: ContentPart = field(default_factory=lambda: ContentPart({"type": "tool_use"}))
    approved: bool = True
    denied_reason: str | None = None


@dataclass(frozen=True)
class ToolExecutionEvent:
    """Tool call execution status update.

    Triggered by the ``before_tool_execution`` and ``after_tool_execution`` hooks.
    Contains the tool_result ContentPart which carries the call_id, execution_status,
    and result content.
    """
    tool_result: ContentPart = field(
        default_factory=lambda: ContentPart({"type": "tool_result", "call_id": "", "content": ""})
    )
