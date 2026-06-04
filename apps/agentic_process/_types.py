from dataclasses import dataclass

from enum import Enum


class TaskState(Enum):
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    OK = "ok"
    DENIED = "denied"
    PENDING = "pending"
    DISABLED = "disabled"


class EdgeState(Enum):
    SCHEDULED = "scheduled"
    ENABLED = "enabled"
    DISABLED = "disabled"


@dataclass
class TaskStatus:
    """Structured output for task decision.

    Args:
        decision: One of 'ready', 'deny', or 'pending'.
    """
    decision: str


@dataclass
class EdgeEvaluation:
    """Result of evaluating one outgoing edge condition.

    Args:
        edge_id: The ID of the edge being evaluated.
        met: Whether the edge condition is satisfied.
    """
    edge_id: str
    met: bool
