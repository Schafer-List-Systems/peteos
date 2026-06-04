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
    decision: str
