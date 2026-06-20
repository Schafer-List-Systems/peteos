"""ExecStatus — return codes for the execution loop."""

from enum import Enum


class ExecStatus(str, Enum):
    """Status returned by ``REPLExecutionEnvironment.step()`` and the driver loop."""

    FINISHED = "finished"
    CONTINUE = "continue"
    PENDING = "pending"
    ERROR = "error"
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_FAILED = "tool_failed"
    TOOL_DENIED = "tool_denied"


_EXEC_SEVERITY: dict[ExecStatus | None, int] = {
    None: -1,
    ExecStatus.CONTINUE: 0,
    ExecStatus.FINISHED: 1,
    ExecStatus.PENDING: 2,
    ExecStatus.TOOL_NOT_FOUND: 3,
    ExecStatus.TOOL_DENIED: 4,
    ExecStatus.TOOL_FAILED: 5,
    ExecStatus.ERROR: 6,
}


def _merge_exec_status(a: ExecStatus | None, b: ExecStatus | None) -> ExecStatus | None:
    """Merge two ExecStatus values by severity (worst wins).

    This is commutative and associative — merging in any order yields
    the same result.  ``None`` acts as the identity element.
    """
    if a is None:
        return b
    if b is None:
        return a
    return a if _EXEC_SEVERITY[a] >= _EXEC_SEVERITY[b] else b