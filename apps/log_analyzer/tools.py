"""Log Analyzer tools — stub implementations for wake/sleep state and filtering."""

from typing import TYPE_CHECKING

from peteos.toolmanager import ToolManager

if TYPE_CHECKING:
    from peteos.channels import ReadStdoutChannel


class LogState:
    """Shared mutable state for log analyzer tools."""

    awake: bool = False
    _filters: list[str] = []
    channel: "ReadStdoutChannel | None" = None


_state = LogState()


def wake_up() -> str:
    """Wake up the router and set the system to awake state.

    Returns:
        A status message: 'Awake' on transition, 'Already awake' if no-op.
    """
    if _state.awake:
        return "Already awake"
    _state.awake = True
    return "Awake"


def sleep() -> str:
    """Put the router to sleep and set the system to sleep state.

    Returns:
        A status message: 'Asleep' on transition, 'Already asleep' if no-op.
    """
    if not _state.awake:
        return "Already asleep"
    _state.awake = False
    return "Asleep"


def add_exclude_pattern(pattern: str) -> str:
    """Add a regex exclude pattern to filter out future log messages.

    Lines matching an exclude pattern are dropped. Use this to suppress
    unwanted noise from spamming you in the future.

    Args:
        pattern: Regular expression pattern to exclude.

    Returns:
        Status message with the index, or that the pattern already exists.
    """
    ch = _state.channel
    if ch is None:
        return "Error: channel not configured."
    index = ch.add_exclude_pattern(pattern)
    if index is not None:
        return f"Added exclude pattern at index {index}: {pattern!r}"
    return f"Pattern {pattern!r} already exists"


def remove_exclude_pattern(index: int, pattern: str) -> str:
    """Remove an exclude pattern from the exclusion list.

    Args:
        index: Position of the pattern to remove.
        pattern: The pattern to remove (sanity check).

    Returns:
        Status message: 'Removed', 'Invalid index', or mismatch warning.
    """
    ch = _state.channel
    if ch is None:
        return "Error: channel not configured."
    try:
        patterns = ch.list_exclude_patterns()
        if index < 0 or index >= len(patterns):
            return f"Invalid index: {index}"
        if patterns[index] != pattern:
            return f"Mismatch: pattern at index {index} is {patterns[index]!r}, not {pattern!r}. Check list_exclude_patterns() and try again."
    except Exception:
        return f"Invalid index: {index}"
    ch.remove_exclude_pattern(index)
    return f"Removed pattern at index {index}: {pattern!r}"


def list_exclude_patterns() -> str:
    """List all current exclude patterns.

    Returns:
        Formatted list of active exclude patterns with their indices.
    """
    ch = _state.channel
    if ch is None:
        return "Error: channel not configured."
    patterns = ch.list_exclude_patterns()
    if not patterns:
        return "No exclude patterns configured."
    lines = [f"  [{i}] {p!r}" for i, p in enumerate(patterns)]
    return "Exclude patterns:\n" + "\n".join(lines)


def register_state_tools(tool_manager: ToolManager) -> None:
    """Register wake/sleep state tools on the given tool manager.

    Args:
        tool_manager: The ToolManager to register the tools on.
    """
    tool_manager.register_tool(func=wake_up)
    tool_manager.register_tool(func=sleep)


def register_filter_tools(
    tool_manager: ToolManager,
    channel: "ReadStdoutChannel",
) -> None:
    """Register exclude pattern management tools on the given tool manager.

    Args:
        tool_manager: The ToolManager to register the tools on.
        channel: The ReadStdoutChannel instance whose exclude patterns
            will be managed.
    """
    _state.channel = channel
    tool_manager.register_tool(func=add_exclude_pattern)
    tool_manager.register_tool(func=remove_exclude_pattern)
    tool_manager.register_tool(func=list_exclude_patterns)
