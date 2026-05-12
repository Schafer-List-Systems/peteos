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
        A status message indicating the router is now awake.
    """
    _state.awake = True
    return "Awake"


def sleep() -> str:
    """Put the router to sleep and set the system to sleep state.

    Returns:
        A status message indicating the router is now asleep.
    """
    _state.awake = False
    return "Asleep"


def add_exclude_pattern(pattern: str) -> str:
    """Add a regex exclude pattern to filter out log messages.

    Lines matching an exclude pattern are dropped. Use this to suppress
    unwanted noise.

    Args:
        pattern: Regular expression pattern to exclude.

    Returns:
        Status message.
    """
    ch = _state.channel
    if ch is None:
        return "Error: channel not configured."
    if ch.add_exclude_pattern(pattern):
        return f"Added exclude pattern: {pattern!r}"
    return f"Pattern {pattern!r} already exists"


def remove_exclude_pattern(index: int) -> str:
    """Remove an exclude pattern by its index in the list.

    Args:
        index: Position of the pattern to remove.

    Returns:
        Status message.
    """
    ch = _state.channel
    if ch is None:
        return "Error: channel not configured."
    if ch.remove_exclude_pattern(index):
        return f"Removed pattern at index {index}"
    return f"Invalid index: {index}"


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
