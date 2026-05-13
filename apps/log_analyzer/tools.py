"""Log Analyzer tools — stub implementations for wake/sleep state and filtering."""

import uuid
from typing import TYPE_CHECKING

from peteos.toolmanager import ToolManager

if TYPE_CHECKING:
    from peteos.channels import NextcloudTalkChannel, ReadStdoutChannel


class LogState:
    """Shared mutable state for log analyzer tools."""

    _filters: list[str] = []
    channel: "ReadStdoutChannel | None" = None
    nextcloud_channel: "NextcloudTalkChannel | None" = None
    router_session_uuid: "uuid.UUID | None" = None

    def set_nextcloud_channel(self, ch: "NextcloudTalkChannel") -> None:
        self.nextcloud_channel = ch


_state = LogState()


def silence_router() -> str:
    """Stop the router from sending messages to Nextcloud.

    Use this when there are no serious issues to report. The router
    remains operational — it just does not send any output to the
    Nextcloud conversation. Call verbose_router when something needs
    to be reported.
    """
    ch = _state.nextcloud_channel
    session = _state.router_session_uuid
    if ch is not None and session is not None and not ch.is_session_silent(session):
        ch.set_session_silent(session, True)
        return "Silenced"
    return "Already silenced"


def verbose_router() -> str:
    """Allow the router to send messages to Nextcloud again.

    Use this when you have detected something serious that needs to be
    reported to the user (security issues, hardware failures, etc.).
    Once active, all your messages will be delivered via Nextcloud.
    Call silence_router when the conversation is done and no more
    output is needed.
    """
    ch = _state.nextcloud_channel
    session = _state.router_session_uuid
    if ch is not None and session is not None and ch.is_session_silent(session):
        ch.set_session_silent(session, False)
        return "Verbose"
    return "Already verbose"


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
    """Register router silence/verbose tools on the given tool manager.

    Args:
        tool_manager: The ToolManager to register the tools on.
    """
    tool_manager.register_tool(func=silence_router)
    tool_manager.register_tool(func=verbose_router)


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
