"""Log Analyzer tools — stub implementations for wake/sleep state and filtering."""

import logging
import uuid
from typing import TYPE_CHECKING

from peteos.toolmanager import ToolManager

if TYPE_CHECKING:
    from peteos.channels import NextcloudTalkChannel, ReadStdoutChannel

_logger = logging.getLogger(__name__)


class LogState:
    """Shared mutable state for log analyzer tools."""

    _filters: list[str] = []
    channel: "ReadStdoutChannel | None" = None
    nextcloud_channel: "NextcloudTalkChannel | None" = None
    router_session_uuid: "uuid.UUID | None" = None
    last_context_tokens: int = 0
    _previous_muted: bool | None = None

    def set_nextcloud_channel(self, ch: "NextcloudTalkChannel") -> None:
        self.nextcloud_channel = ch

    def status_text(self) -> str:
        """Return the status line for the system prompt.

        Logs a debug message when the muted status changes
        from the previous call.
        """
        is_muted = (
            self.nextcloud_channel
            and self.router_session_uuid
            and self.nextcloud_channel.is_session_muted(self.router_session_uuid)
        )
        if self._previous_muted is not None and is_muted != self._previous_muted:
            emoji = "🌙" if is_muted else "☀️"
            _logger.debug(
                "Muted status changed: %s -> %s %s",
                "muted" if self._previous_muted else "unmuted",
                "muted" if is_muted else "unmuted",
                emoji,
            )
        self._previous_muted = is_muted
        return f"Status: you are currently {'muted' if is_muted else 'unmuted'}.\n"


_state = LogState()


def mute_router() -> str:
    """Mute the router so it does not send messages to the user. Muted messages are not visible to the user.

    Use this when there are no serious issues to report. The router
    remains operational — it just does not send any output to the
    Nextcloud conversation. Call unmute_router when something needs
    to be reported.
    """
    ch = _state.nextcloud_channel
    session = _state.router_session_uuid
    was_muted = ch.is_session_muted(session) if ch and session else False
    if ch is not None and session is not None and not was_muted:
        ch.set_session_muted(session, True)
        _logger.debug("mute_router(): changed unmuted -> muted")
        return "Muted"
    _logger.debug("mute_router(): already muted")
    return "Already muted"


def unmute_router() -> str:
    """Unmute the router so it can send messages to the user.

    Use this when you have detected something serious that needs to be
    reported to the user (security issues, hardware failures, etc.).
    Once active, all your messages will be delivered via Nextcloud.
    Call mute_router when the conversation is done and no more
    output is needed.
    """
    ch = _state.nextcloud_channel
    session = _state.router_session_uuid
    was_muted = ch.is_session_muted(session) if ch and session else False
    if ch is not None and session is not None and was_muted:
        ch.set_session_muted(session, False)
        _logger.debug("unmute_router(): changed muted -> unmuted")
        return "Unmuted"
    _logger.debug("unmute_router(): already unmuted")
    return "Already unmuted"


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
    """Register router mute/unmute tools on the given tool manager.

    Args:
        tool_manager: The ToolManager to register the tools on.
    """
    tool_manager.register_tool(func=mute_router)
    tool_manager.register_tool(func=unmute_router)


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
