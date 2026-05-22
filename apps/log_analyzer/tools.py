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
    last_context_tokens: int = 0
    is_muted: bool = False
    session: "Session | None" = None  # Set after session creation


_state = LogState()


def mute_router() -> str:
    """Mute the router so it does not send messages to the user. Muted messages are not visible to the user.

    Use this when there are no serious issues to report. The router
    remains operational — it just does not send any output to the
    Nextcloud conversation. Call unmute_router when something needs
    to be reported.
    """
    if _state.is_muted:
        _logger.debug("mute_router(): already muted")
        return "Already muted"
    _state.is_muted = True
    _logger.debug("mute_router(): changed unmuted -> muted")
    return "Muted"


def unmute_router() -> str:
    """Unmute the router so it can send messages to the user.

    Use this when you have detected something serious that needs to be
    reported to the user (security issues, hardware failures, etc.).
    Once active, all your messages will be delivered via Nextcloud.
    Call mute_router when the conversation is done and no more
    output is needed.
    """
    if not _state.is_muted:
        _logger.debug("unmute_router(): already unmuted")
        return "Already unmuted"
    _state.is_muted = False
    _logger.debug("unmute_router(): changed muted -> unmuted")
    return "Unmuted"


def add_exclude_pattern(pattern: str, reason: str = "", triggering_log_line: str = "") -> str:
    """Add a regex pattern to exclude future log messages from further observation.

    Lines matching an exclude pattern are dropped. Use this to suppress
    unwanted noise from spamming th log.

    Patterns must start with '^' and '.*' is only allowed at the very end.

    Before adding a pattern, reason about whether it could ever hide a
    security issue or hardware failure in any future log message. If
    you cannot confidently explain why it is safe, do not add it.

    Always provide the actual log line that triggered this pattern so the
    tool can verify the pattern matches the line that caused you to add it.

    Args:
        pattern: Regular expression pattern to exclude.
        reason: Your explanation of why this pattern is safe and will not hide future security or hardware issues.
        triggering_log_line: The actual log entry that caused you to want this pattern. The tool will verify the pattern matches this line.

    Returns:
        Status message with the index, or an error if the pattern is invalid or does not match the triggering log line.
    """
    ch = _state.channel
    if ch is None:
        return "Error: channel not configured."

    # Pattern must start with '^'
    if not pattern.startswith("^"):
        return "Pattern not included. Pattern must start with '^' to match the beginning of the line."

    # .* may only appear once, at the very end of the pattern (before optional $)
    stripped = pattern.rstrip("$")
    if ".*" in stripped:
        prefix, rest = stripped.split(".*", 1)
        if rest or ".*" in prefix:
            return "Pattern not included. Avoid matching arbitrary strings (.*) except at the end of the pattern."

    # Verify pattern matches the actual triggering log line
    import re
    if triggering_log_line and not re.search(pattern, triggering_log_line):
        return f"Pattern does not match the triggering log line. Test with: re.search({pattern!r}, {triggering_log_line!r})"

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


namespaces = {}

def eval_python(python_string: str, namespace_name: str = "") -> str:
    """Execute Python code and return stdout and return_value.

    The return value is captured by setting _result in the code.
    Use the same namespace_name across calls to maintain state (variables defined in one call are available in subsequent calls).
    Omit namespace_name or pass '' for a fresh anonymous namespace destroyed after each call.
    Pass 'globals' to execute in the module's global namespace (sharing module-level imports and definitions).
    Pass a named namespace_name for persistent state.

    Args:
        python_string: A string containing valid Python code to execute.
        namespace_name: The namespace name for state persistence. Empty string for ephemeral (default).
    """
    import io
    import sys

    if namespace_name == "globals":
        ns: dict = globals()
    elif namespace_name == "":
        ns = {}
    else:
        ns = namespaces.get(namespace_name)
        if ns is None:
            namespaces[namespace_name] = {}
            ns = namespaces[namespace_name]

    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    return_value = None
    try:
        sys.stdout = stdout_capture
        code = compile(python_string, "<eval>", "exec")
        exec(code, ns)
        return_value = ns.get("_result")
    except Exception as e:
        return_value = f"Error: {type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout

    stdout = stdout_capture.getvalue()
    return f"stdout: {stdout!r}\nreturn_value: {return_value!r}"


def register_state_tools(tool_manager: ToolManager) -> None:
    """Register state tools on the given tool manager.

    Args:
        tool_manager: The ToolManager to register the tools on.
    """
    tool_manager.register_tool(func=mute_router)
    tool_manager.register_tool(func=unmute_router)
    tool_manager.register_tool(func=eval_python)


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
