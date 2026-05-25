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
    agent: "Agent | None" = None  # Set after session creation


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


async def add_exclude_pattern(pattern: str, reason: str = "", triggering_log_line: str = "") -> str:
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
    if not pattern.startswith("^") or not pattern.endswith("$"):
        return "Pattern not included. Pattern must start with '^' to match the beginning of the line and end with '$' to match the end of the line."

    # .* may only appear once, at the very end of the pattern (before optional $)
    stripped = pattern.rstrip("$")
    if ".*" in stripped:
        return "Pattern not included. Avoid matching arbitrary strings (.*)."

    # Verify pattern matches the actual triggering log line
    import re
    import asyncio
    if triggering_log_line and not re.search(pattern, triggering_log_line):
        return f"Pattern does not match the triggering log line.\n- Pattern: '{pattern!r}'\n- Triggering log line: {triggering_log_line}"

    # --- Invoke pattern_reviewer for approval ---
    agent = _state.agent
    if agent is None:
        return "Error: agent not configured."

    try:
        from peteos.session import invoke_agent

        existing_patterns = ""
        if ch is not None:
            patterns_list = ch.list_exclude_patterns()
            if patterns_list:
                existing_patterns = "Current exclude patterns:\n" + "\n".join(
                    f"  [{i}] p" for i, p in enumerate(patterns_list)
                ) + "\n\n"

        reviewer_prompt = (
            f"Review this proposed log exclusion pattern for safety.\n\n"
            f"Proposed pattern: {pattern}\n"
            f"Stated reason: {reason}\n"
            f"Triggering log line: {triggering_log_line}\n\n"
            f"{existing_patterns}"
        )

        review_result = await invoke_agent(
            role_name="pattern_reviewer",
            prompt=reviewer_prompt,
            agent=agent,
            keep_session=True,
            timeout=120,
        )
    except asyncio.TimeoutError:
        return "Pattern addition aborted: reviewer did not respond within timeout."

    approved = review_result["session"].state.get("approved") if review_result.get("session") else None
    reviewer_reason = review_result["session"].state.get("reason") if review_result.get("session") else "Timeout - no result"

    if approved != "yes":
        return f"Pattern matches but is denied. Improve your pattern! Reason: {reviewer_reason}"

    index = ch.add_exclude_pattern(pattern)
    if index is not None:
        return f"Added exclude pattern at index {index}: {pattern}"
    return f"Pattern already existed: {pattern}"


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
            return f"Mismatch: pattern at index {index} is {patterns[index]}, not {pattern!r}. Check list_exclude_patterns() and try again."
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


def fold(message_ids: list[str], summary: str) -> str:
    """Fold consecutive messages into a single summary message.

    Takes a list of message IDs, finds them in the session's unanchored
    messages, verifies they are consecutive, removes them, and inserts
    a single FoldedMessage with the provided summary.

    Args:
        message_ids: List of message IDs to fold (must be consecutive).
        summary: Concise summary of the folded messages.

    Returns:
        Status message with token savings information.
    """
    from peteos.chatbot import FoldedMessage

    sess = _state.session
    if not sess:
        return "Error: session not configured."

    unanchored = sess.chat_history._unanchored

    # Find the first matching message
    first_idx = None
    for i, msg in enumerate(unanchored):
        if msg.get_id() == message_ids[0]:
            first_idx = i
            break

    if first_idx is None:
        return f"Error: Could not find message with ID {message_ids[0]!r} in unanchored history."

    # Verify consecutive match
    matched = []
    for j, expected_id in enumerate(message_ids):
        target_idx = first_idx + j
        if target_idx >= len(unanchored):
            return (
                f"Error: Message at expected position {target_idx} does not exist. "
                "Messages must be consecutive."
            )
        actual_id = unanchored[target_idx].get_id()
        if actual_id != expected_id:
            return (
                f"Error: Messages must be consecutive. Expected ID {expected_id!r} "
                f"at position {first_idx + j} but found {actual_id!r}."
            )
        matched.append(unanchored[target_idx])

    # Calculate original token count
    original_token_count = sum(msg.count_tokens() for msg in matched)

    # Remove matched messages from unanchored
    del unanchored[first_idx:first_idx + len(matched)]

    # Create and insert FoldedMessage
    folded = FoldedMessage(
        summary=summary,
        original_messages=matched,
        message_id=str(uuid.uuid4()),
    )
    unanchored.insert(first_idx, folded)

    # Calculate folded token count
    folded_token_count = folded.count_tokens()
    savings = original_token_count - folded_token_count

    return (
        f"Folded {len(matched)} messages into a summary. "
        f"Token savings: {savings} (original: {original_token_count}, "
        f"folded: {folded_token_count})."
    )


def unfold(message_id: str) -> str:
    """Unfold a FoldedMessage, restoring the original messages.

    Finds the FoldedMessage containing the specified message ID in the
    session's unanchored messages, removes it, and inserts all original
    messages at that position.

    Args:
        message_id: ID of a message that is inside a FoldedMessage.

    Returns:
        Status message with number of restored messages.
    """
    from peteos.chatbot import FoldedMessage

    sess = _state.session
    if not sess:
        return "Error: session not configured."

    unanchored = sess.chat_history._unanchored

    # Find the folded message containing the target ID
    folded_idx = None
    folded_msg = None
    for i, msg in enumerate(unanchored):
        if isinstance(msg, FoldedMessage):
            for orig in msg._original_messages:
                if orig.get_id() == message_id:
                    folded_idx = i
                    folded_msg = msg
                    break
            if folded_msg:
                break

    if folded_msg is None:
        return f"Error: No FoldedMessage contains a message with ID {message_id!r}."

    # Remove the folded message
    del unanchored[folded_idx]

    # Insert original messages in order
    for orig_msg in folded_msg._original_messages:
        unanchored.insert(folded_idx, orig_msg)
        folded_idx += 1

    return f"Unfolded: restored {len(folded_msg._original_messages)} messages."


def set_approval_result(
    approved: str = "",
    reason: str = "",
    session: "Session | None" = None,
) -> str:
    """Set the review decision. Call with approved='yes' or 'no' and a reason.

    Writes the decision to the reviewer session's AgenticState so the
    invoking agent can read it out.

    Args:
        approved: 'yes' to approve the pattern, 'no' to deny it.
        reason: Detailed explanation of the decision.
        session: The reviewer session (injected by the execution environment).

    Returns:
        Confirmation string.
    """
    if session is None:
        return "Error: session not available."

    if approved not in ("yes", "no"):
        return "Error: approved must be 'yes' or 'no'."
    if not reason:
        return "Error: reason must be non-empty."

    try:
        session.state.create("approved", approved)
        session.state.create("reason", reason)
    except ValueError as e:
        return f"Error: {e}"

    return f"Approval result set: {approved}"


def register_approval_tools(tool_manager: ToolManager) -> None:
    """Register set_approval_result tool on the given tool manager.

    Args:
        tool_manager: The ToolManager to register the tool on.
    """
    tool_manager.register_tool(func=set_approval_result)


def register_fold_tools(tool_manager: ToolManager) -> None:
    """Register fold/unfold tools on the given tool manager.

    Args:
        tool_manager: The ToolManager to register the tools on.
    """
    tool_manager.register_tool(func=fold)
    tool_manager.register_tool(func=unfold)
