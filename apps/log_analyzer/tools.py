"""Log Analyzer tools — stub implementations for wake/sleep state and filtering."""


class LogState:
    """Shared mutable state for log analyzer tools."""

    awake: bool = False
    _filters: list[str] = []


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


def filter(pattern: str) -> str:
    """Add a regex filter pattern to filter out unimportant log messages.

    Args:
        pattern: Regular expression pattern to match messages.

    Returns:
        A confirmation message.
    """
    return "OK"


def list_filter() -> str:
    """List all currently registered filter patterns.

    Returns:
        A string listing all active filter patterns.
    """
    return "OK"
