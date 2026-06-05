"""Utility functions for the agentic process module."""

from datetime import datetime


def format_timestamp() -> str:
    """Return the current local time as [YYYY-MM-DD HH:MM:SS]."""
    now = datetime.now()
    return f"[{now.strftime('%d.%m.%Y %H:%M:%S')}]"