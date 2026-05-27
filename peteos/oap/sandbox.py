"""Restricted exec sandbox for OAP code execution."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_sandbox_globals(
    self_obj: Any,
    imports: list[object] | None = None,
) -> dict[str, Any]:
    """Create a restricted globals dict for exec() in the sandbox.

    Security properties:
    - __builtins__ is an empty dict - no built-in functions
    - self is the only object reference available to the agent's code
    - Imported modules are passed as named globals
    - No __import__, no importlib, no network, no filesystem

    Args:
        self_obj: The AgenticObjectBase instance (injected as "self").
        imports: List of modules to inject as named globals.

    Returns:
        A restricted globals dictionary.
    """
    globals_dict: dict[str, Any] = {
        "__builtins__": {},
        "self": self_obj,
    }

    if imports:
        for mod in imports:
            globals_dict[mod.__name__] = mod

    return globals_dict
