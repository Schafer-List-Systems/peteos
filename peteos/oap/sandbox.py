"""Restricted exec sandbox for OAP code execution."""

from __future__ import annotations

from typing import Any


def create_sandbox_globals(
    self_obj: Any,
    imports: list[object] | None = None,
) -> dict[str, Any]:
    """Create a restricted globals dict for exec() in the sandbox."""
    return {"__builtins__": {}, "self": self_obj}
