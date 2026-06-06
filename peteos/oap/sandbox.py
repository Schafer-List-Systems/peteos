"""Restricted exec sandbox for OAP code execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peteos.oap.base import AgenticObjectBase


def create_sandbox_globals(
    self_obj: AgenticObjectBase,
    imports: list[object] | None = None,
) -> dict[str, Any]:
    """Create a restricted globals dict for exec() in the sandbox."""
    globals_dict: dict[str, Any] = {"__builtins__": {}, "self": self_obj}
    if imports:
        for mod in imports:
            globals_dict[mod.__name__] = mod
    return globals_dict
