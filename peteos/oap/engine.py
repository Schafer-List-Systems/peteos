"""OAP invoke() — thin wrapper around peteos infrastructure."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peteos.oap.base import AgenticObjectBase


def invoke(
    oap_object: "AgenticObjectBase",
    prompt: str = "",
    output_schema: type | None = None,
    thread_id: str | None = None,
    agent: Any | None = None,
    persistent: bool = False,
) -> Any:
    """Invoke an OAP agent and return structured output."""
    raise NotImplementedError("OAP engine not yet implemented")
