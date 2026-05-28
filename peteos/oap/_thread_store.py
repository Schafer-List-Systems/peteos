"""Thread-to-session mapping for persistent OAP threads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peteos.agent import Agent
    from peteos.session import Session


class ThreadStore:
    """Maps thread_id to (session_uuid, agent) for persistent threads."""

    _registry: dict[str, dict[str, Any]] = {}

    @classmethod
    def get_session(cls, thread_id: str) -> "Session | None":
        """Get the Session for a thread_id, or None if not found."""
        return None

    @classmethod
    def get_agent(cls, thread_id: str) -> "Agent | None":
        """Get the Agent for a thread_id."""
        return None

    @classmethod
    def set_session(cls, thread_id: str, session: "Session", agent: "Agent") -> None:
        """Register a session for a thread_id."""
        pass

    @classmethod
    def destroy(cls, thread_id: str) -> None:
        """Remove a thread from the registry."""
        cls._registry.pop(thread_id, None)
