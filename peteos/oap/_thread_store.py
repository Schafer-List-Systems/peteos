"""Thread-to-session mapping for persistent OAP threads."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peteos.agent import Agent
    from peteos.session import Session

logger = logging.getLogger(__name__)


class ThreadStore:
    """Maps thread_id to (session_uuid, agent) for persistent threads.

    Since AgenticObjectBase instances are user objects (not peteos objects),
    persistent thread state lives here as a class-level registry mapping
    thread_id to the underlying peteos Session and Agent.
    """

    _registry: dict[str, dict[str, Any]] = {}

    @classmethod
    def get_session(cls, thread_id: str) -> Session | None:
        """Get the Session for a thread_id, or None if not found."""
        entry = cls._registry.get(thread_id)
        if entry is None:
            return None
        return entry.get("session")  # type: ignore[return-value]

    @classmethod
    def get_agent(cls, thread_id: str) -> Any:
        """Get the Agent for a thread_id."""
        entry = cls._registry.get(thread_id)
        if entry is None:
            return None
        return entry.get("agent")

    @classmethod
    def set_session(cls, thread_id: str, session: Session, agent: Any) -> None:
        """Register a session for a thread_id."""
        cls._registry[thread_id] = {"session": session, "agent": agent}

    @classmethod
    def destroy(cls, thread_id: str) -> None:
        """Remove a thread from the registry. Caller must stop the session."""
        cls._registry.pop(thread_id, None)

    @classmethod
    def list_ids(cls) -> list[str]:
        """List all registered thread_ids."""
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Destroy all threads and stop all sessions."""
        for thread_id, entry in list(cls._registry.items()):
            session = entry.get("session")
            if session is not None and hasattr(session, "is_running") and session.is_running():
                session.stop()
        cls._registry.clear()
