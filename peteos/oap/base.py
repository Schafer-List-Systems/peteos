"""AgenticObjectBase - base class for all OAP objects."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeVar

from peteos.oap.error import Error

if TYPE_CHECKING:
    from peteos.agent import Agent

T = TypeVar("T", bound="AgenticObjectBase")

logger = logging.getLogger(__name__)


class AgenticObjectBase:
    """Base class for all Object-Agentic Programming objects.

    User objects inherit from this class to expose methods as agent-callable
    tools via the @tool decorator. The class itself must be decorated with
    @agentic_object() to configure agent capabilities.

    Attributes:
        _oap_agent: Optional Agent instance for LLM invocations.
        _oap_role: Optional custom peteos Role to use for invocations.
                   If None, a minimal Role is created from scratch.
        _oap_threads: Thread-local storage for per-object conversation state.
    """

    def __init__(self) -> None:
        self._oap_agent: Agent | None = None
        self._oap_role: Any = None  # peteos Role or None
        self._oap_threads: dict[str, dict[str, Any]] = {}

    @property
    def agent(self) -> Agent | None:
        """The Agent instance used for LLM invocations."""
        return self._oap_agent

    @agent.setter
    def agent(self, value: Agent | None) -> None:
        self._oap_agent = value

    @property
    def role(self) -> Any | None:
        """The peteos Role used for invocations, or None for auto-generated."""
        return self._oap_role

    @role.setter
    def role(self, value: Any | None) -> None:
        self._oap_role = value

    def get_thread_state(self, thread_id: str) -> dict[str, Any]:
        """Get or create thread-local state for this object."""
        if thread_id not in self._oap_threads:
            self._oap_threads[thread_id] = {"messages": []}
        return self._oap_threads[thread_id]

    def invoke(
        self,
        target: AgenticObjectBase,
        prompt: str,
        output_schema: type | None = None,
        persistent: bool = False,
    ) -> Any:
        """Invoke a sub-agent on a target AgenticObjectBase.

        Prerequisites: the target must have @agentic_object(invoke_sub_agents=True).
        Calling code must be sandboxed (allow_code_execution=True on the caller).

        Args:
            target: The AgenticObjectBase instance to invoke.
            prompt: The task prompt for the target agent.
            output_schema: Optional type for structured output.
            persistent: If True, inherit parent's thread_id for persistence.

        Returns:
            The result from the target agent, or Error if the gatekeeper blocks.

        Raises:
            ValueError: If the target does not have invoke_sub_agents enabled.
        """
        # Gatekeeper: check target's invoke_sub_agents flag
        target_config = getattr(target.__class__, "_oap_config", {})
        if not target_config.get("invoke_sub_agents", False):
            return Error("Sub-agent invocation not enabled on target")

        # Determine thread_id: persistent inherits caller's thread context,
        # non-persistent uses a fresh thread
        thread_id: str | None = None
        if persistent:
            # Look for an active thread on the caller
            active_threads = [
                tid for tid, state in self._oap_threads.items() if state.get("active", False)
            ]
            if active_threads:
                thread_id = active_threads[0]

        # Import here to avoid circular import at module load time
        from peteos.oap.engine import invoke as _invoke

        return _invoke(
            target,
            prompt=prompt,
            output_schema=output_schema,
            thread_id=thread_id,
            agent=self._oap_agent,
        )
