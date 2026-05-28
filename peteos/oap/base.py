"""AgenticObjectBase - base class for all OAP objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peteos.agent import Agent


class AgenticObjectBase:
    """Base class for all Object-Agentic Programming objects."""

    def __init__(self) -> None:
        self._oap_agent: Agent | None = None
        self._oap_role: Any = None

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

    def invoke(
        self,
        target: "AgenticObjectBase",
        prompt: str,
        output_schema: type | None = None,
        persistent: bool = False,
    ) -> Any:
        """Invoke a sub-agent on a target AgenticObjectBase."""
        from peteos.oap.engine import invoke as _invoke

        return _invoke(
            target,
            prompt=prompt,
            output_schema=output_schema,
            persistent=persistent,
        )
