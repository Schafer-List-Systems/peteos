"""OAP decorators: @tool and @agentic_object."""

from __future__ import annotations

from typing import Callable


def tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable], Callable]:
    """Mark a method on an AgenticObjectBase subclass as callable by agents.

    Parameters and return types are derived from the method signature.
    Only name and description can be overridden.
    Undecorated methods are invisible to agents.
    """

    def decorator(func: Callable) -> Callable:
        func._tool_name = name or func.__name__
        func._tool_description = description or (func.__doc__ or "").strip()
        return func

    return decorator


def agentic_object(
    imports: list[object] | None = None,
    invoke_sub_agents: bool = False,
    allow_code_execution: bool = False,
) -> Callable[[type], type]:
    """Configure agent capabilities per AgenticObjectBase subclass.

    All three parameters default to disabled (opt-in).

    imports:
        Modules injected into the sandbox as real Python object references.
        Stored in a private registry keyed by class; not discoverable via
        class attributes. Inheritance does not merge imports.

    invoke_sub_agents:
        Enables invoke() for sub-agent calls on this object.
        The gatekeeper checks this flag on the target object, not the caller.

    allow_code_execution:
        Allows the agent to write and execute sandboxed Python.
        The sandbox provides exactly one global variable: self.
        No __builtins__, no __import__, no network, no filesystem.
    """

    def decorator(cls: type) -> type:
        cls._oap_config = {
            "imports": list(imports) if imports else [],
            "invoke_sub_agents": invoke_sub_agents,
            "allow_code_execution": allow_code_execution,
        }
        return cls

    return decorator
