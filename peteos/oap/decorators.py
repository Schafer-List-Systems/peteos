"""OAP decorators: @tool and @agentic_object."""

from __future__ import annotations

from typing import Callable


def tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable], Callable]:
    """Mark a method on an AgenticObjectBase subclass as callable by agents."""

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
    """Configure agent capabilities per AgenticObjectBase subclass."""

    def decorator(cls: type) -> type:
        cls._oap_config = {
            "imports": list(imports) if imports else [],
            "invoke_sub_agents": invoke_sub_agents,
            "allow_code_execution": allow_code_execution,
        }
        return cls

    return decorator
