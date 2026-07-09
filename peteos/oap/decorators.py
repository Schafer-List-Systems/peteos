"""OAP decorators: @tool and @agentic_object."""

from __future__ import annotations

from typing import Callable


def tool(
    name: str | None | Callable = None,
    description: str | None = None,
) -> Callable[[Callable], Callable] | Callable:
    """Mark a method on an AgenticObject subclass as callable by agents.

    Supports both @tool and @tool(name="custom_name").
    """
    def apply(func: Callable) -> Callable:
        func._tool_name = name if isinstance(name, str) else func.__name__
        func._tool_description = description or (func.__doc__ or "").strip()
        return func

    # Handle @tool (no parentheses) — func passed as first positional arg
    if callable(name):
        return apply(name)
    return apply


def sandbox(
    name: str | None | Callable = None,
    description: str | None = None,
) -> Callable[[Callable], Callable] | Callable:
    """Mark a method on an AgenticObject subclass as callable from sandbox code.

    Supports both @sandbox and @sandbox(name="custom_name").
    A method can be both @tool (agent-loop) and @sandbox (sandbox-self).
    """
    def apply(func: Callable) -> Callable:
        func._sandbox_name = name if isinstance(name, str) else func.__name__
        func._sandbox_description = description or (func.__doc__ or "").strip()
        return func

    # Handle @sandbox (no parentheses) — func passed as first positional arg
    if callable(name):
        return apply(name)
    return apply


def agentic_object(
    imports: list[object] | None = None,
    import_aliases: dict[str, str] | None = None,
    invoke_sub_agents: bool = False,
    allow_code_execution: bool = False,
    persist_functions: bool = False,
    role: str | None = None,
) -> Callable[[type], type]:
    """Configure agent capabilities per AgenticObject subclass."""

    def decorator(cls: type) -> type:
        cls._oap_config = {
            "imports": list(imports) if imports else [],
            "import_aliases": import_aliases or {},
            "invoke_sub_agents": invoke_sub_agents,
            "allow_code_execution": allow_code_execution,
            "persist_functions": persist_functions,
            "role": role,
        }
        return cls

    return decorator
