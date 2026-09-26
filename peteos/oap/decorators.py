"""OAP decorators: @tool and @agentic_object."""

from __future__ import annotations

import types
from typing import Callable


def _make_dummy_closure(n: int) -> tuple:
    """Create n dummy cell objects for FunctionType closure injection."""
    result = []
    for _ in range(n):
        _sentinel = object()
        fn = (lambda: _sentinel)
        result.append(fn.__closure__[0])
    return tuple(result)


def _find_tool_policy(func: Callable) -> tuple[Callable, tuple[str, ...]] | None:
    """Find a nested tool_policy function inside func and return (policy_fn, freevar_names).

    Inspects func's code object constants to find the code object by name,
    then wraps it in a FunctionType with func's globals and a dummy closure
    so it is callable. Returns None if no tool_policy is defined.
    """
    code = func.__code__
    for const in code.co_consts:
        if isinstance(const, type(code)) and const.co_name == "tool_policy":
            freevar_names = const.co_freevars
            closure = _make_dummy_closure(len(freevar_names))
            policy_fn = types.FunctionType(const, func.__globals__, const.co_name, (), closure)
            return (policy_fn, freevar_names)
    return None


def tool(
    name: str | None | Callable = None,
    description: str | None = None,
) -> Callable[[Callable], Callable] | Callable:
    """Mark a method on an AgenticObject subclass as callable by agents.

    Supports both @tool and @tool(name="custom_name").
    Optionally contains a nested tool_policy() function for per-call argument-based
    approval decisions. The policy has no parameters and reads the enclosing method's
    local variables directly. Returns True/False/None (or ApprovalDecision values).
    """
    def apply(func: Callable) -> Callable:
        policy_info = _find_tool_policy(func)
        func._tool_name = name if isinstance(name, str) else func.__name__
        func._tool_description = description or (func.__doc__ or "").strip()
        if policy_info:
            func._tool_policy, func._tool_policy_freevars = policy_info
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
    define_functions: bool = False,
    role: str | None = None,
) -> Callable[[type], type]:
    """Configure agent capabilities per AgenticObject subclass."""

    def decorator(cls: type) -> type:
        cls._oap_config = {
            "imports": list(imports) if imports else [],
            "import_aliases": import_aliases or {},
            "invoke_sub_agents": invoke_sub_agents,
            "allow_code_execution": allow_code_execution,
            "define_functions": define_functions,
            "role": role,
        }
        return cls

    return decorator
