"""Restricted exec sandbox for OAP code execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peteos.oap.base import AgenticObjectBase

# Safe builtins: pure functions for data manipulation and output.
# No file I/O, no network, no type() metaclass tricks, no exec/eval.
_SAFE_BUILTINS: dict[str, Any] = {
    # Data creation / conversion
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    # Iteration
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    # Math
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "round": round,
    "divmod": divmod,
    # Logic
    "any": any,
    "all": all,
    # Output
    "print": print,
    # Type helpers
    "isinstance": isinstance,
    "type": type,
}


def create_sandbox_globals(
    self_obj: AgenticObjectBase,
    imports: list[object] | None = None,
    import_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Create a restricted globals dict for exec() in the sandbox."""
    # Per-call registry so each sandbox has its own isolated import whitelist.
    registry: dict[str, Any] = {}
    globals_dict: dict[str, Any] = {
        "__builtins__": {**_SAFE_BUILTINS, "__import__": lambda name, *_a, **_kw: _restricted_import(name, registry)},
        "this": self_obj,
    }
    if imports:
        for mod in imports:
            registry[mod.__name__] = mod
            globals_dict[mod.__name__] = mod
    if import_aliases:
        for mod_name, alias in import_aliases.items():
            obj = registry.get(mod_name)
            if obj:
                globals_dict[alias] = obj
    return globals_dict


def _restricted_import(name: str, registry: dict[str, Any]) -> Any:
    """Resolve import from the per-call registry only."""
    resolved = registry.get(name)
    if resolved is None:
        raise ImportError(f"No module named {name!r}")
    return resolved