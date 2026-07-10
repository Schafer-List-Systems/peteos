"""Restricted exec sandbox for OAP code execution."""

from __future__ import annotations

from typing import Any, Callable

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


class SandboxSelf:
    """Per-invocation wrapper passed as `self` into sandbox code.

    An empty object that gets populated with proxy methods per
    invocation.  The proxy lambdas capture the real self and runner in
    their closure scope, so the sandboxed code has no way to reach the
    underlying AgenticObject or runner via introspection.

    The ``invoke`` method is attached conditionally by ``_python_exec``
    when sub-agent invocation is enabled on the parent AgenticObject.
    """
    pass

    @staticmethod
    def populate(sandbox_self: "SandboxSelf", real_self: object) -> None:
        """Attach proxied @tool and @sandbox methods to an empty SandboxSelf.

        Walks the MRO of *real_self*'s class, finds methods decorated
        with ``@tool`` or ``@sandbox``, and attaches them as proxy
        attributes using the ``_sandbox_name`` (falling back to the
        Python method name).

        Args:
            sandbox_self: The empty SandboxSelf instance to populate.
            real_self: The AgenticObject instance whose methods to proxy.
        """
        registered: set[str] = set()
        for cls in real_self.__class__.__mro__:
            for method_name, method in cls.__dict__.items():
                if not callable(method):
                    continue
                if method_name in registered:
                    continue
                if hasattr(method, "_tool_name"):
                    sandbox_name = method._tool_name
                elif hasattr(method, "_sandbox_name"):
                    sandbox_name = method._sandbox_name
                else:
                    continue
                registered.add(method_name)
                proxy_fn = getattr(real_self, method_name)
                setattr(sandbox_self, sandbox_name, proxy_fn)


def build_sandbox_description(imports: list[object] | None = None) -> str:
    """Build the description string for the python_exec tool.

    Args:
        imports: List of modules available in the sandbox.

    Returns:
        Description string listing available modules if any.
    """
    prompt = (
        "The value must be a Python function with the exact signature "
        "`func(self)` where `self` is a wrapper object for the agentic "
        "instance. "
        "Define exactly one function named `func` — the harness will find "
        "and execute it. The function must return the final result (not print it). "
        "Use `self.produce_output(result)` or `self.produce_error(message)` to return data or signal failure. "
        "Forbidden: __builtins__, __import__, network, filesystem."
    )
    if imports:
        mods_list = ", ".join(
            m.__name__ if hasattr(m, "__name__") else str(m) for m in imports
        )
        prompt += f" Imported modules: {mods_list}."
    return prompt


def create_sandbox_globals(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Create a restricted globals dict for exec() in the sandbox.

    Args:
        config: MRO-merged OAP config dict with 'imports' and 'import_aliases' keys.

    Returns:
        Globals dict ready for exec().
    """
    registry: dict[str, Any] = {}
    imports = config.get("imports", [])
    import_aliases = config.get("import_aliases", {})
    globals_dict: dict[str, Any] = {
        "__builtins__": {**_SAFE_BUILTINS, "__import__": lambda name, *_a, **_kw: _restricted_import(name, registry)},
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


def sandbox_compile(
    config: dict[str, Any],
    code: str,
) -> "tuple[dict[str, Any], list[Any]] | str":
    """Exec *code* in a sandbox and return the globals plus newly defined callables.

    This is a pure compilation step — it does not filter by signature or
    invoke the function. Callers decide which of the returned callables to
    use and how to invoke them.

    Args:
        config: MRO-merged OAP config dict.
        code: Python code to exec.

    Returns:
        A (globals, new_callables) tuple on success, or an error string.
    """
    # Validate syntax up front before sandboxing.
    try:
        compile(code, "<persisted>", "exec")
    except SyntaxError as e:
        return f"Error: {type(e).__name__}: {e}"

    sandbox_globals = create_sandbox_globals(config)
    original_keys = set(sandbox_globals.keys())

    try:
        exec(code, sandbox_globals)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    new_callables: list[Any] = []
    for k in set(sandbox_globals.keys()) - original_keys:
        obj = sandbox_globals[k]
        if callable(obj) and hasattr(obj, "__code__"):
            new_callables.append(obj)

    return sandbox_globals, new_callables


def filter_callables_by_args(
    callables: list[Callable],
    *args: Any,
    **kwargs: Any,
) -> list[Callable]:
    """Filter *callables* to those compatible with the given arguments.

    Returns functions with ``argcount == n`` (where *n* is the total number
    of positional + keyword arguments), plus functions with ``argcount == n``
    ``+ 1`` where the sole extra parameter is named ``"self"`` (matched for
    SandboxSelf dispatch).

    Args:
        callables: Callables from ``sandbox_compile``.
        *args: Positional arguments that will be passed to the function.
        **kwargs: Keyword arguments that will be passed to the function.

    Returns:
        Matching callables.
    """
    n = len(args) + len(kwargs)
    matching: list[Callable] = []
    for func in callables:
        try:
            argcount = func.__code__.co_argcount
        except AttributeError:
            continue
        if argcount == n:
            matching.append(func)
        elif argcount == n + 1 and func.__code__.co_varnames[0] == "self":
            matching.append(func)
    return matching


def _compile_and_select(
    config: dict[str, Any],
    code: str,
    *args: Any,
    **kwargs: Any,
) -> "Callable | str":
    """Compile *code* in a sandbox, filter by *args/kwargs, and return the matching function.

    This is the common pipeline of sandbox_compile → filter_callables_by_args →
    single-match validation.  Returns the single matching function object, or
    an error string.

    Args:
        config: MRO-merged OAP config dict.
        code: Python code to exec.
        *args: Positional arguments used to filter matching functions.
        **kwargs: Keyword arguments used to filter matching functions.

    Returns:
        The matching function object on success, or an error string.
    """
    result = sandbox_compile(config, code)
    if isinstance(result, str):
        return result
    sandbox_globals, new_callables = result

    matching = filter_callables_by_args(new_callables, *args, **kwargs)
    if len(matching) != 1:
        raise ValueError(
            f"expected exactly one new function. Found: {[m.__name__ for m in matching]}."
        )

    return matching[0]


