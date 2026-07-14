"""Restricted exec sandbox for OAP code execution."""

from __future__ import annotations

import textwrap
from typing import Any, Callable


################################################################################
# Source Code Compilation

def _extract_func_name_and_params(func_obj: Callable) -> tuple[str, dict, bool]:
    """Extract function name, parameters, and whether self is unbound.

    Args:
        func_obj: A Python callable to introspect.

    Returns:
        A tuple of (func_name, parameters_schema, has_unbound_self).
        ``has_unbound_self`` is True only when the callable has a self
        argument that has not yet been bound (i.e. it is not a bound method).

    Raises:
        ValueError: If the signature cannot be introspected.
    """
    import inspect

    func_name = func_obj.__name__
    try:
        sig = inspect.signature(func_obj)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"could not introspect function signature: {e}"
        ) from e

    # Determine whether self needs binding.
    # Bound methods already have self consumed, so we only need to bind
    # when the callable is not a bound method and its first parameter
    # is named "self".
    has_unbound_self = False
    if not inspect.ismethod(func_obj):
        params = list(sig.parameters.items())
        has_unbound_self = params and params[0][0] == "self"

    # Build parameter schema, skipping the self parameter.
    parameters = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        param_info: dict = {}
        if param.annotation != inspect.Parameter.empty:
            name = param.annotation.__name__ if hasattr(param.annotation, "__name__") else str(param.annotation)
            param_info["type"] = name
        else:
            param_info["type"] = "any"
        if param.default == inspect.Parameter.empty:
            param_info["required"] = True
        else:
            param_info["required"] = False
            param_info["default"] = param.default
        parameters[param_name] = param_info

    return func_name, parameters, has_unbound_self


def _compile_and_extract(
    code: str, globals_dict: dict[str, Any]
) -> tuple[list[Callable], list[Callable]]:
    """Execute *code* in *globals_dict* and return member and global functions.

    Args:
        code: Python source code to compile.
        globals_dict: The globals namespace to exec into.

    Returns:
        A tuple of (member_functions, global_functions) where member
        functions have ``self`` as the first parameter and global
        functions do not.
    """
    # Normalize indentation so agent-provided code works regardless of level.
    code = textwrap.dedent(code)

    # Validate syntax up front — avoid partial exec with broken code.
    try:
        compile(code, "<defined>", "exec")
    except SyntaxError as e:
        raise ValueError(f"SyntaxError: {e}")

    # Record which keys existed before exec, so we only find NEW definitions.
    original_keys = set(globals_dict.keys())

    # Execute the code into the namespace. Any exception means the code failed.
    try:
        exec(code, globals_dict)
    except Exception as e:
        raise ValueError(f"{type(e).__name__}: {e}")

    # Scan for new callables and classify them by whether they need self-binding.
    member_functions: list[Callable] = []
    global_functions: list[Callable] = []
    for key in set(globals_dict.keys()) - original_keys:
        obj = globals_dict[key]
        if callable(obj) and hasattr(obj, "__code__"):
            _, _, has_unbound_self = _extract_func_name_and_params(obj)
            if has_unbound_self:
                member_functions.append(obj)
            else:
                global_functions.append(obj)
    return member_functions, global_functions


################################################################################
# Sandbox Global Scope

def _restricted_import(name: str, registry: dict[str, Any]) -> Any:
    """Resolve import from the per-call registry only."""
    resolved = registry.get(name)
    if resolved is None:
        raise ImportError(f"No module named {name!r}")
    return resolved


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


################################################################################
# Sandbox

import contextvars


class Scope:
    """A hierarchical namespace with a parent reference."""

    def __init__(self, name: str, parent: Scope | None = None) -> None:
        self.name = name
        self.parent = parent
        self.entries: dict[str, object] = {}

    def __getattr__(self, name: str) -> object:
        if name in self.entries:
            return self.entries[name]
        if self.parent is not None:
            return getattr(self.parent, name)
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")


class Sandbox:
    """A sandbox object where attribute resolution follows the caller's namespace."""

    _calling_ns: contextvars.ContextVar[Scope | None] = contextvars.ContextVar(
        "_calling_ns", default=None,
    )

    def __init__(self, scope: Scope) -> None:
        self._scope = scope

    def __getattr__(self, name: str) -> object:
        """Resolve attribute via the caller's scope, then walk up the parent chain."""
        caller_ns = Sandbox._calling_ns.get()
        ns = caller_ns if caller_ns is not None else self._scope
        obj = getattr(ns, name)
        if callable(obj):
            return lambda *args, **kwargs: obj(self, *args, **kwargs)
        return obj

    def __repr__(self) -> str:
        names: list[str] = []
        ns: Scope | None = self._scope
        while ns is not None:
            names.append(ns.name)
            ns = ns.parent
        return f"{self.__class__.__name__}(namespaces={names})"


class SandboxBuilder:
    # Why the sandbox is safe regarding sibling functions with real_self:
    #
    #   1. Agent's import is restricted — _restricted_import only resolves
    #      modules from the per-call registry.  Importing the wrapper's
    #      module always raises ImportError.
    #   2. No path to sys.modules — sys is absent from _SAFE_BUILTINS and
    #      the lambda's __globals__ does not import sys.
    #   3. No __closure__ cells — the wrapper has zero captured variables;
    #      real_self lives inside the wrapper's globals dict, not in closure.
    #   4. Wrapper's __globals__ is cleaned after each call — the finally
    #      block removes the registry reference from the function's globals.
    #   5. Wrapper's __globals__ is clean before each call — the finally
    #      ensures no residual references persist across invocations.

    def __init__(self, name: str, base: "SandboxBuilder | None" = None) -> None:
        self._name: str = name
        self._compilation_globals: dict[str, Any] = {}
        self._sandbox_namespace: dict[str, Callable] = {}  # sandbox_namespace: name → function for this level
        self._base: SandboxBuilder | None = base  # parent builder in the chain (inner → outer)
        self._attached_sandbox: Sandbox | None = None
        self._proxies: list[tuple[str, Callable, Any | None]] = []
        self._source_code: dict[int, tuple[str, bool]] = {}
        self._name_to_entry: dict[str, int] = {}  # maps function name -> source code id
        self._source_code_counter: int = 0

    @property
    def sandbox_namespace(self) -> dict[str, Callable]:
        """Sandbox namespace dict for this level: name → function."""
        return self._sandbox_namespace

    def add_safe_builtins(self, overwrite: bool = True) -> None:
        """Add the safe builtins to the sandbox's global namespace.

        See ``_SAFE_BUILTINS`` for the default safe set. Also injects a
        restricted ``__import__`` that only resolves names from the
        builder's configured imports.

        Args:
            overwrite: If ``False``, raise ``ValueError`` for any name
                       collision with the safe builtin set. Defaults to
                       ``True``.
        """
        for name, func in _SAFE_BUILTINS.items():
            if not overwrite and name in self._compilation_globals:
                raise ValueError(f"name {name!r} already exists in the sandbox namespace")
        self._compilation_globals.update(_SAFE_BUILTINS)
        if not overwrite and "__import__" in self._compilation_globals:
            raise ValueError(f"name '__import__' already exists in the sandbox namespace")
        self._compilation_globals["__import__"] = lambda name, *_a, **_kw: _restricted_import(
            name, self._compilation_globals
        )

    def add_caller_builtins(self, overwrite: bool = True) -> None:
        """Mirror the caller's builtins into the sandbox's global namespace.

        Captures a snapshot of ``vars(builtins)`` at call time.

        Args:
            overwrite: If ``False``, raise ``ValueError`` for any name
                       collision with the caller's builtins. Defaults to
                       ``True``.
        """
        import builtins
        if not overwrite:
            for name in builtins.__dict__:
                if name in self._compilation_globals:
                    raise ValueError(f"name {name!r} already exists in the sandbox namespace")
        self._compilation_globals.update(builtins.__dict__)

    def add_builtin(self, name: str, func: Callable, overwrite: bool = True) -> None:
        """Manually add a builtin to the sandbox's global namespace.

        Args:
            name: The name of the builtin.
            func: The builtin callable.
            overwrite: If ``False``, raise ``ValueError`` when a name
                       collision exists. Defaults to ``True``.
        """
        if not overwrite and name in self._compilation_globals:
            raise ValueError(f"name {name!r} already exists in the sandbox namespace")
        self._compilation_globals[name] = func

    def build_sandbox_description(self) -> str:
        """Build the description string for the python_exec tool.

        Lists all registered module names from the builder's configured
        imports and globals.

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
        if self._compilation_globals:
            mods_list = ", ".join(
                name for name in self._compilation_globals
                if hasattr(self._compilation_globals[name], "__name__")
            )
            prompt += f" Imported modules: {mods_list}."
        return prompt

    def add_import(
        self,
        module: object,
        name: str | None = None,
        overwrite: bool = True,
    ) -> None:
        """Add a module to the sandbox's global namespace.

        Args:
            module: The module object to add.
            name: Optional name to register the module under.
                  If not provided, uses ``module.__name__``.
            overwrite: If ``False``, raise ``ValueError`` when a name
                       collision exists. Defaults to ``True``.
        """
        key = name or (module.__name__ if hasattr(module, "__name__") else str(module))
        if not overwrite and key in self._compilation_globals:
            raise ValueError(f"name {key!r} already exists in the sandbox namespace")
        self._compilation_globals[key] = module

    def get_import(self, name: str) -> object | None:
        """Look up a module by its name or alias.

        Returns the module object if found, or ``None`` if no import
        with that name exists.

        Args:
            name: The name or alias of the module to look up.

        Returns:
            The module object, or ``None`` if not found.
        """
        return self._compilation_globals.get(name)

    def add_imports(
        self,
        imports: list[object],
        aliases: dict[str, str] | None = None,
    ) -> None:
        """Add multiple modules to the sandbox's available imports.

        Args:
            imports: List of module objects to add.
            aliases: Optional mapping from alias string to import name.
        """
        for mod in imports:
            self.add_import(mod)
        if aliases:
            for alias, name in aliases.items():
                mod = self.get_import(name)
                self.add_import(mod, alias)

    def add_global_var(self, name: str, value: Any, overwrite: bool = True) -> None:
        """Add a variable to the sandbox's global namespace.

        Args:
            name: The name under which to expose the variable.
            value: The value to make available.
            overwrite: If ``False``, raise ``ValueError`` when a name
                       collision exists. Defaults to ``True``.
        """
        if not overwrite and name in self._compilation_globals:
            raise ValueError(f"name {name!r} already exists in the sandbox namespace")
        self._compilation_globals[name] = value

    def add_global_func(self, name: str, func: Callable, overwrite: bool = True) -> None:
        """Add a function to the sandbox's global namespace.

        Args:
            name: The name under which to expose the function.
            func: The callable to make available.
            overwrite: If ``False``, raise ``ValueError`` when a name
                       collision exists. Defaults to ``True``.
        """
        if not overwrite and name in self._compilation_globals:
            raise ValueError(f"name {name!r} already exists in the sandbox namespace")
        self._compilation_globals[name] = func

    def get_globals(self) -> dict[str, Any]:
        """Return a merged copy of compilation globals from the full builder chain.

        Walks up to the parent builder and starts with its globals, then updates
        with this builder's globals, so child namespaces override parents.
        """
        merged: dict[str, Any] = {}
        if self._base is not None:
            merged.update(self._base.get_globals())
        merged.update(self._compilation_globals)
        return merged

    def add_proxy(
        self,
        name: str,
        func: Callable,
        real_self: Any | None = None,
    ) -> None:
        """Attach a callable proxy to the sandbox under *name*.

        Binds *real_self* as the first argument if the function has an
        unbound self parameter, then registers the callable in the namespace.

        Args:
            name: The attribute name for the proxy on the sandbox object.
            func: The original callable to proxy.
            real_self: The owning object (used to look up the actual method).
        """
        _, _, has_unbound_self = _extract_func_name_and_params(func)
        
        def make_proxy():
            def proxy(self: Sandbox, *args, **kwargs):
                if has_unbound_self:
                    assert real_self is not None
                    return func(real_self, *args, **kwargs)
                else:
                    return func(*args, **kwargs)

            return proxy

        self._sandbox_namespace[name] = make_proxy()

        # legacy code:
        self._proxies.append((name, func, real_self))

    def add_proxies(
        self,
        proxies: dict[str, Callable],
        real_self: Any | None = None,
    ) -> None:
        """Batch-add callables to be proxied and attached to the Sandbox.

        Raises ``ValueError`` if any name in *proxies* conflicts with an
        already registered proxy name. Delegates to :meth:`add_proxy` for
        each callable.

        Args:
            proxies: Mapping of sandbox attribute names to callables.
            real_self: The owning object, passed to :meth:`add_proxy` for each.
        """
        existing = {name for name, *_ in self._proxies}
        collisions = set(proxies.keys()) & existing
        if collisions:
            raise ValueError(f"proxy name(s) {collisions!r} already registered")
        for name, func in proxies.items():
            self.add_proxy(name, func, real_self)

    def add_source_code(
        self,
        code: str,
        static_functions: bool = True,
    ) -> list[tuple[str, dict]]:
        """Store agent source code, compile immediately, and register functions in the sandbox namespace.

        Compiles in a merged globals namespace (inherited from all parent builders)
        to extract function names and parameter schemas, registers each function
        immediately in :attr:`sandbox_namespace`, registers each name in the internal
        name-to-entry map, and returns the list of (name, parameters) tuples.

        Args:
            code: Python source code defining sandboxed functions.
            static_functions: If ``True``, global functions in the code
                (no ``self`` parameter) will be added as static member
                functions on the sandbox instance.

        Returns:
            List of (func_name, parameters) tuples for each function that
            is registered on the sandbox.
        """
        key = self._source_code_counter
        self._source_code[key] = (code, static_functions)
        self._source_code_counter += 1

        # Compile in a copy to gather names and parameters.
        globals_ns = self.get_globals()
        member_functions, global_functions = _compile_and_extract(code, globals_ns)
        if static_functions:
            member_functions.extend(global_functions)

        ns_name = self._name

        def make_proxy(compiled_function, has_unbound_self: bool):
            def proxy(self: Sandbox, *args, **kwargs):
                # Find this builder's Scope in the chain and push it.
                caller_scope: Scope = self._scope
                while caller_scope is not None and caller_scope.name != ns_name:
                    caller_scope = caller_scope.parent
                if caller_scope is None:
                    raise RuntimeError(
                        f"Scope '{ns_name}' not found in sandbox chain — this should not happen"
                    )
                token = Sandbox._calling_ns.set(caller_scope)
                try:
                    return compiled_function(self, *args, **kwargs) if has_unbound_self else compiled_function(*args, **kwargs)
                finally:
                    Sandbox._calling_ns.reset(token)
            return proxy

        result: list[tuple[str, dict]] = []
        for mf in member_functions:
            func_name, params, has_unbound_self = _extract_func_name_and_params(mf)
            result.append((func_name, params))
            self._sandbox_namespace[func_name] = make_proxy(mf, has_unbound_self)
            self._name_to_entry[func_name] = key

        return result

    def remove_source_code(self, name: str) -> bool:
        """Remove the source code entry that defines *name*.

        Removes the entry from the source code dict and all associated
        names from the name-to-entry map. Returns ``True`` if found,
        ``False`` otherwise.

        Args:
            name: A function name that was registered by a source code entry.

        Returns:
            ``True`` if the source code entry was found and removed,
            ``False`` if no entry defines that name.
        """
        key = self._name_to_entry.pop(name, None)
        if key is None:
            return False
        del self._source_code[key]
        del self._sandbox_namespace[name]
        # Clean up all names that pointed to this key.
        to_remove = [n for n, k in self._name_to_entry.items() if k == key]
        for n in to_remove:
            del self._sandbox_namespace[n]
            del self._name_to_entry[n]
        return True

    def get_sandbox(self, freeze_namespaces: bool = True) -> Sandbox:
        """Return a sandbox snapshot from the builder chain.

        Traverses the builder chain (from self to outermost base), builds
        Scopes from each builder's :attr:`sandbox_namespace`, and returns
        a :class:`Sandbox` whose innermost Scope points to the session-level
        Scope and each Scope's :attr:`parent` points to the next outer one.

        When ``freeze_namespaces`` is True, each builder's namespace is
        shallow-copied into a fresh entries dict so the returned sandbox
        is independent — changes to the builder chain after this call do
        not affect the returned sandbox. When False, each Scope shares
        the builder's live namespace dict.

        Args:
            freeze_namespaces: If True, copy namespaces for snapshot isolation.

        Returns:
            A sandbox snapshot.
        """
        builders: list[SandboxBuilder] = []
        builder: SandboxBuilder | None = self
        while builder is not None:
            builders.append(builder)
            builder = builder._base

        # Build Scope chain: first builder = innermost, last = outermost root.
        prev: Scope | None = None
        for b in reversed(builders):
            ns_dict = dict(b._sandbox_namespace) if freeze_namespaces else b._sandbox_namespace
            scope = Scope(b._name, parent=prev)
            scope.entries = ns_dict
            prev = scope

        return Sandbox(prev)

    def compile(self, sandbox: Sandbox, code: str, globals_dict: dict[str, Any]) -> tuple[list[Callable], list[Callable]]:
        """Compile *code* in *globals_dict* and attach found member functions to this Sandbox.

        Returns the list of member functions (proxied and bound to the *sandbox*) and global functions found during compilation.

        Args:
            sandbox: The Sandbox instance to attach proxies to.
            code: Python source code to compile.
            globals_dict: Globals namespace to exec into.

        Returns:
            Tuple of (proxied member functions, global functions) defined in *code*.
        """
        member_functions, global_functions = _compile_and_extract(code, globals_dict)
        proxied_member_functions: list[Callable] = []
        for mf in member_functions:
            proxy = self.create_proxy(mf, sandbox)
            proxied_member_functions.append(proxy)
            setattr(sandbox, mf.__name__, proxy)
        return proxied_member_functions, global_functions

    def call(self, sandbox: Sandbox, code: str, globals_dict: dict[str, Any], *args, **kwargs) -> Any:
        """Compile *code*, find the matching member function, call it, and clean up.

        Compiles *code* in *globals_dict*, filters the resulting member functions
        by exact argument count (proxies already have *self* bound), invokes the
        single matching function with *args* and *kwargs*, then detaches the
        member functions for a clean state on the next call.

        Args:
            sandbox: The Sandbox instance to attach proxies to.
            code: Python source code to compile.
            globals_dict: Globals namespace to exec into.
            *args: Positional arguments passed to the matched function.
            **kwargs: Keyword arguments passed to the matched function.

        Returns:
            The return value of the matched function.

        Raises:
            ValueError: If the number of matching functions is not exactly one.
        """
        # compile the code and attach member functions (those with self as first argument)
        member_functions, global_functions = self.compile(sandbox, code, globals_dict)

        # find matching function by the number of arguments
        # TODO: once create_proxy carries a proper __signature__ / __code__, match on
        #       inspect.signature(proxy) instead of stored _effective_argcount.
        n = len(args) + len(kwargs)
        matching = [f for f in member_functions if getattr(f, "_effective_argcount", 0) == n]
        if not matching:
            matching = [f for f in global_functions if getattr(f.__code__, "co_argcount", 0) == n]
        if len(matching) != 1:
            raise ValueError(f"expected exactly one function, found {len(matching)}: {[f.__name__ for f in matching]}")

        # call matching function and get result
        result = matching[0](*args, **kwargs)

        # detach member functions for clean state on next call
        for mf in member_functions:
            delattr(sandbox, getattr(mf, "_original_name", mf.__name__))

        return result

    @staticmethod
    def create_proxy(func: Callable, real_self: Any | None = None) -> Any:
        """Create a sandbox proxy that normalizes any callable to a uniform signature.

        Works with any callable — regular functions, bound methods, lambdas,
        __call__ instances, partial objects, etc.  Uses functools.partial to
        bind *real_self* as the first argument, so every proxy has the same
        call signature (_proxy(self, *args, **kwargs)) regardless of whether
        the original callable required a self reference.

        Args:
            func: The original callable to proxy.
            real_self: The owning object (used to look up the actual method).

        Returns:
            A compiled proxy function ready for use in sandboxed code.
        """
        func_name, _parameters, has_unbound_self = _extract_func_name_and_params(func)

        if has_unbound_self:
            assert real_self is not None
            from functools import partial
            _callable = partial(func, real_self)
        else:
            _callable = func

        return _callable


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

        def _register(name: str, method: Callable) -> None:
            """Register a tool/sandbox method if not already registered."""
            if name in registered:
                return
            sandbox_name = getattr(method, "_tool_name", None) or getattr(method, "_sandbox_name", None)
            if sandbox_name is None:
                return
            registered.add(name)
            setattr(sandbox_self, sandbox_name, getattr(real_self, name))

        for cls in real_self.__class__.__mro__:
            for method_name, method in cls.__dict__.items():
                if callable(method):
                    _register(method_name, method)
        for method_name, method in real_self.__dict__.items():
            if callable(method):
                _register(method_name, method)


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


def sandbox_compile(
    config: dict[str, Any],
    code: str,
    sandbox_globals: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[Any]]:
    """Exec *code* in a sandbox and return the globals plus newly defined callables.

    This is a pure compilation step — it does not filter by signature or
    invoke the function. Callers decide which of the returned callables to
    use and how to invoke them.

    Args:
        config: MRO-merged OAP config dict.
        code: Python code to exec.
        sandbox_globals: Optional pre-existing globals dict to compile into.
            If provided, the code executes in this namespace (e.g. to share
            definitions across multiple compiles). If omitted, a fresh
            sandbox globals dict is created.

    Returns:
        A (globals, new_callables) tuple on success.

    Raises:
        ValueError: On syntax error or exec failure.
    """
    # Remove common leading whitespace so agent-provided code works
    # regardless of indentation level.
    code = textwrap.dedent(code)

    # Validate syntax up front before sandboxing.
    try:
        compile(code, "<defined>", "exec")
    except SyntaxError as e:
        raise ValueError(f"SyntaxError: {e}")

    if sandbox_globals is None:
        sandbox_globals = create_sandbox_globals(config)
    original_keys = set(sandbox_globals.keys())

    try:
        exec(code, sandbox_globals)
    except Exception as e:
        raise ValueError(f"{type(e).__name__}: {e}")

    new_callables: list[Any] = []
    for k in set(sandbox_globals.keys()) - original_keys:
        obj = sandbox_globals[k]
        if callable(obj) and hasattr(obj, "__code__"):
            new_callables.append(obj)

    return sandbox_globals, new_callables
