"""AdaptiveObject - AgenticObject with runtime function persistence."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from peteos.oap.agentic_object import AgenticObject, _collect_oap_config
from peteos.oap.decorators import agentic_object
from peteos.oap.sandbox import _exec_sandboxed
from peteos.persona.toolmanager import Tool


def _build_persisted_tool_proxy(
    real_self: "AdaptiveObject",
    func_name: str,
) -> Callable[..., str]:
    """Build a proxy callable that executes a persisted function in SandboxSelf.

    The proxy delegates all sandbox logic to ``_exec_sandboxed``, which
    provides the same environment as ``python_exec``: full globals,
    closure-based produce_output/produce_error, conditional invoke,
    and SandboxSelf.populate() for tool/sandbox method proxies.

    Args:
        real_self: The AdaptiveObject instance.
        func_name: The stored function name.

    Returns:
        A proxy that forwards *args/**kwargs to the stored function.
    """
    config = _collect_oap_config(real_self.__class__)
    stored = real_self._oap_persisted_tools[func_name]

    def proxy(
        *args: Any,
        **kwargs: Any,
    ) -> str:
        result = _exec_sandboxed(config, stored["code"], real_self, None, *args, **kwargs)
        return str(result) if result is not None else "OK"

    return proxy


@agentic_object(persist_functions=True)
class AdaptiveObject(AgenticObject):
    """AgenticObject with runtime function persistence.

    Subclasses inherit persist_functions automatically through MRO.
    This object can write and register new Python functions as tools at
    runtime via ``persist_function`` and remove them via ``remove_tool``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._oap_persisted_tools: dict[str, Any] = {}
        self._register_persisted_tools()

    def _register_persisted_tools(self) -> None:
        """Register persist_function and remove_tool tools."""
        self._oap_tool_manager.register_tool(
            Tool(
                name="persist_function",
                description=(
                    "Persist a Python function as a tool callable by the agent. "
                    "Pass the full function definition as a string (must be a valid `def`). "
                    "The function name and parameters are auto-extracted from the signature. "
                    "The description should explain what the function does and its parameter schema."
                ),
                func=self._persist_function,
            )
        )
        self._oap_tool_manager.register_tool(
            Tool(
                name="remove_tool",
                description=(
                    "Remove a tool that was previously registered by this agent "
                    "via the ``persist_function`` tool. Cannot remove static (class-defined) tools."
                ),
                func=self._remove_tool,
            )
        )

    def _persist_function(self, code: str, description: str) -> str:
        """Persist a Python function definition as a new tool.

        The function code is stored and will be executed in a sandbox with
        SandboxSelf at call time, so the agent has no direct access to the
        real AgenticObject instance.

        Args:
            code: Full Python function definition as a string (e.g. "def my_func(x: int) -> int:\\n    return x * 2").
            description: Description of the tool including parameter explanation.

        Returns:
            "OK: registered as '<name>'" on success, or an error message.
        """
        # Validate the code by compiling it
        try:
            compile(code, "<persisted>", "exec")
        except SyntaxError as e:
            return f"Error: {type(e).__name__}: {e}"

        # Build a sandbox namespace for exec
        sandbox_globals: dict[str, Any] = {
            "__builtins__": {"str": str, "int": int, "float": float, "bool": bool, "list": list,
                             "dict": dict, "set": set, "tuple": tuple, "len": len, "range": range,
                             "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
                             "sorted": sorted, "reversed": reversed, "abs": abs, "min": min,
                             "max": max, "sum": sum, "round": round, "divmod": divmod,
                             "any": any, "all": all, "print": print, "isinstance": isinstance,
                             "type": type},
        }
        try:
            exec(code, sandbox_globals)
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

        # Find the newly defined function
        new_names = [k for k in sandbox_globals if k != "__builtins__" and callable(sandbox_globals[k])]
        if len(new_names) != 1:
            return f"Error: expected exactly one function in code. Found: {new_names or 'none'}."

        func_name = new_names[0]

        # Validate signature
        try:
            sig = inspect.signature(sandbox_globals[func_name])
        except (ValueError, TypeError) as e:
            return f"Error: could not introspect function signature: {e}"

        # Check for name collision with static tools
        if self._oap_tool_manager.get_tool(func_name):
            return f"Error: tool '{func_name}' already exists (collision with static tool)."

        # Build parameters schema from signature
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

        # Store the code string; the actual function will be exec'd at call time
        # inside SandboxSelf, ensuring isolation from the real AgenticObject.
        self._oap_persisted_tools[func_name] = {
            "code": code,
            "description": description,
            "parameters": parameters,
        }

        # Register a proxy as the Tool.func — it executes the code in a
        # sandbox with SandboxSelf at call time.
        proxy = _build_persisted_tool_proxy(self, func_name)
        tool = Tool(name=func_name, description=description, func=proxy, parameters=parameters)
        self._oap_tool_manager.register_tool(tool)

        return f"OK: registered as '{func_name}'"

    def _remove_tool(self, name: str) -> str:
        """Remove a persisted tool.

        Args:
            name: The tool name to remove.

        Returns:
            "OK" on success, or an error message.
        """
        if name not in self._oap_persisted_tools:
            return f"Error: tool '{name}' not found or not persisted by this agent."

        del self._oap_persisted_tools[name]
        self._oap_tool_manager._tools.pop(name, None)
        return "OK"
