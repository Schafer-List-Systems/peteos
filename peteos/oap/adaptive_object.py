"""AdaptiveObject - AgenticObject with runtime function definition."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from peteos.oap.agentic_object import AgenticObject, _collect_oap_config
from peteos.engine import Runner
from peteos.oap.decorators import agentic_object, tool
from peteos.oap.sandbox import sandbox_compile
from peteos.persona.toolmanager import Tool

from peteos.oap.agentic_registry import AgenticObjectRegistry

@agentic_object(define_functions=True)
class AdaptiveObject(AgenticObject):
    """
    Create and write reusable Python functions for recurring computations using `define_function`!
    Address its description at yourself.
    Describe the parameters and return values including their types in the docstring!
    It will immediately become available for you as tool after being defined.
    Use these functions as tool directly or from within python functions.
    Use `remove_function` to unregister functions you previously registered.
    You cannot remove built-in/static tools.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        AgenticObjectRegistry.register(cls)

    def __init__(self) -> None:
        super().__init__()
        self._oap_define_functions: dict[str, Any] = {}

    def _extract_func_name_and_params(self, func_obj: Callable) -> tuple[str, dict]:
        """Extract function name and parameters schema from a callables.

        Args:
            func_obj: A Python callable to introspect.

        Returns:
            A tuple of (func_name, parameters_schema) on success, or ("Error: ...", {}) on failure.
        """
        func_name = func_obj.__name__

        try:
            sig = inspect.signature(func_obj)
        except (ValueError, TypeError) as e:
            return f"Error: could not introspect function signature: {e}", {}

        # Check for name collision with static tools
        if self._oap_tool_manager.get_tool(func_name):
            return f"Error: tool '{func_name}' already exists (collision with static tool)", {}

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

        return func_name, parameters

    def _build_defined_tool_proxy(self, func_name: str) -> Callable[..., str]:
        """Build a proxy callable that executes a defined function in SandboxSelf.

        Args:
            func_name: The stored function name.

        Returns:
            A proxy that forwards *args/**kwargs to the stored function.
        """
        config = _collect_oap_config(self.__class__)
        stored = self._oap_define_functions[func_name]

        def proxy(
            *args: Any,
            runner: Runner | None = None,
            **kwargs: Any,
        ) -> str:
            return self._call_sandboxed(config, stored["code"], runner, *args, **kwargs)

        return proxy

    @tool
    def define_function(self, code: str, docstring: str, runner: "Runner | None") -> str:
        """Define a Python function as a new tool.

        Pass the full function definition as a string and a docstring of what the function does including its arguments and return type.

        Args:
            code: Full Python function definition as a string (e.g. "def my_func(x: int) -> int:\\n    return x * 2").
            docstring: The Python docstring of the function with parameter description.
        """
        config = _collect_oap_config(self.__class__)
        result = sandbox_compile(config, code)
        if isinstance(result, str):
            return result
        sandbox_globals, new_callables = result

        if len(new_callables) != 1:
            names = [c.__name__ for c in new_callables] if new_callables else ["none"]
            return f"Error: expected exactly one function in code. Found: {names}."

        func_name, parameters = self._extract_func_name_and_params(new_callables[0])
        if isinstance(func_name, str) and func_name.startswith("Error:"):
            return func_name

        # Store the code string; the actual function will be exec'd at call time
        # inside SandboxSelf, ensuring isolation from the real AgenticObject.
        self._oap_define_functions[func_name] = {
            "code": code,
            "docstring": docstring,
            "parameters": parameters,
        }

        # Register a proxy as the Tool.func — it executes the code in a
        # sandbox with SandboxSelf at call time.
        proxy = self._build_defined_tool_proxy(func_name)
        proxy._tool_name = func_name
        defined_tool = Tool(name=func_name, description=docstring, func=proxy, parameters=parameters)

        self.__dict__[func_name] = proxy
        self._oap_tool_manager.register_tool(defined_tool)

        # Auto-approve tool for this AND for future sessions 
        self._oap_auto_approve_tools.append(func_name)
        runner._execution_environment.auto_approve_tools.append(func_name)

        return f"OK: registered as '{func_name}'"

    @tool
    def remove_function(self, name: str, runner: "Runner | None") -> str:
        """Remove a defined tool.

        Args:
            name: The tool name to remove.
            runner: Optional runner injected by the framework.
        """
        if name not in self._oap_define_functions:
            return f"Error: tool '{name}' not found or not defined by this agent."

        del self._oap_define_functions[name]
        self._oap_tool_manager._tools.pop(name, None)
        self.__dict__.pop(name, None)

        self._oap_auto_approve_tools.remove(name)
        runner._execution_environment.auto_approve_tools.remove(name)

        return "OK"
