"""AdaptiveObject - AgenticObject with runtime function definition."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from peteos.oap.agentic_object import AgenticObject, _collect_oap_config
from peteos.engine import Runner
from peteos.oap.decorators import agentic_object, tool
from peteos.oap.sandbox import SandboxSelf, create_sandbox_globals, sandbox_compile
from peteos.persona.toolmanager import Tool

from peteos.oap.agentic_registry import AgenticObjectRegistry

@agentic_object(define_functions=True)
class AdaptiveObject(AgenticObject):
    """
    Create and write reusable Python functions for recurring computations using `define_function`!
    Address its description at yourself.
    Describe the parameters and return values including their types in the docstring!
    Formulate the optional `tests` as a member function of this object.
    It will immediately become available for you as tool and as a member function of the `self` object after being defined.
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
        """Extract function name and parameters schema from a callable.

        Args:
            func_obj: A Python callable to introspect.

        Returns:
            A tuple of (func_name, parameters_schema) on success.

        Raises:
            ValueError: If the signature cannot be introspected.
        """
        func_name = func_obj.__name__

        try:
            sig = inspect.signature(func_obj)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"could not introspect function signature: {e}"
            ) from e

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

    def _define_function(self, code: str, docstring: str, runner: Runner) -> str:
        """Define a Python function as a new tool.

        Pass the full function definition as a string and a docstring of what the function does including its arguments and return type.

        Args:
            code: Full Python function definition as a string (e.g. "def my_func(x: int) -> int:\\n    return x * 2").
            docstring: The Python docstring of the function with parameter description.
        """
        config = _collect_oap_config(self.__class__)

        sandbox_globals, new_callables = sandbox_compile(config, code)
        if len(new_callables) != 1:
            names = [c.__name__ for c in new_callables] if new_callables else ["none"]
            raise ValueError(f"expected exactly one function in code. Found: {names}.")

        func_name, parameters = self._extract_func_name_and_params(new_callables[0])
        if self._oap_tool_manager.get_tool(func_name):
            raise ValueError(f"tool '{func_name}' already exists")

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
    def remove_function(self, name: str, runner: Runner) -> str:
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

    def _test_function_in_sandbox(
        self,
        code: str,
        mocked_functions: dict[str, str],
        tests: str,
    ) -> None:
        """Compile and test a function in an isolated sandbox.

        Creates a shared globals namespace, compiles mocks and the main
        function into it, builds a fresh SandboxSelf with all callables
        bound as methods, then runs each test function from the single
        test code string. Stops on the first failure. Raises ValueError on
        any failure.

        Args:
            code: Full Python function definition to test.
            mocked_functions: Dict mapping method name → code for mock methods.
            tests: Python code containing multiple test functions (each
                accepting self) with assertions.

        Raises:
            ValueError: On compilation failure, validation error, or test failure.
        """
        config = _collect_oap_config(self.__class__)

        def _make_test_proxy(func):
            """Proxy that passes sandbox_self to functions expecting self."""
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            expects_self = params[0] == "self"
            def wrapper(*a, **kw):
                if expects_self:
                    return func(sandbox_self, *a, **kw)
                return func(*a, **kw)
            return wrapper

        # 1. Shared sandbox globals
        shared_globals = create_sandbox_globals(config)

        # 2. Compile mocks into shared globals
        all_callables: list[tuple[str, Callable]] = []
        for mock_name, mock_code in mocked_functions.items():
            _, callables = sandbox_compile(config, mock_code, sandbox_globals=shared_globals)
            if len(callables) != 1:
                names = [c.__name__ for c in callables] if callables else ["none"]
                raise ValueError(f"mock '{mock_name}' should define exactly one function. Found: {names}.")
            all_callables.append((mock_name, callables[0]))

        # 3. Compile main function into shared globals
        _, callables = sandbox_compile(config, code, sandbox_globals=shared_globals)
        if len(callables) != 1:
            names = [c.__name__ for c in callables] if callables else ["none"]
            raise ValueError(f"expected exactly one function in code. Found: {names}.")

        # 4. Validate main function
        main_func_name, _ = self._extract_func_name_and_params(callables[0])
        if self._oap_tool_manager.get_tool(main_func_name):
            raise ValueError(
                f"tool '{main_func_name}' already exists (collision with static tool)"
            )
        all_callables.append((main_func_name, callables[0]))

        # 5. Build test sandbox — attach callables via proxy, produce helpers
        sandbox_self = SandboxSelf()
        for name, func in all_callables:
            setattr(sandbox_self, name, _make_test_proxy(func))
        setattr(sandbox_self, "produce_output",
            lambda data: f"produce_output({data!r})")
        setattr(sandbox_self, "produce_error",
            lambda msg: f"produce_error({msg!r})")

        # 6. Compile test code and run each callable as a test
        try:
            _, test_callables = sandbox_compile(config, tests)
        except ValueError as e:
            raise ValueError(f"failed to compile tests: {e}") from e
        for i, test_func in enumerate(test_callables):
            sandbox_self._test = _make_test_proxy(test_func)
            try:
                sandbox_self._test()
            except Exception as e:
                raise ValueError(f"test {i} ({test_func.__name__}) failed: {e}")

    @tool
    def define_function(
        self,
        code: str,
        docstring: str,
        runner: Runner,
        mocked_functions: dict[str, str] | None = None,
        tests: str | None = None,
    ) -> str:
        """Define a Python member function with optional unit tests.

        Pass the full member function definition as a string and a docstring.
        Optionally provide mocked_functions (name→code) and a tests string
        containing multiple test functions. Tests run in an isolated sandbox
        before registration. On failure, a ValueError is raised and the
        function is NOT registered.

        Args:
            code: Full Python member function definition.
            docstring: The Python docstring of the function.
            runner: Runner injected by the framework.
            mocked_functions: Dict of name→code for mock methods on SandboxSelf.
            tests: Python code containing multiple test functions (each
                accepting self) with assertions.

        Raises:
            ValueError: On compilation failure, validation error, or test failure.
        """
        if mocked_functions is not None or tests is not None:
            self._test_function_in_sandbox(
                code, mocked_functions or {}, tests or ""
            )

        return self._define_function(code, docstring, runner)
