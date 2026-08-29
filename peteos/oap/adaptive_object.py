"""AdaptiveObject - AgenticObject with runtime function definition."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from peteos.oap.agentic_object import AgenticObject
from peteos.engine import Runner
from peteos.oap.decorators import agentic_object, tool
from peteos.sandbox import SandboxBuilder
from peteos.persona.toolmanager import Tool
from peteos.conversation.session import Session

from peteos.oap.agentic_registry import AgenticObjectRegistry

@agentic_object(define_functions=True)
class AdaptiveObject(AgenticObject):
    """
    Create and write reusable Python functions for recurring computations using `define_function`.
    - Prefer already existing functions if applicable!
    - Use detailed and self-descriptive function names! Generic function names yield later conflicts.
    - Build the signature including Python type hints.
    - Address `description` string at yourself.
    - Describe the parameters and return values including their types in the docstring!
    - Formulate the optional `tests` as a member function of this object.
      It will immediately become available for you as tool and as a member function of the `self` object after being defined.
      Use these functions as tool directly or from within python functions.
    - Use `remove_function` to unregister functions you previously registered.
    - You cannot remove built-in/static tools.
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

    def _add_tool(
        self,
        func_name: str,
        docstring: str,
        code: str,
        parameters: dict
    ) -> None:
        """Register a dynamically defined function as an agentic tool.

        Creates a proxy, registers it on the tool manager, stores metadata,
        and auto-approves it.

        Args:
            func_name: The function name to register.
            docstring: Description of the tool.
            parameters: The function's parameter schema.
            runner: Runner injected by the framework.
        """
        # Store metadata.
        self._oap_define_functions[func_name] = {
            "docstring": docstring,
            "parameters": parameters,
            "code": code,
        }

        # Build a proxy that closes over runner, so self.func_name(...) works.
        def proxy(*args: Any, runner: Runner, **kwargs: Any) -> str:
            return getattr(runner.sandbox, func_name)(*args, **kwargs)

        proxy._tool_name = func_name
        defined_tool = Tool(name=func_name, description=docstring, func=proxy, parameters=parameters)
        self.__dict__[func_name] = proxy
        self._oap_tool_manager.register_tool(defined_tool)

        # Auto-approve on instance level — propagated to each session via hook.
        self._oap_auto_approve_tools.append(func_name)

    def _try_compile(self, code: str) -> tuple[str, dict]:
        """Compile code, validating exactly one function. Rolls back on failure.

        Returns (func_name, parameters) on success.
        Raises ValueError if code doesn't compile to exactly one function.
        """
        func_and_params = self._oap_sandbox_builder.add_source_code(code)

        if len(func_and_params) != 1:
            if func_and_params:
                self._oap_sandbox_builder.remove_source_code(func_and_params[0][0])
            names = [f[0] for f in func_and_params] if func_and_params else ["none"]
            raise ValueError(f"expected exactly one function in code. Found: {names}.")

        func_name, parameters = func_and_params[0]
        if self._oap_tool_manager.get_tool(func_name):
            self._oap_sandbox_builder.remove_source_code(func_name)
            raise ValueError(f"tool '{func_name}' already exists")

        return func_name, parameters

    def _define_function(self, code: str, docstring: str) -> str:
        """Define a Python function as a new tool.

        Pass the full function definition as a string and a docstring of what the function does including its arguments and return type.

        Args:
            code: Full Python function definition as a string (e.g. "def my_func(x: int) -> int:\\n    return x * 2").
            docstring: The Python docstring of the function with parameter description.
        """
        func_name, parameters = self._try_compile(code)
        self._add_tool(func_name, docstring, code, parameters)
        return f"OK: registered as '{func_name}'"

    def _delete_tool(self, func_name: str) -> None:
        """Remove a dynamically defined tool.

        Args:
            func_name: The function name to remove.
            runner: Runner injected by the framework.
        """
        del self._oap_define_functions[func_name]
        self._oap_tool_manager._tools.pop(func_name, None)
        self.__dict__.pop(func_name, None)
        self._oap_auto_approve_tools.remove(func_name)

    async def _start_session(self, session: Session) -> Runner:
        runner = await super()._start_session(session)

        def _on_add_member(func_name: str, _params: dict) -> None:
            runner._execution_environment.auto_approve_tools.append(func_name)

        def _on_remove_member(func_name: str) -> None:
            runner._execution_environment.auto_approve_tools.remove(func_name)

        runner.sandbox_builder.add_hook("on_add_member", _on_add_member)
        runner.sandbox_builder.add_hook("on_remove_member", _on_remove_member)

        return runner

    @tool
    def remove_function(self, name: str) -> str:
        """Remove a defined tool.

        Args:
            name: The tool name to remove.
            runner: Optional runner injected by the framework.
        """
        if name not in self._oap_define_functions:
            return f"Error: tool '{name}' not found or not defined by this agent."

        self._oap_sandbox_builder.remove_source_code(name)
        self._delete_tool(name)

        return "OK"

    def _test_function_in_sandbox(
        self,
        code: str,
        mocked_functions: dict[str, str],
        tests: str,
    ) -> None:
        """Compile and test a function in a fully isolated sandbox.

        Creates a new SandboxBuilder with no parent, adds produce_output/
        produce_error helpers, compiles mocks, the main function, and test
        code into it, then runs each test function. Raises ValueError on
        any failure.

        Args:
            code: Full Python function definition to test.
            mocked_functions: Dict mapping method name → code for mock methods.
            tests: Python code containing multiple test functions (each
                accepting self) with assertions.

        Raises:
            ValueError: On compilation failure, validation error, or test failure.
        """
        test_builder = SandboxBuilder("tests")
        test_builder.add_safe_builtins()

        def _test_produce_output(data):
            return f"produce_output({data!r})"

        def _test_produce_error(msg):
            return f"produce_error({msg!r})"

        test_builder.add_proxy("produce_output", _test_produce_output)
        test_builder.add_proxy("produce_error", _test_produce_error)

        test_builder.add_global_func("assert_equal", lambda expected, actual: None if expected == actual else (_ for _ in ()).throw(AssertionError(f"expected {expected!r}, got {actual!r}")))
        test_builder.add_global_func("assert_not_equal", lambda expected, actual: None if expected != actual else (_ for _ in ()).throw(AssertionError(f"expected {expected!r} != {actual!r}")))

        for mock_name, mock_code in mocked_functions.items():
            try:
                funcs = test_builder.add_source_code(mock_code)
            except Exception as e:
                raise ValueError(
                    f"failed to compile mock function '{mock_name}': {e}"
                ) from None
            if len(funcs) != 1:
                names = [f[0] for f in funcs] if funcs else ["none"]
                raise ValueError(f"mock '{mock_name}' should define exactly one function. Found: {names}.")

        try:
            main_funcs = test_builder.add_source_code(code)
        except Exception as e:
            raise ValueError(f"failed to compile main function: {e}") from None
        if len(main_funcs) != 1:
            names = [f[0] for f in main_funcs] if main_funcs else ["none"]
            raise ValueError(f"expected exactly one function in code. Found: {names}.")

        main_func_name, _ = main_funcs[0]
        if self._oap_tool_manager.get_tool(main_func_name):
            raise ValueError(
                f"tool '{main_func_name}' already exists (collision with static tool)"
            )

        try:
            test_funcs = test_builder.add_source_code(tests)
        except Exception as e:
            raise ValueError(f"failed to compile test code: {e}") from None

        sandbox = test_builder.get_sandbox()
        for i, (name, _) in enumerate(test_funcs):
            try:
                getattr(sandbox, name)()
            except Exception as e:
                raise ValueError(f"test {i} ({name}) failed: {e}")

    @tool
    def define_function(
        self,
        code: str,
        docstring: str,
        mocked_functions: dict[str, str] | None = None,
        tests: str | None = None,
    ) -> str:
        """Define a Python member function with optional unit tests.

        Pass the full member function definition as a string and a docstring.
        Optionally provide mocked member functions (name→code) and a tests string
        containing multiple test member functions. Tests run in an isolated sandbox
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

        return self._define_function(code, docstring)
