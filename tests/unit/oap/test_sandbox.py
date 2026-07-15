"""Tests for OAP sandbox."""

import os

import pytest

from peteos.oap.decorators import sandbox, tool
from peteos.sandbox import SandboxBuilder


class TestSandboxBuilderState:
    """Verify SandboxBuilder creates a clean, isolated state."""

    def test_fresh_builder_has_no_proxies(self):
        """A new SandboxBuilder starts with no registered proxies."""
        builder = SandboxBuilder("fresh")
        assert builder._proxies == []

    def test_fresh_builder_has_no_source_code_functions(self):
        """A new SandboxBuilder starts with no functions from source code."""
        builder = SandboxBuilder("fresh")
        builder.add_safe_builtins()
        # No source code compiled yet — namespace is empty of user functions
        assert builder._sandbox_namespace == {}

    def test_closures_in_proxies_dont_leak_sensitive_objects(self):
        """Proxies registered via add_proxy should not leak real_self."""
        class Sensitive:
            def __init__(self):
                self.secret = "hidden_data"

        sensitive = Sensitive()

        # Use a no-self function so real_self is not required
        def sensitive_fn(value):
            return f"val={value}, secret={sensitive.secret}"

        builder = SandboxBuilder("test")
        builder.add_safe_builtins()
        builder.add_proxy("sensitive_fn", sensitive_fn)

        sandbox = builder.get_sandbox()
        fn = getattr(sandbox, "sensitive_fn")

        # The callable should still be invocable
        result = fn(42)
        assert "val=42" in result

        # Introspecting the proxy should not reveal the sensitive object
        # The proxy is a lambda with no closure, so this is already safe
        closure_cells = fn.__closure__ if hasattr(fn, "__closure__") else None
        if closure_cells:
            for cell in closure_cells:
                cell_val = cell.cell_contents
                assert not isinstance(cell_val, Sensitive), \
                    f"Sensitive object leaked via __closure__ cell"


class TestSandboxBuilderProxies:
    """Verify proxy behavior mirrors the old SandboxSelf.populate behavior."""

    def test_sandbox_decorator_func_can_be_proxied(self):
        """Functions decorated with @sandbox can be registered via add_proxy."""
        _my_sandbox_method = sandbox(lambda x: x * 2)

        builder = SandboxBuilder("test")
        builder.add_safe_builtins()
        builder.add_proxy("my_sandbox_method", _my_sandbox_method)

        sb = builder.get_sandbox()
        assert hasattr(sb, "my_sandbox_method")
        assert sb.my_sandbox_method(5) == 10

    def test_tool_name_used_as_sandbox_name(self):
        """@tool methods should be accessible on the sandbox via their tool name."""
        @tool(name="custom_tool", description="test")
        def my_tool(x):
            return x + 1

        builder = SandboxBuilder("test")
        builder.add_safe_builtins()
        builder.add_proxy("custom_tool", my_tool)

        sandbox = builder.get_sandbox()
        assert hasattr(sandbox, "custom_tool")
        assert sandbox.custom_tool(10) == 11

    def test_duplicate_proxy_name_raises_value_error(self):
        """Adding a proxy with a duplicate name raises ValueError."""
        builder = SandboxBuilder("test")
        builder.add_safe_builtins()
        builder.add_proxy("dup", lambda x: 1)

        with pytest.raises(ValueError, match="already registered"):
            builder.add_proxy("dup", lambda x: 2)

    def test_duplicate_proxy_names_via_add_proxies(self):
        """Adding duplicate names in add_proxies raises ValueError."""
        builder = SandboxBuilder("test")
        builder.add_safe_builtins()
        builder.add_proxies({"dup": lambda x: 1})

        with pytest.raises(ValueError, match="already registered"):
            builder.add_proxies({"dup": lambda x: 2})


class TestSandboxBuilderSecurity:
    """Verify sandbox security: safe builtins, import isolation, no file access."""

    def test_safe_builtins(self):
        """Safe builtins are available; dangerous functions are not."""
        builder = SandboxBuilder("test")
        builder.add_safe_builtins()
        globals_ = builder.get_globals()
        assert "str" in globals_
        assert "int" in globals_
        assert "list" in globals_
        # open should not be available
        assert "open" not in globals_

    def test_imports_injected(self):
        """Imports added via add_imports are available in the sandbox."""
        builder = SandboxBuilder("test")
        builder.add_safe_builtins()
        builder.add_imports([os], None)
        globals_ = builder.get_globals()
        assert "os" in globals_
        assert globals_["os"] is os

    def test_import_not_available(self):
        """Unlisted imports should not be available in the sandbox."""
        builder = SandboxBuilder("test")
        builder.add_safe_builtins()
        # Use add_source_code which executes in the sandbox namespace
        builder.add_source_code("pass")
        # Verify no module was imported
        globals_ = builder.get_globals()
        assert "sys" not in globals_

    def test_no_file_access(self):
        """File open is not available in the sandbox globals."""
        builder = SandboxBuilder("test")
        builder.add_safe_builtins()
        # open is not in safe builtins
        assert "open" not in builder.get_globals()
        # source code compilation doesn't expose open
        builder.add_source_code("pass")
        assert "open" not in builder.get_globals()


class TestSandboxBuilder:
    def test_create_proxy_cross_object_router(self):
        """create_proxy acts as a router: the proxy is placed on a different object
        but when called, the target method on the original object runs.

        Mirrors the manual test:
            - define empty class A
            - define class B with function f(self, *args, **kwargs)
            - create instance a=A()
            - create instance b=B()
            - f_proxy = create_proxy(b.f, real_self=b)
            - setattr(a, 'f_proxy', f_proxy)
            - a.f_proxy(test='hello world')  # calls b.f internally
        """
        class A:
            pass

        class B:
            def f(self, *args, **kwargs):
                return (id(self), args, kwargs)

        a = A()
        b = B()

        proxy = SandboxBuilder.create_proxy(b.f)
        setattr(a, "f_proxy", proxy)

        result = a.f_proxy(test="hello world")
        assert result[0] == id(b)  # self is b, not a
        assert result[1] == ()
        assert result[2] == {"test": "hello world"}

    def test_create_proxy_unbound_function_with_self(self):
        """Unbound function with self: a plain function whose first parameter
        is 'self', stored as an attribute on the real_self object.

        The proxy binds real_self as the first argument via functools.partial.
        """
        class A:
            pass

        class B:
            def f(self, x):
                return (id(self), x)

        a = A()
        b = B()
        b.func = B.f  # store the unbound function

        proxy = SandboxBuilder.create_proxy(b.func, real_self=b)
        setattr(a, "f_proxy", proxy)

        result = a.f_proxy(42)
        assert result[0] == id(b)
        assert result[1] == 42

    def test_create_proxy_no_self_callable(self):
        """No-self callable: a function with no self argument.

        The proxy uses the callable directly without binding.
        """
        class A:
            pass

        def standalone(x):
            return x

        proxy = SandboxBuilder.create_proxy(standalone)
        a = A()
        setattr(a, "f_proxy", proxy)

        result = a.f_proxy(42)
        assert result == 42

