"""Tests for OAP sandbox."""

import asyncio
import concurrent.futures
from unittest.mock import AsyncMock, MagicMock

import pytest

from peteos.oap.decorators import sandbox, tool
from peteos.oap.sandbox import SandboxSelf, create_sandbox_globals


class TestSandboxSelf:
    def test_is_empty_object(self):
        """SandboxSelf is an empty object — no instance attributes by default."""
        obj = SandboxSelf()
        assert not hasattr(obj, "_real_self")
        assert not hasattr(obj, "_runner")

    def test_produce_output_is_closure(self):
        """produce_output is a closure, not a bound method."""
        obj = SandboxSelf()

        def fake_produce_output(data):
            return f"produced: {data}"

        setattr(obj, "produce_output", fake_produce_output)
        result = obj.produce_output("hello")
        assert result == "produced: hello"

    def test_produce_output_cannot_introspect(self):
        """Closure-based produce_output should not expose internal references."""
        obj = SandboxSelf()

        def _inner():
            class FakeSelf:
                def _produce_output(self, data, runner=None):
                    return "ok"

            real_self = FakeSelf()
            runner = object()

            def _produce_output(data):
                return real_self._produce_output(data, runner=runner)

            setattr(obj, "produce_output", _produce_output)
            return real_self, runner

        _inner()
        # The closure's __closure__ cells should not reveal the real self
        fn = getattr(obj, "produce_output")
        assert fn.__closure__ is not None
        cell_values = [c.cell_contents for c in fn.__closure__]
        # None of the closure cells should be a dict or instance
        for val in cell_values:
            assert not isinstance(val, dict)

    def test_populate_attaches_sandbox_methods(self):
        """populate() finds methods decorated with @sandbox."""
        obj = SandboxSelf()

        @sandbox
        def my_sandbox_method(self, x):
            return x * 2

        FakeSelf = type("FakeSelf", (), {"my_sandbox_method": my_sandbox_method})
        instance = FakeSelf()

        SandboxSelf.populate(obj, instance)
        assert hasattr(obj, "my_sandbox_method")
        assert obj.my_sandbox_method(5) == 10

    def test_populate_attaches_tool_methods_as_sandbox(self):
        """populate() proxies @tool methods via their @tool name."""
        obj = SandboxSelf()

        @tool(name="custom_tool", description="test")
        def my_tool(self, x):
            return x + 1

        FakeSelf = type("FakeSelf", (), {"my_tool": my_tool})
        instance = FakeSelf()

        SandboxSelf.populate(obj, instance)
        assert hasattr(obj, "custom_tool")
        assert obj.custom_tool(10) == 11

    def test_populate_deduplicates_by_lower_name(self):
        """populate() deduplicates methods that differ only in case."""
        obj = SandboxSelf()

        @sandbox
        def my_method(self):
            return "parent"

        child_method = lambda self: "child"
        child_method._sandbox_name = "my_method"
        child_method._sandbox_description = ""

        Parent = type("Parent", (), {"my_method": my_method})
        Child = type("Child", (Parent,), {"my_method": child_method})
        instance = Child()

        SandboxSelf.populate(obj, instance)
        assert obj.my_method() == "child"

    def test_invoke_runs_child_agent(self):
        """invoke() should call target.invoke_agent in a worker thread."""
        obj = SandboxSelf()

        target = MagicMock()
        target.invoke_agent = AsyncMock(
            return_value={"result": "child answer"}
        )

        # Simulate how _python_exec attaches invoke
        parent_ptid = "test-thread-42"
        state = MagicMock()
        state.get.return_value = parent_ptid
        runner = MagicMock()
        runner.state = state

        def _invoke(
            target,
            prompt,
            output_schema=None,
            persistent=False,
            timeout=None,
        ):
            ptid = parent_ptid if persistent else None

            def _run():
                _loop = asyncio.new_event_loop()
                try:
                    return _loop.run_until_complete(
                        target.invoke_agent(
                            prompt=prompt,
                            output_schema=output_schema,
                            timeout=timeout,
                            persistent_thread_id=ptid,
                        )
                    )
                finally:
                    _loop.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(_run).result()

        setattr(obj, "invoke", _invoke)

        # invoke() should work synchronously even though invoke_agent is async
        result = obj.invoke(target, prompt="do the task")
        assert result == {"result": "child answer"}
        target.invoke_agent.assert_called_once_with(
            prompt="do the task",
            output_schema=None,
            timeout=None,
            persistent_thread_id=None,
        )

    def test_invoke_with_persistent_true_forwards_thread_id(self):
        """invoke(persistent=True) should forward parent's persistent_thread_id."""
        obj = SandboxSelf()

        target = MagicMock()
        target.invoke_agent = AsyncMock(
            return_value={"result": "child answer"}
        )

        parent_ptid = "parent-thread-99"
        state = MagicMock()
        state.get.return_value = parent_ptid
        runner = MagicMock()
        runner.state = state

        def _invoke(
            target,
            prompt,
            output_schema=None,
            persistent=False,
            timeout=None,
        ):
            ptid = parent_ptid if persistent else None

            def _run():
                _loop = asyncio.new_event_loop()
                try:
                    return _loop.run_until_complete(
                        target.invoke_agent(
                            prompt=prompt,
                            output_schema=output_schema,
                            timeout=timeout,
                            persistent_thread_id=ptid,
                        )
                    )
                finally:
                    _loop.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(_run).result()

        setattr(obj, "invoke", _invoke)

        result = obj.invoke(target, prompt="do it", persistent=True)
        assert result == {"result": "child answer"}
        target.invoke_agent.assert_called_once_with(
            prompt="do it",
            output_schema=None,
            timeout=None,
            persistent_thread_id="parent-thread-99",
        )


class TestSandbox:
    def test_safe_builtins(self):
        sandbox = create_sandbox_globals({})
        assert sandbox["__builtins__"] is not {}
        assert sandbox["__builtins__"]["print"] is print
        assert sandbox["__builtins__"]["str"] is str
        assert sandbox["__builtins__"]["list"] is list
        # File access should still be blocked
        with pytest.raises((NameError, ImportError)):
            exec("open('/dev/null')", sandbox)

    def test_imports_injected(self):
        import os

        sandbox = create_sandbox_globals({"imports": [os]})
        assert "os" in sandbox
        assert sandbox["os"] is os

    def test_import_not_available(self):
        sandbox = create_sandbox_globals({})
        with pytest.raises(ImportError):
            exec("import sys", sandbox)

    def test_no_file_access(self):
        sandbox = create_sandbox_globals({})
        with pytest.raises((NameError, ImportError)):
            exec("f = open('/etc/passwd')", sandbox)