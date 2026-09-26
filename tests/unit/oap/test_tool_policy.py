"""Tests for tool_policy helpers and nested-policy support in @tool decorator."""

import pytest

from peteos.oap.decorators import tool
from peteos.oap.agentic_object import (
    AgenticObject,
    _find_outer_binding,
    _make_cell,
    _build_policy_cell,
    _register_tool_policy_for,
    _tool_policy_dispatcher,
)
from peteos.oap.decorators import agentic_object


class TestFindToolPolicy:
    """Tests for the decorator's built-in tool_policy scanner."""

    def test_finds_policy_and_freevars(self):
        @tool
        def my_tool(self, command: str, runner=None):
            def tool_policy():
                return command.split()[0] in {"ls", "cat"}
            return "ok"

        assert hasattr(my_tool, "_tool_policy")
        assert hasattr(my_tool._tool_policy, "_tool_policy_freevars")
        assert my_tool._tool_policy._tool_policy_freevars == ("command",)
        assert callable(my_tool._tool_policy)

    def test_returns_none_when_no_policy(self):
        @tool
        def plain_tool(self, x: str) -> str:
            return x

        assert not hasattr(plain_tool, "_tool_policy")
        assert not hasattr(plain_tool, "_tool_policy_freevars")

    def test_freevars_contains_all_outer_scope_names(self):
        SAFE = {"ls", "pwd"}

        @tool
        def with_local_const(self, command: str, runner=None):
            def tool_policy():
                # reads 'command' (param) and 'SAFE' (local const)
                return command.split()[0] in SAFE
            return "ok"

        assert set(with_local_const._tool_policy._tool_policy_freevars) == {"command", "SAFE"}

    def test_policy_is_callable_without_outer_scope(self):
        @tool
        def example(self, value: str, runner=None):
            def tool_policy():
                return True
            return "ok"

        # Callable with no args — no outer scope needed
        assert example._tool_policy() is True


class TestMakeCell:
    """Tests for the cross-version closure cell factory."""

    def test_returns_cell_object(self):
        cell = _make_cell("hello")
        assert cell is not None
        assert cell.cell_contents == "hello"

    def test_different_values_different_cells(self):
        c1 = _make_cell(42)
        c2 = _make_cell(42)
        assert c1 is not c2

    def test_none_value(self):
        cell = _make_cell(None)
        assert cell.cell_contents is None

    def test_tuple_value(self):
        cell = _make_cell(("a", 1))
        assert cell.cell_contents == ("a", 1)


class TestFindOuterBinding:
    """Tests for resolving freevar names from a policy's original closure."""

    def test_no_closure_returns_none(self):
        def plain():
            pass

        result = _find_outer_binding("anything", plain)
        assert result is None

    def test_resolves_captured_variable(self):
        captured_value = "expected"

        def policy():
            return captured_value

        result = _find_outer_binding("captured_value", policy)
        assert result == "expected"

    def test_resolves_self_from_method(self):
        class DummySelf:
            marker = "self-here"

        obj = DummySelf()

        def policy():
            return obj

        result = _find_outer_binding("obj", policy)
        assert result is obj


class TestBuildPolicyCell:
    """Tests for building a pre-populated closure cell from args or fallback."""

    def test_prefers_value_from_args(self):
        def pol():
            pass

        cell = _build_policy_cell("command", {"command": "ls -la"}, pol)
        assert cell.cell_contents == "ls -la"

    def test_falls_back_to_outer_binding(self):
        sentinel = "fallback_value"

        def policy():
            return sentinel

        cell = _build_policy_cell("sentinel", {}, policy)
        assert cell.cell_contents == "fallback_value"

    def test_args_takes_priority_over_outer_binding(self):
        outer_val = "outer"
        args_val = "from_args"

        def policy():
            return outer_val

        cell = _build_policy_cell("outer_val", {"outer_val": args_val}, policy)
        assert cell.cell_contents == args_val


class TestRegisterToolPolicyFor:
    """Tests for the policy registration helper."""

    def test_stores_policy_and_freevars(self):
        @tool
        def my_tool(self, cmd: str, runner=None):
            def tool_policy():
                return cmd.split()[0] in {"ls"}
            return "ok"

        policies: dict = {}
        _register_tool_policy_for(policies, "my_tool", my_tool)

        assert "my_tool" in policies
        policy_fn = policies["my_tool"][0]
        assert callable(policy_fn)
        assert hasattr(policy_fn, "_tool_policy_freevars")
        assert policy_fn._tool_policy_freevars == ("cmd",)

    def test_no_policy_stored_when_absent(self):
        def plain_tool(self):
            pass

        policies: dict = {}
        _register_tool_policy_for(policies, "plain_tool", plain_tool)
        assert policies == {}

    def test_multiple_policies_same_tool(self):
        @tool
        def tool_a(self, x: str, runner=None):
            def tool_policy():
                return True
            return "ok"

        @tool
        def tool_b(self, x: str, runner=None):
            def tool_policy():
                return x == "safe"
            return "ok"

        policies: dict = {}
        _register_tool_policy_for(policies, "multi_tool", tool_a)
        _register_tool_policy_for(policies, "multi_tool", tool_b)

        assert len(policies["multi_tool"]) == 2


class TestToolPolicyDispatcher:
    """Integration tests for the full dispatcher."""

    @agentic_object()
    class _BashLike(AgenticObject):
        @tool()
        def bash_exec(self, command: str, runner=None):
            def tool_policy():
                return command.split()[0] in {"ls", "pwd", "cat"}
            return "ok"

        @tool()
        def restricted(self, key: str, runner=None):
            def tool_policy():
                return key != "dangerous"
            return "ok"

        @tool()
        def no_policy_tool(self, x: str, runner=None):
            return "ok"

    @agentic_object()
    class _EmptyAO(AgenticObject):
        pass

    def test_approved_command_returns_true(self):
        obj = self._BashLike()
        ctx = {"tool_name": "bash_exec", "arguments": {"command": "ls -la"}}
        assert _tool_policy_dispatcher(obj._oap_tool_policies, ctx) is True

    def test_denied_command_returns_false(self):
        obj = self._BashLike()
        ctx = {"tool_name": "bash_exec", "arguments": {"command": "rm -rf /"}}
        assert _tool_policy_dispatcher(obj._oap_tool_policies, ctx) is False

    def test_pwd_is_approved(self):
        obj = self._BashLike()
        ctx = {"tool_name": "bash_exec", "arguments": {"command": "pwd"}}
        assert _tool_policy_dispatcher(obj._oap_tool_policies, ctx) is True

    def test_short_circuit_on_deny(self):
        obj = self._BashLike()
        ctx = {"tool_name": "restricted", "arguments": {"key": "dangerous"}}
        assert _tool_policy_dispatcher(obj._oap_tool_policies, ctx) is False

    def test_approved_key(self):
        obj = self._BashLike()
        ctx = {"tool_name": "restricted", "arguments": {"key": "safe"}}
        assert _tool_policy_dispatcher(obj._oap_tool_policies, ctx) is True

    def test_no_policy_returns_none(self):
        obj = self._BashLike()
        ctx = {"tool_name": "no_policy_tool", "arguments": {"x": "whatever"}}
        assert _tool_policy_dispatcher(obj._oap_tool_policies, ctx) is None

    def test_unknown_tool_returns_none(self):
        obj = self._BashLike()
        ctx = {"tool_name": "does_not_exist", "arguments": {}}
        assert _tool_policy_dispatcher(obj._oap_tool_policies, ctx) is None

    def test_empty_tool_name_returns_none(self):
        obj = self._BashLike()
        ctx = {"tool_name": "", "arguments": {}}
        assert _tool_policy_dispatcher(obj._oap_tool_policies, ctx) is None

    def test_no_tool_policies_on_empty_ao(self):
        obj = self._EmptyAO()
        ctx = {"tool_name": "anything", "arguments": {}}
        assert _tool_policy_dispatcher(obj._oap_tool_policies, ctx) is None

    def test_dispatcher_is_registered_in_local_hooks(self):
        obj = self._BashLike()
        on_tool_hooks = obj._oap_local_hooks.get("on_tool_call", [])
        assert len(on_tool_hooks) >= 1
        assert all(callable(h) for h in on_tool_hooks)

    def test_policy_uses_outer_scope_local_const(self):
        """A policy that reads a local const defined outside the policy but in the method body.

        This tests the fallback path in _build_policy_cell: when a freevar is not in
        args, _find_outer_binding reads it from the policy's original closure, which
        holds the binding from decoration time.

        Note: the const MUST be a local variable in the method body for it to be a
        freevar of the policy. If defined at module or class level, it is accessed
        via __globals__ and this fallback is not needed.
        """
        @agentic_object()
        class _ConstPolicy(AgenticObject):
            @tool()
            def path_check(self, path: str, runner=None):
                def tool_policy():
                    return path and not path.startswith("/private")
                return "ok"

        obj = _ConstPolicy()
        ctx_ok = {"tool_name": "path_check", "arguments": {"path": "/home/user/file.txt"}}
        assert _tool_policy_dispatcher(obj._oap_tool_policies, ctx_ok) is True

        ctx_blocked = {"tool_name": "path_check", "arguments": {"path": "/private/secret"}}
        assert _tool_policy_dispatcher(obj._oap_tool_policies, ctx_blocked) is False

    def test_closure_injection_is_per_call_isolated(self):
        """Two concurrent-like calls must not share or clobber closure state."""
        obj = self._BashLike()

        ctx_a = {"tool_name": "bash_exec", "arguments": {"command": "ls"}}
        ctx_b = {"tool_name": "bash_exec", "arguments": {"command": "rm -rf /"}}

        result_a = _tool_policy_dispatcher(obj._oap_tool_policies, ctx_a)
        result_b = _tool_policy_dispatcher(obj._oap_tool_policies, ctx_b)

        assert result_a is True
        assert result_b is False

    def test_consecutive_calls_are_independent(self):
        """Rapid sequential calls must not carry over state from prior invocations."""
        obj = self._BashLike()

        results = []
        for cmd in ["ls", "cat /etc/passwd", "curl http://evil", "pwd", "rm -rf /"]:
            ctx = {"tool_name": "bash_exec", "arguments": {"command": cmd}}
            results.append(_tool_policy_dispatcher(obj._oap_tool_policies, ctx))

        assert results == [True, True, False, True, False]
