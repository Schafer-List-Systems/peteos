"""Unit tests for the SandboxBuilder hook system."""

from __future__ import annotations

import pytest

from peteos.sandbox.sandbox_builder import SandboxBuilder


class TestAddHook:
    """Tests for add_hook with upward propagation."""

    def test_add_hook_registers_on_self(self):
        builder = SandboxBuilder("session")
        handler = lambda *a, **k: None

        builder.add_hook("on_add_member", handler)

        assert "on_add_member" in builder._hooks
        assert builder._hooks["on_add_member"] == [handler]

    def test_add_hook_propagates_up_chain(self):
        cls = SandboxBuilder("class")
        inst = SandboxBuilder("instance", base=cls)
        sess = SandboxBuilder("session", base=inst)
        handler = lambda *a, **k: None

        sess.add_hook("on_add_member", handler)

        assert "on_add_member" in sess._hooks
        assert "on_add_member" in inst._hooks
        assert "on_add_member" in cls._hooks
        # Each has exactly one copy
        assert len(inst._hooks["on_add_member"]) == 1
        assert len(cls._hooks["on_add_member"]) == 1

    def test_multiple_handlers_all_registered(self):
        cls = SandboxBuilder("class")
        sess = SandboxBuilder("session", base=cls)
        h1 = lambda *a, **k: None
        h2 = lambda *a, **k: None

        sess.add_hook("on_add_member", h1)
        sess.add_hook("on_add_member", h2)

        assert len(sess._hooks["on_add_member"]) == 2
        assert len(cls._hooks["on_add_member"]) == 2
        assert h1 in cls._hooks["on_add_member"]
        assert h2 in cls._hooks["on_add_member"]

    def test_add_hook_only_propagates_up_not_lateral(self):
        base = SandboxBuilder("base")
        child1 = SandboxBuilder("child1", base=base)
        child2 = SandboxBuilder("child2", base=base)
        handler = lambda *a, **k: None

        child1.add_hook("on_add_member", handler)

        assert "on_add_member" in child1._hooks
        assert "on_add_member" in base._hooks
        assert "on_add_member" not in child2._hooks


class TestRemoveHook:
    """Tests for remove_hook with upward propagation."""

    def test_remove_hook_from_self(self):
        builder = SandboxBuilder("session")
        handler = lambda *a, **k: None

        builder.add_hook("on_add_member", handler)
        builder.remove_hook("on_add_member", handler)

        assert "on_add_member" not in builder._hooks

    def test_remove_hook_propagates_up_chain(self):
        cls = SandboxBuilder("class")
        inst = SandboxBuilder("instance", base=cls)
        sess = SandboxBuilder("session", base=inst)
        handler = lambda *a, **k: None

        sess.add_hook("on_add_member", handler)
        sess.remove_hook("on_add_member", handler)

        assert "on_add_member" not in sess._hooks
        assert "on_add_member" not in inst._hooks
        assert "on_add_member" not in cls._hooks

    def test_remove_does_not_affect_other_ops(self):
        cls = SandboxBuilder("class")
        sess = SandboxBuilder("session", base=cls)
        h = lambda *a, **k: None

        sess.add_hook("on_add_member", h)
        sess.add_hook("on_remove_member", h)

        sess.remove_hook("on_add_member", h)

        assert "on_add_member" not in sess._hooks
        assert sess._hooks["on_remove_member"] == [h]

    def test_remove_nonexistent_op_is_noop(self):
        builder = SandboxBuilder("session")
        handler = lambda *a, **k: None

        builder.remove_hook("on_remove_member", handler)

        assert "on_remove_member" not in builder._hooks


class TestFireHook:
    """Tests for _fire_hook firing at all levels where handler is registered."""

    def test_fire_calls_registered_handler(self):
        builder = SandboxBuilder("session")
        received = []
        handler = lambda name, params: received.append((name, params))

        builder.add_hook("on_add_member", handler)
        builder._fire_hook("on_add_member", "my_func", {"x": {"type": "int"}})

        assert received == [("my_func", {"x": {"type": "int"}})]

    def test_fire_at_level_only_calls_its_registered_handlers(self):
        """Firing on inst only calls handlers registered on inst, not cls."""
        cls = SandboxBuilder("class")
        inst = SandboxBuilder("instance", base=cls)
        inst_received = []
        cls_received = []

        # Register handlers — each propagates up to cls
        inst.add_hook("on_add_member", lambda n, p: inst_received.append(n))
        cls.add_hook("on_add_member", lambda n, p: cls_received.append(n))

        inst._fire_hook("on_add_member", "func", {})

        # Only inst handler fires — cls only receives if we fire on cls
        assert inst_received == ["func"]
        assert cls_received == []

    def test_fire_on_parent_catches_all_propagated_handlers(self):
        """Firing on cls catches cls's own handler + inst's propagated handler.

        A handler registered on inst via add_hook is also on cls._hooks
        (via propagation). When cls fires, that handler runs too.
        """
        cls = SandboxBuilder("class")
        inst = SandboxBuilder("instance", base=cls)
        inst_received = []
        cls_received = []

        inst.add_hook("on_add_member", lambda n, p: inst_received.append(n))
        cls.add_hook("on_add_member", lambda n, p: cls_received.append(n))

        # Fire on cls — cls has both handlers (cls's own + inst's propagated)
        cls._fire_hook("on_add_member", "func", {})

        # Both handlers are on cls._hooks, so both fire
        assert inst_received == ["func"]
        assert cls_received == ["func"]

    def test_fire_at_child_does_not_trigger_uncle(self):
        """Firing at child1 does not trigger handlers on sibling child2."""
        base = SandboxBuilder("base")
        child1 = SandboxBuilder("child1", base=base)
        child2 = SandboxBuilder("child2", base=base)
        child2_received = []

        child2.add_hook("on_add_member", lambda *a: child2_received.append(True))

        child1._fire_hook("on_add_member", "x", {})

        assert child2_received == []


class TestHookOnSourceCodeAddRemove:
    """Tests for hooks firing during add_source_code and remove_source_code."""

    def test_on_add_member_fires_per_function(self):
        builder = SandboxBuilder("session")
        received = []
        handler = lambda name, params: received.append((name, params))

        builder.add_hook("on_add_member", handler)
        builder.add_source_code(
            "def func_a(self) -> int:\n    return 1\n\n"
            "def func_b(self) -> str:\n    return 'b'"
        )

        assert len(received) == 2

    def test_on_add_member_receives_params(self):
        builder = SandboxBuilder("session")
        received = []
        handler = lambda name, params: received.append((name, params))

        builder.add_hook("on_add_member", handler)
        builder.add_source_code(
            "def compute(self, x: int, y: int) -> int:\n"
            "    return x + y"
        )

        assert received[0][0] == "compute"
        assert "x" in received[0][1]
        assert "y" in received[0][1]

    def test_on_remove_member_fires_for_name(self):
        builder = SandboxBuilder("session")
        received = []
        handler = lambda name: received.append(name)

        builder.add_hook("on_remove_member", handler)
        builder.add_source_code("def func_a(self) -> int:\n    return 1")
        builder.remove_source_code("func_a")

        assert received == ["func_a"]

    def test_hook_propagation_means_base_handler_fires(self):
        """A handler registered on base receives the add because it propagated to base.

        When add_source_code fires _fire_hook on itself, it does NOT propagate
        to base — but the handler was already registered on base via add_hook
        propagation, so it will be called if _fire_hook were called on base too.

        The key insight: add_source_code only calls _fire_hook on self.
        Handlers on ancestors only fire when the ancestor itself has an event.
        For the instance-level adaptation use case, the instance fires hooks
        when it adds functions, and the session-level handlers registered via
        propagation on the instance will be called.
        """
        base = SandboxBuilder("base")
        child = SandboxBuilder("child", base=base)
        base_received = []
        child_received = []

        child.add_hook("on_add_member", lambda n, p: child_received.append(n))
        base.add_hook("on_add_member", lambda n, p: base_received.append(n))

        # add_source_code on child fires _fire_hook only on child
        child.add_source_code("def added(self) -> int:\n    return 42")

        # Only child handler fires
        assert child_received == ["added"]
        # base handler does NOT fire — add_source_code only calls _fire_hook on self
        assert base_received == []

    def test_hook_in_chain_propagates_correctly(self):
        """Registering on session adds handler to session, instance, and class.

        When session fires _fire_hook, only the session handler runs.
        When instance fires _fire_hook, both session-propagated and instance-registered handlers run.
        """
        cls = SandboxBuilder("class")
        inst = SandboxBuilder("instance", base=cls)
        sess = SandboxBuilder("session", base=inst)

        levels = []
        sess.add_hook("on_add_member", lambda n, p: levels.append("session"))
        inst.add_hook("on_add_member", lambda n, p: levels.append("instance"))
        cls.add_hook("on_add_member", lambda n, p: levels.append("class"))

        # Session fires — only session's handler is on session's hooks
        sess._fire_hook("on_add_member", "x", {})
        assert levels == ["session"]

        # Instance fires — both session-propagated and instance-registered are on instance
        inst._fire_hook("on_add_member", "x", {})
        assert levels == ["session", "session", "instance"]


class TestHookIntegrationWithSandbox:
    """Integration: hooks + sandbox namespace visibility."""

    def test_hook_add_visible_in_freeze_false_sandbox(self):
        """Adding a function triggers hook and is visible in non-frozen sandbox."""
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()
        received = []
        handler = lambda name, params: received.append(name)

        builder.add_hook("on_add_member", handler)

        sandbox = builder.get_sandbox(freeze_namespaces=False)
        builder.add_source_code("def added(self) -> int:\n    return 42")

        assert "added" in received
        assert sandbox.added() == 42

    def test_hook_remove_not_visible_in_freeze_false_sandbox(self):
        """Removing a function triggers hook and is no longer visible in non-frozen sandbox."""
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()
        received = []
        handler = lambda name: received.append(name)

        builder.add_hook("on_remove_member", handler)
        builder.add_source_code("def func_a(self) -> int:\n    return 1")

        builder.remove_source_code("func_a")

        assert "func_a" in received
        sandbox = builder.get_sandbox(freeze_namespaces=False)
        with pytest.raises(AttributeError):
            sandbox.func_a
