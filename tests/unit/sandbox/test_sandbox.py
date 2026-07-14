"""Unit tests for the new Scope-based sandbox architecture."""

from __future__ import annotations

import pytest

from peteos.sandbox import Sandbox, SandboxBuilder
from peteos.sandbox.sandbox_builder import Scope


class TestScope:
    """Tests for the Scope class."""

    def test_single_scope_finds_entry(self):
        scope = Scope("test")
        scope.entries = {"a": 1}
        assert scope.a == 1

    def test_single_scope_raises_on_missing(self):
        scope = Scope("test")
        with pytest.raises(AttributeError):
            scope.nonexistent

    def test_parent_chain_resolution(self):
        child = Scope("child", parent=Scope("parent"))
        child.entries = {"x": 1}
        child.parent.entries = {"y": 2}
        assert child.x == 1
        assert child.y == 2

    def test_child_shadows_parent(self):
        child = Scope("child", parent=Scope("parent"))
        child.entries = {"x": 10}
        child.parent.entries = {"x": 20}
        assert child.x == 10

    def test_deep_parent_chain(self):
        root = Scope("root")
        mid = Scope("mid", parent=root)
        leaf = Scope("leaf", parent=mid)
        root.entries = {"z": 1}
        mid.entries = {"y": 2}
        leaf.entries = {"x": 3}
        assert leaf.x == 3
        assert leaf.y == 2
        assert leaf.z == 1


class TestSandbox:
    """Tests for the Sandbox class."""

    def test_getattr_via_scope_chain(self):
        root = Scope("root")
        sandbox = Sandbox(root)
        root.entries = {"foo": "bar"}
        assert sandbox.foo == "bar"

    def test_getattr_missing_raises(self):
        sandbox = Sandbox(Scope("root"))
        with pytest.raises(AttributeError):
            sandbox.missing

    def test_repr(self):
        # session -> instance -> class chain
        cls_scope = Scope("class")
        inst_scope = Scope("instance", parent=cls_scope)
        sess_scope = Scope("session", parent=inst_scope)
        sess_scope.entries = {}
        sandbox = Sandbox(sess_scope)
        assert "session" in repr(sandbox)
        assert "instance" in repr(sandbox)
        assert "class" in repr(sandbox)

    def test_caller_context_from_innermost(self):
        """No caller context => starts at innermost (self._scope)."""
        session = Scope("session")
        instance = Scope("instance", parent=session)
        root = Scope("root", parent=instance)
        sandbox = Sandbox(session)

        root.entries = {"x": "root"}
        instance.entries = {"x": "instance"}
        session.entries = {"x": "session"}

        assert sandbox.x == "session"


class TestNamespaceProxy:
    """Tests for source code namespace proxies via SandboxBuilder."""

    def _create_builder(self):
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()
        return builder

    def test_add_source_code_creates_proxy(self):
        builder = self._create_builder()
        result = builder.add_source_code(
            "def compute(self, x: int) -> int:\n    return x * 2"
        )
        assert result == [("compute", {"x": {"type": "int", "required": True}})]
        assert "compute" in builder.sandbox_namespace

    def test_proxy_call_with_args(self):
        builder = self._create_builder()
        builder.add_source_code(
            "def add(self, a: int, b: int) -> int:\n    return a + b"
        )

        sandbox = builder.get_sandbox()
        assert sandbox.add(1, 2) == 3

    def test_proxy_call_with_kwargs(self):
        builder = self._create_builder()
        builder.add_source_code(
            "def greet(self, name: str, suffix: str = '!') -> str:\n"
            "    return f'Hello, {name}{suffix}'"
        )

        sandbox = builder.get_sandbox()
        assert sandbox.greet("World") == "Hello, World!"
        assert sandbox.greet("World", suffix="???") == "Hello, World???"


class TestSandboxBuilderChain:
    """Tests for multi-level builder chains."""

    def test_builder_chain_creates_scope_hierarchy(self):
        """Class -> instance -> session chain."""
        class_builder = SandboxBuilder("class")
        instance_builder = SandboxBuilder("instance", base=class_builder)
        session_builder = SandboxBuilder("session", base=instance_builder)

        class_builder.add_safe_builtins()
        instance_builder.add_source_code(
            "def instance_f(self) -> str:\n    return 'instance'"
        )
        session_builder.add_source_code(
            "def session_f(self) -> str:\n    return 'session'"
        )

        sandbox = session_builder.get_sandbox()
        assert sandbox.session_f() == "session"

    def test_shadowing_in_chain(self):
        """Inner level shadows outer level by name."""
        class_builder = SandboxBuilder("class")
        instance_builder = SandboxBuilder("instance", base=class_builder)
        session_builder = SandboxBuilder("session", base=instance_builder)

        class_builder.add_safe_builtins()
        class_builder.add_source_code(
            "def shared(self) -> str:\n    return 'class'"
        )
        instance_builder.add_source_code(
            "def shared(self) -> str:\n    return 'instance'"
        )
        session_builder.add_source_code(
            "def shared(self) -> str:\n    return 'session'"
        )

        sandbox = session_builder.get_sandbox()
        # From session level, should resolve session's shared
        assert sandbox.shared() == "session"

    def test_fallback_to_outer_when_missing(self):
        """If name not in caller's scope, walk up parent chain."""
        class_builder = SandboxBuilder("class")
        instance_builder = SandboxBuilder("instance", base=class_builder)
        session_builder = SandboxBuilder("session", base=instance_builder)

        class_builder.add_safe_builtins()
        class_builder.add_source_code(
            "def outer(self) -> str:\n    return 'class'"
        )
        session_builder.add_source_code(
            "def caller(self) -> str:\n    return self.outer()"
        )

        sandbox = session_builder.get_sandbox()
        assert sandbox.caller() == "class"

    def test_non_virtual_dispatch(self):
        """Caller's scope determines resolution, not call site."""
        class_builder = SandboxBuilder("class")
        instance_builder = SandboxBuilder("instance", base=class_builder)
        session_builder = SandboxBuilder("session", base=instance_builder)

        class_builder.add_safe_builtins()

        # instance defines both f and g
        instance_builder.add_source_code(
            "def f(self) -> str:\n    return self.g()"
        )
        instance_builder.add_source_code(
            "def g(self) -> str:\n    return 'g(instance)'"
        )

        # session overrides g
        session_builder.add_source_code(
            "def g(self) -> str:\n    return 'g(session)'"
        )
        session_builder.add_source_code(
            "def caller(self) -> str:\n    return self.f()"
        )

        sandbox = session_builder.get_sandbox()
        # caller calls f -> f resolves in instance scope -> f calls self.g()
        # _calling_ns is set to instance scope when f runs -> g resolves to instance.g
        result = sandbox.caller()
        assert result == "g(instance)"


class TestMirrorMethods:
    """Tests for mirrored methods via add_proxy."""

    def test_add_proxy_without_self(self):
        """Standalone function with no self parameter."""
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()

        def standalone(x: int) -> int:
            return x + 1

        builder.add_proxy("standalone", standalone)
        sandbox = builder.get_sandbox()
        assert sandbox.standalone(5) == 6

    def test_add_proxy_with_unbound_self(self):
        """Method-like callable with self parameter gets real_self bound."""
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()

        class MyClass:
            value = 42

            def get_value(self) -> int:
                return self.value

        obj = MyClass()
        builder.add_proxy("get_value", obj.get_value, real_self=obj)
        sandbox = builder.get_sandbox()
        assert sandbox.get_value() == 42

    def test_add_proxy_does_not_receive_sandbox_as_self(self):
        """Mirrored methods should NOT receive the sandbox as their self."""
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()

        received_args = []

        def mirrored(self, x: int) -> int:
            received_args.append((type(self).__name__, x))
            return x

        class Core:
            pass

        builder.add_proxy("mirrored", mirrored, real_self=Core())
        sandbox = builder.get_sandbox()
        sandbox.mirrored(10)
        assert received_args == [("Core", 10)]


class TestGetSandbox:
    """Tests for get_sandbox with freeze_namespaces."""

    def test_freeze_namespaces_true_is_isolated(self):
        """freeze_namespaces=True means mutations don't affect snapshot."""
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()
        builder.add_source_code("def get_one(self) -> int:\n    return 1")

        sandbox = builder.get_sandbox(freeze_namespaces=True)
        assert sandbox.get_one() == 1

        # Remove from builder — snapshot should be unaffected
        builder.remove_source_code("get_one")
        assert sandbox.get_one() == 1

    def test_freeze_namespaces_false_shares_entries(self):
        """freeze_namespaces=False means mutations ARE visible."""
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()
        builder.add_source_code("def get_one(self) -> int:\n    return 1")

        sandbox = builder.get_sandbox(freeze_namespaces=False)
        assert sandbox.get_one() == 1

        # Remove from builder
        builder.remove_source_code("get_one")

        # Snapshot reflects change
        with pytest.raises(AttributeError):
            sandbox.get_one

    def test_freeze_namespaces_false_adds_visible(self):
        """Adding after get_sandbox with freeze=False is visible."""
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()

        sandbox = builder.get_sandbox(freeze_namespaces=False)

        # Add after snapshot
        builder.add_source_code("def added(self) -> int:\n    return 42")
        assert sandbox.added() == 42


class TestRemoveSourceCode:
    """Tests for remove_source_code."""

    def test_remove_source_code(self):
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()
        builder.add_source_code("def func(self) -> int:\n    return 1")

        removed = builder.remove_source_code("func")
        assert removed is True
        assert "func" not in builder.sandbox_namespace
        assert builder.remove_source_code("nonexistent") is False

    def test_remove_makes_sandbox_unavailable(self):
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()
        builder.add_source_code("def func(self) -> int:\n    return 1")

        sandbox = builder.get_sandbox(freeze_namespaces=False)
        builder.remove_source_code("func")

        with pytest.raises(AttributeError):
            sandbox.func


class TestMixedSourceAndProxy:
    """Tests for mixing namespace proxies and mirrored methods."""

    def test_source_calls_proxy(self):
        """Source functions can call mirrored methods on self."""
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()

        def core_add(a: int, b: int) -> int:
            return a + b

        builder.add_proxy("core_add", core_add)
        builder.add_source_code(
            "def use_add(self, x: int, y: int) -> int:\n"
            "    return self.core_add(x, y)"
        )

        sandbox = builder.get_sandbox()
        assert sandbox.use_add(3, 4) == 7
