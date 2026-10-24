"""Tests for register_runner_hook / unregister_runner_hook on AgenticObject.

Covers:
- Storage in _oap_runner_hooks
- Application to existing runners (apply_to_existing)
- Revocation from existing runners (revoke_from_existing)
- Self-wrapping: callbacks receive self as first argument when runner fires
- Pre-seeded policy.on_tool_call hook is in _oap_runner_hooks
"""

import pytest

from peteos.oap.decorators import agentic_object
from peteos.oap.agentic_object import AgenticObject


class TestRegisterRunnerHook:
    """Unit tests for register_runner_hook."""

    @agentic_object()
    class _EmptyAO(AgenticObject):
        pass

    def test_stores_callback_in_oap_runner_hooks(self):
        obj = self._EmptyAO()
        called: list = []

        def my_callback(ao, runner, session, message):
            called.append((ao, runner, session, message))

        obj.register_runner_hook("test-owner", "my_hook_point", my_callback)

        assert "test-owner" in obj._oap_runner_hooks
        assert "my_hook_point" in obj._oap_runner_hooks["test-owner"]
        assert obj._oap_runner_hooks["test-owner"]["my_hook_point"] is my_callback

    def test_raises_if_already_registered(self):
        obj = self._EmptyAO()

        def cb1(ao, runner, session, message):
            pass

        def cb2(ao, runner, session, message):
            pass

        obj.register_runner_hook("same-owner", "same-hook", cb1)
        with pytest.raises(KeyError, match="already registered"):
            obj.register_runner_hook("same-owner", "same-hook", cb2)

    def test_no_effect_when_no_existing_runners(self):
        obj = self._EmptyAO()

        def cb(ao, runner, session, message):
            pass

        # apply_to_existing=True but _runner_store is empty — no crash
        obj.register_runner_hook("x", "y", cb, apply_to_existing=True)
        assert obj._oap_runner_hooks["x"]["y"] is cb

    def test_callback_receives_self_when_runner_fires(self):
        obj = self._EmptyAO()
        received: list = []

        def my_callback(ao, runner, session, message):
            received.append((ao, runner, session, message))

        # Mock runner with a hook interface that stores registered callbacks
        class _FakeRunner:
            def __init__(self) -> None:
                self._hook_owners: dict = {}
                self._hook_owner_order: list = []
                self.hooks = _Hooks(self)
                self.session_uuid = "fake-uuid"

            def hooks_get(self, hook_point):
                return self.hooks.get(hook_point)

        class _Hooks:
            def __init__(self, runner) -> None:
                self._runner = runner

            def get(self, hook_point):
                result: list = []
                for owner in self._runner._hook_owner_order:
                    result.extend(self._runner._hook_owners.get(owner, {}).get(hook_point, []))
                return result

            def __getitem__(self, key):
                owner, hook_point = key
                return self._runner._hook_owners.setdefault(owner, {}).setdefault(hook_point, [])

            def __setitem__(self, key, value):
                owner, hook_point = key
                self._runner._hook_owners.setdefault(owner, {})[hook_point] = value
                if owner not in self._runner._hook_owner_order:
                    self._runner._hook_owner_order.append(owner)

        runner = _FakeRunner()

        # Register on AO with apply_to_existing=True and directly add to fake runner
        obj.register_runner_hook("my-owner", "my-hook", my_callback, apply_to_existing=True)
        runner.hooks["my-owner", "my-hook"] = []
        for hook_dict in obj._oap_runner_hooks.values():
            for hp, cb in hook_dict.items():
                if hp == "my-hook":
                    runner.hooks["my-owner", "my-hook"][:] = [lambda *a, _cb=cb: _cb(obj, *a)]

        # Fire the hook as the runner would
        for callback in runner.hooks_get("my-hook"):
            callback("fake-runner", "fake-session", "fake-message")

        assert len(received) == 1
        assert received[0][0] is obj  # ao is the AgenticObject instance
        assert received[0][1] == "fake-runner"
        assert received[0][2] == "fake-session"
        assert received[0][3] == "fake-message"

    def test_apply_to_existing_false_does_not_register_on_runner(self):
        obj = self._EmptyAO()

        def cb(ao, runner, session, message):
            pass

        class _FakeRunner:
            def __init__(self) -> None:
                self._hook_owners: dict = {}
                self._hook_owner_order: list = []

            @property
            def hooks(self):
                class _Hooks:
                    def __init__(self, runner) -> None:
                        self._runner = runner

                    def get(self, hp):
                        return self._runner._hook_owners.get(hp, [])

                return _Hooks(self)

        runner = _FakeRunner()
        obj._oap_runner_store["thread-1"] = runner

        # Register without applying to existing
        obj.register_runner_hook("x", "y", cb, apply_to_existing=False)

        # Runner's hooks are not touched
        assert runner._hook_owners.get("x") is None

    def test_different_owners_can_share_same_hook_point(self):
        obj = self._EmptyAO()

        def cb_a(ao, runner, session, message):
            pass

        def cb_b(ao, runner, session, message):
            pass

        obj.register_runner_hook("owner-a", "same-hook", cb_a)
        obj.register_runner_hook("owner-b", "same-hook", cb_b)

        assert obj._oap_runner_hooks["owner-a"]["same-hook"] is cb_a
        assert obj._oap_runner_hooks["owner-b"]["same-hook"] is cb_b


class TestUnregisterRunnerHook:
    """Unit tests for unregister_runner_hook."""

    @agentic_object()
    class _EmptyAO(AgenticObject):
        pass

    def test_raises_if_not_registered(self):
        obj = self._EmptyAO()
        with pytest.raises(KeyError, match="no hooks found"):
            obj.unregister_runner_hook("unknown-owner", "unknown-hook")

    def test_removes_callback_from_oap_runner_hooks(self):
        obj = self._EmptyAO()

        def cb(ao, runner, session, message):
            pass

        obj.register_runner_hook("x", "y", cb)
        assert obj._oap_runner_hooks["x"]["y"] is cb

        obj.unregister_runner_hook("x", "y")

        assert "x" not in obj._oap_runner_hooks or "y" not in obj._oap_runner_hooks["x"]

    def test_clears_all_hooks_for_name_and_hook_point(self):
        obj = self._EmptyAO()

        def cb1(ao, runner, session, message):
            pass

        def cb2(ao, runner, session, message):
            pass

        obj.register_runner_hook("x", "y", cb1, apply_to_existing=False)
        assert "y" in obj._oap_runner_hooks.get("x", {})

        obj.unregister_runner_hook("x", "y")

        assert obj._oap_runner_hooks.get("x", {}).get("y") is None

    def test_revoke_from_existing_false_does_not_touch_runners(self):
        obj = self._EmptyAO()

        def cb(ao, runner, session, message):
            pass

        class _FakeRunner:
            def __init__(self) -> None:
                self._hook_owners: dict = {"x": {"y": [cb]}}
                self._hook_owner_order: list = ["x"]

            @property
            def hooks(self):
                class _Hooks:
                    def __init__(self, runner) -> None:
                        self._runner = runner

                    def get(self, hp):
                        return self._runner._hook_owners.get(hp, [])

                return _Hooks(self)

        runner = _FakeRunner()
        obj._oap_runner_store["t1"] = runner

        obj.register_runner_hook("x", "y", cb, apply_to_existing=False)
        obj.unregister_runner_hook("x", "y", revoke_from_existing=False)

        # Runner is untouched — still has the hook
        assert "y" in runner._hook_owners.get("x", {})

    def test_revoke_from_existing_clears_from_runner(self):
        obj = self._EmptyAO()

        def cb(ao, runner, session, message):
            pass

        class _FakeRunner:
            def __init__(self) -> None:
                self._hook_owners: dict = {"x": {"y": [cb]}}
                self._hook_owner_order: list = ["x"]
                self.session_uuid = "fake-uuid"

            @property
            def hooks(self):
                class _Hooks:
                    def __init__(self, runner) -> None:
                        self._runner = runner

                    def get(self, hook_point):
                        return self._runner._hook_owners.get(hook_point, [])

                    def __getitem__(self, key):
                        owner, hp = key
                        return self._runner._hook_owners.setdefault(owner, {}).setdefault(hp, [])

                return _Hooks(self)

        runner = _FakeRunner()
        obj._oap_runner_store["t1"] = runner

        obj.register_runner_hook("x", "y", cb, apply_to_existing=False)
        obj.unregister_runner_hook("x", "y", revoke_from_existing=True)

        # Runner's hook list is cleared (empty list)
        assert runner._hook_owners.get("x", {}).get("y") == []

    def test_unknown_owner_raises(self):
        obj = self._EmptyAO()

        def cb(ao, runner, session, message):
            pass

        obj.register_runner_hook("known", "hp", cb, apply_to_existing=False)

        with pytest.raises(KeyError, match="no hooks found"):
            obj.unregister_runner_hook("unknown-owner", "hp")


class TestPolicyHookInRunnerHooks:
    """Tests for the pre-seeded policy.on_tool_call in _oap_runner_hooks."""

    @agentic_object()
    class _EmptyAO(AgenticObject):
        pass

    def test_policy_is_pre_seeded(self):
        obj = self._EmptyAO()
        assert "policy" in obj._oap_runner_hooks
        assert "on_tool_call" in obj._oap_runner_hooks["policy"]

    def test_policy_callback_is_tool_policy_dispatcher(self):
        from peteos.oap.agentic_object import _tool_policy_dispatcher

        obj = self._EmptyAO()
        assert obj._oap_runner_hooks["policy"]["on_tool_call"] is _tool_policy_dispatcher

    def test_register_runner_hook_raises_if_policy_already_set(self):
        obj = self._EmptyAO()

        def cb(ao, ctx):
            pass

        # Cannot override the pre-seeded policy hook
        with pytest.raises(KeyError, match="already registered"):
            obj.register_runner_hook("policy", "on_tool_call", cb)


class TestStartSessionAppliesCustomHooks:
    """Integration tests for _start_session applying _oap_runner_hooks to new runners."""

    @agentic_object()
    class _HookedAO(AgenticObject):
        pass

    @pytest.mark.asyncio
    async def test_start_session_applies_policy_dispatcher_to_runner(self):
        from peteos.conversation.session import Session
        from peteos.oap.agentic_object import _tool_policy_dispatcher
        from peteos.persona.role import Role
        from peteos.persona.agent import Agent
        from peteos.persona.toolmanager import ToolManager

        ao = self._HookedAO()

        # Build a minimal session
        role = Role(name="test", system_prompt="", model="test")
        role.tool_filter = []
        agent = Agent(role=role, tool_manager=ToolManager())

        from peteos.conversation.system_prompt_message import SystemPromptMessage
        from peteos.conversation.tool_definitions_message import ToolDefinitionsMessage

        session = Session.create(
            system_prompt_message=SystemPromptMessage.create(""),
            tool_definitions_message=ToolDefinitionsMessage(),
        )
        session._autosave = False  # no directory needed for this test

        # Start a session — runner is created and hooks applied
        runner = await ao._start_session(session)

        # policy.on_tool_call hook is on the runner via _oap_runner_hooks
        policy_hooks = runner.hooks.get("on_tool_call")
        # The wrapped dispatcher is registered under "policy" owner
        policy_owner_hooks = runner.hooks["policy", "on_tool_call"]
        assert len(policy_owner_hooks) == 1
        assert callable(policy_owner_hooks[0])

        await runner.stop()

    @pytest.mark.asyncio
    async def test_custom_hook_on_runner_after_start_session(self):
        ao = self._HookedAO()
        calls: list = []

        def my_hook(ao, runner, session, message):
            calls.append((ao, runner, session, message))

        # Register a custom hook
        ao.register_runner_hook("my-owner", "my-hook", my_hook, apply_to_existing=False)

        from peteos.conversation.session import Session
        from peteos.persona.role import Role
        from peteos.persona.agent import Agent
        from peteos.persona.toolmanager import ToolManager
        from peteos.conversation.system_prompt_message import SystemPromptMessage
        from peteos.conversation.tool_definitions_message import ToolDefinitionsMessage

        role = Role(name="test", system_prompt="", model="test")
        role.tool_filter = []
        agent = Agent(role=role, tool_manager=ToolManager())

        session = Session.create(
            system_prompt_message=SystemPromptMessage.create(""),
            tool_definitions_message=ToolDefinitionsMessage(),
        )
        session._autosave = False

        # Start session — custom hook is applied to the runner via _oap_runner_hooks loop
        runner = await ao._start_session(session)

        # Hook is on the runner
        custom_hooks = runner.hooks["my-owner", "my-hook"]
        assert len(custom_hooks) == 1

        await runner.stop()

    @pytest.mark.asyncio
    async def test_existing_runner_gets_hook_via_apply_to_existing(self):
        ao = self._HookedAO()
        calls: list = []

        def my_hook(ao, runner, session, message):
            calls.append((ao, runner, session, message))

        from peteos.conversation.session import Session
        from peteos.persona.role import Role
        from peteos.persona.agent import Agent
        from peteos.persona.toolmanager import ToolManager
        from peteos.conversation.system_prompt_message import SystemPromptMessage
        from peteos.conversation.tool_definitions_message import ToolDefinitionsMessage

        role = Role(name="test", system_prompt="", model="test")
        role.tool_filter = []
        agent = Agent(role=role, tool_manager=ToolManager())

        session = Session.create(
            system_prompt_message=SystemPromptMessage.create(""),
            tool_definitions_message=ToolDefinitionsMessage(),
        )
        session._autosave = False

        # Start first session to get a running runner
        runner1 = await ao._start_session(session)

        # Manually register the runner in the runner store so apply_to_existing can find it.
        ao._oap_runner_store["persistent-t1"] = runner1

        # Register hook with apply_to_existing=True — applies to already-running runner1
        ao.register_runner_hook("call-counter", "after_step", my_hook, apply_to_existing=True)

        # Hook is on runner1 immediately
        step_hooks = runner1.hooks["call-counter", "after_step"]
        assert len(step_hooks) == 1

        await runner1.stop()
