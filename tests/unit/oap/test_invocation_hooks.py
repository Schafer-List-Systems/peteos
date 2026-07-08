"""Tests for invocation hooks (on_invoke, on_invoke_complete)."""

from unittest.mock import AsyncMock, MagicMock

from peteos.oap.base import AgenticObject
from peteos.oap.decorators import agentic_object, tool
from peteos.oap.error import Error
from peteos.conversation.session import Session


# --- Test fixtures ---

@agentic_object()
class HookTestAO(AgenticObject):
    """A simple object for testing invocation hooks."""

    @tool()
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b


def _make_mock_runner(produced="done", error=None):
    """Create a mock Runner whose state.get distinguishes produced vs error.

    Default ``produced="done"`` makes the loop exit immediately since
    ``produced_data`` is truthy. Pass explicit values for scenarios that
    need to exercise error or timeout paths.
    """
    mock_runner = MagicMock()
    mock_state = MagicMock()

    def state_get(key, default=None):
        if key == "_oap_produced_data":
            return produced
        if key == "_oap_error":
            return error
        return default

    mock_state.get = state_get
    mock_runner.state = mock_state
    mock_runner.queue_message = AsyncMock()
    mock_runner.wait_for_idle = AsyncMock()
    mock_runner.execution_environment = MagicMock()
    return mock_runner


# --- on_invoke hook tests ---

class TestOnInvoke:
    """Test on_invoke hook prevents invocation."""

    async def test_on_invoke_prevents_with_error_string(self):
        """A hook returning a string prevents the invocation."""
        obj = HookTestAO()
        hook_called = []
        mock_runner = _make_mock_runner()

        def deny_hook(ctx):
            hook_called.append(True)
            return "Access denied by hook"

        async def fake_start_session(session):
            return mock_runner

        obj._start_session = fake_start_session

        result = await obj.invoke_agent(
            prompt="test prompt",
            hooks={"on_invoke": [deny_hook]},
        )
        assert isinstance(result, Error)
        assert result.message == "Access denied by hook"
        assert len(hook_called) == 1

    async def test_on_invoke_first_prevention_stops_subsequent_hooks(self):
        """When first hook prevents, subsequent hooks in the list don't fire."""
        obj = HookTestAO()
        call_order = []
        mock_runner = _make_mock_runner()

        def deny_hook(ctx):
            call_order.append("deny")
            return "Forbidden"

        def no_ops_hook(ctx):
            call_order.append("noop")
            return None

        async def fake_start_session(session):
            return mock_runner

        obj._start_session = fake_start_session

        result = await obj.invoke_agent(
            prompt="test prompt",
            hooks={"on_invoke": [deny_hook, no_ops_hook]},
        )
        assert isinstance(result, Error)
        assert result.message == "Forbidden"
        assert call_order == ["deny"]

    async def test_on_invoke_hook_receives_correct_context(self):
        """The on_invoke hook receives role, prompt, and session in context."""
        obj = HookTestAO()
        received_ctx = {}

        def inspect_hook(ctx):
            received_ctx.update(ctx)
            return None

        mock_runner = _make_mock_runner()

        async def fake_start_session(session):
            return mock_runner

        obj._start_session = fake_start_session

        await obj.invoke_agent(
            prompt="my custom prompt",
            hooks={"on_invoke": [inspect_hook]},
        )

        assert received_ctx.get("role") == "HookTestAO"
        assert received_ctx.get("prompt") == "my custom prompt"
        assert isinstance(received_ctx.get("session"), Session)

    async def test_on_invoke_no_hooks_does_not_fire(self):
        """When no on_invoke hooks provided, no hook callback is invoked."""
        obj = HookTestAO()
        mock_runner = _make_mock_runner()

        async def fake_start_session(session):
            return mock_runner

        obj._start_session = fake_start_session

        await obj.invoke_agent(prompt="test")
        # Just verify it runs without error.

    async def test_on_invoke_hook_returning_none_does_not_prevent(self):
        """A hook returning None does not prevent the invocation."""
        obj = HookTestAO()
        hook_called = []
        mock_runner = _make_mock_runner()

        def hook(ctx):
            hook_called.append(True)
            return None

        async def fake_start_session(session):
            return mock_runner

        obj._start_session = fake_start_session

        await obj.invoke_agent(
            prompt="test",
            hooks={"on_invoke": [hook]},
        )

        assert len(hook_called) == 1

    async def test_on_invoke_multiple_hooks_all_run_when_no_prevention(self):
        """All on_invoke hooks fire when none return a string."""
        obj = HookTestAO()
        call_order = []
        mock_runner = _make_mock_runner()

        def hook_a(ctx):
            call_order.append("a")
            return None

        def hook_b(ctx):
            call_order.append("b")
            return None

        async def fake_start_session(session):
            return mock_runner

        obj._start_session = fake_start_session

        await obj.invoke_agent(
            prompt="test",
            hooks={"on_invoke": [hook_a, hook_b]},
        )

        assert call_order == ["a", "b"]


class TestOnInvokeComplete:
    """Test on_invoke_complete hook fires after agent finishes."""

    async def test_on_invoke_complete_fires_on_produce_output(self):
        """on_invoke_complete fires when the agent produces output."""
        obj = HookTestAO()
        received_ctx = {}

        mock_runner = _make_mock_runner(produced="42")

        async def fake_start_session(session):
            return mock_runner

        obj._start_session = fake_start_session

        await obj.invoke_agent(
            prompt="test",
            output_schema=int,
            hooks={"on_invoke_complete": [lambda ctx: received_ctx.update(ctx)]},
        )

        assert received_ctx.get("role") == "HookTestAO"
        assert "test" in received_ctx.get("prompt", "")
        assert isinstance(received_ctx.get("session"), Session)
        assert received_ctx.get("result") == 42

    async def test_on_invoke_complete_fires_on_error(self):
        """on_invoke_complete fires when the agent produces an error."""
        obj = HookTestAO()
        received_ctx = {}

        mock_runner = _make_mock_runner(produced=None, error="simulated error")

        async def fake_start_session(session):
            return mock_runner

        obj._start_session = fake_start_session

        await obj.invoke_agent(
            prompt="test",
            hooks={"on_invoke_complete": [lambda ctx: received_ctx.update(ctx)]},
        )

        assert isinstance(received_ctx.get("result"), Error)
        assert received_ctx.get("result").message == "simulated error"

    async def test_on_invoke_complete_does_not_fire_when_prevented(self):
        """on_invoke_complete does not fire when on_invoke prevents."""
        obj = HookTestAO()
        complete_fired = []

        mock_runner = _make_mock_runner()

        async def fake_start_session(session):
            return mock_runner

        obj._start_session = fake_start_session

        await obj.invoke_agent(
            prompt="test",
            hooks={
                "on_invoke": [lambda ctx: "blocked"],
                "on_invoke_complete": [lambda ctx: complete_fired.append(True)],
            },
        )

        assert len(complete_fired) == 0

    async def test_on_invoke_complete_context_has_expected_keys(self):
        """on_invoke_complete context has role, prompt, session, result."""
        obj = HookTestAO()
        keys_seen = set()

        mock_runner = _make_mock_runner(produced="hello")

        async def fake_start_session(session):
            return mock_runner

        obj._start_session = fake_start_session

        await obj.invoke_agent(
            prompt="my prompt",
            hooks={"on_invoke_complete": [
                lambda ctx: keys_seen.update(ctx.keys())
            ]},
        )

        assert "result" in keys_seen
        assert "role" in keys_seen
        assert "prompt" in keys_seen
        assert "session" in keys_seen
