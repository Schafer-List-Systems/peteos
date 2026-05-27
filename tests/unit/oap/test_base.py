"""Tests for AgenticObjectBase."""

import pytest

from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import agentic_object, tool
from peteos.oap.error import Error


class TestAgenticObjectBaseInit:
    def test_default_init(self):
        obj = AgenticObjectBase()
        assert obj.agent is None
        assert obj.role is None
        assert obj._oap_threads == {}

    def test_agent_property(self):
        obj = AgenticObjectBase()
        obj.agent = "mock_agent"
        assert obj.agent == "mock_agent"

    def test_role_property(self):
        obj = AgenticObjectBase()
        obj.role = "mock_role"
        assert obj.role == "mock_role"

    def test_get_thread_state_creates(self):
        obj = AgenticObjectBase()
        state = obj.get_thread_state("thread-1")
        assert state == {"messages": []}
        # Second call returns same state
        assert obj.get_thread_state("thread-1") is state

    def test_get_thread_state_isolated(self):
        obj = AgenticObjectBase()
        state1 = obj.get_thread_state("thread-1")
        state2 = obj.get_thread_state("thread-2")
        assert state1 is not state2


class TestInvokeGatekeeper:
    def test_invoke_disables_target(self):
        @agentic_object()
        class Target(AgenticObjectBase):
            pass

        caller = AgenticObjectBase()
        result = caller.invoke(Target(), "prompt")
        assert isinstance(result, Error)
        assert "not enabled" in result.message

    @pytest.mark.asyncio
    async def test_invoke_enables_target_no_agent(self):
        """invoke() on enabled target still needs an agent."""
        @agentic_object(invoke_sub_agents=True)
        class Target(AgenticObjectBase):
            pass

        caller = AgenticObjectBase()
        with pytest.raises(ValueError, match="No Agent available"):
            await caller.invoke(Target(), "prompt")
