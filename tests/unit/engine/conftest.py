"""Shared pytest fixtures for engine unit tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from peteos.conversation.session import _UserHooks
from peteos.engine.executionenvironment import ExecutionEnvironment


@pytest.fixture
def mock_role():
    role = MagicMock()
    role.name = "test-role"
    return role


@pytest.fixture
def mock_tool_manager():
    tm = MagicMock()
    tool = MagicMock()
    tool.func = lambda a, b: a + b
    tool.func.__name__ = "add"
    tool.parameters = {"a": {"type": "int"}, "b": {"type": "int"}}
    tool.execute = MagicMock(return_value=5)
    tm.get_tool.return_value = tool
    return tm


@pytest.fixture
def mock_session(mock_role, mock_tool_manager):
    session = MagicMock()
    session.role = mock_role
    session.tool_manager = mock_tool_manager
    session._user_hook_owners = {}
    session._user_hook_owner_order = []
    session.user_hooks = _UserHooks(session)
    session.user_hooks["transitive", "on_tool_call"] = []
    return session


@pytest.fixture
def mock_runner(mock_session, mock_role):
    runner = MagicMock()
    runner._session = mock_session
    runner.role = mock_role
    runner._notify_channels = AsyncMock()
    runner._execution_environment = MagicMock()
    runner._execution_environment._elections = {}
    return runner


@pytest.fixture
def env(mock_runner, mock_role, mock_tool_manager):
    env = ExecutionEnvironment(
        tool_manager=mock_tool_manager,
        role=mock_role,
        auto_approve_tools=[],
        tool_failure_policy="abort",
    )
    env._runner = mock_runner
    return env
