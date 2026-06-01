"""Simple unit tests for ExecutionEnvironment hooks."""

import pytest

from peteos.chatbot import ChatHistory, ChatBotManager
from peteos.replexecutionenvironment import REPLExecutionEnvironment
from peteos.role import Role
from peteos.toolmanager import ToolManager
from peteos.session import Session
from unittest.mock import MagicMock


@pytest.fixture
def mock_session():
    """Create a mock Session."""
    return MagicMock(spec=Session)


@pytest.fixture
def mock_chatbot():
    """Create a mock ChatBot."""
    return MagicMock()


@pytest.fixture(autouse=True)
def _setup_mock_chatbot():
    """Set up a mock ChatBot in the class-level ChatBotManager for tests."""
    mock = MagicMock()
    ChatBotManager._backends = {"test-backend": MagicMock(models={"test_model": mock})}
    yield
    ChatBotManager._backends.clear()


@pytest.fixture
def mock_role():
    """Create a mock Role."""
    return Role(name="test", description="Test role")


@pytest.fixture
def mock_chatbot():
    """Create a mock ChatBot."""
    return MagicMock()


class TestHookRegistration:
    """Test hook registration and deregistration."""

    def test_register_hook_valid_point(self, mock_role, mock_session):
        """Test registering a hook for a valid hook point."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        def test_hook():
            pass

        env = REPLExecutionEnvironment(chat_history, tool_manager, mock_role)
        env.register_hook("before_tool_execution", test_hook)

        # Check that the hook is registered
        hooks = env._hooks["before_tool_execution"]
        assert len(hooks) == 1
        assert hooks[0].func == test_hook

    def test_register_hook_invalid_point(self, mock_role, mock_session):
        """Test registering a hook for an invalid hook point raises ValueError."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        def test_hook():
            pass

        env = REPLExecutionEnvironment(chat_history, tool_manager, mock_role)

        with pytest.raises(ValueError, match="Unknown hook point"):
            env.register_hook("invalid_hook", test_hook)

    def test_deregister_hook_valid(self, mock_role, mock_session):
        """Test deregistering a valid hook."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        def test_hook():
            pass

        env = REPLExecutionEnvironment(chat_history, tool_manager, mock_role)
        env.register_hook("before_tool_execution", test_hook)
        env.deregister_hook("before_tool_execution", test_hook)

        assert test_hook not in env._hooks["before_tool_execution"]

    def test_deregister_all_hooks(self, mock_role, mock_session):
        """Test deregistering all hooks for a point."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        def hook1():
            pass

        def hook2():
            pass

        env = REPLExecutionEnvironment(chat_history, tool_manager, mock_role)
        env.register_hook("before_tool_execution", hook1)
        env.register_hook("before_tool_execution", hook2)
        env.deregister_all_hooks("before_tool_execution")

        assert len(env._hooks["before_tool_execution"]) == 0

    def test_all_hook_points_exist(self, mock_role, mock_session):
        """Test that all current hook points are registered."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        env = REPLExecutionEnvironment(chat_history, tool_manager, mock_role)

        assert "before_tool_execution" in env._hooks
        assert "after_tool_execution" in env._hooks
        assert "before_notification_publish" in env._hooks
        assert "before_send_to_chatbot" in env._hooks


class TestHookCalling:
    """Test hook calling directly."""

    @pytest.mark.asyncio
    async def test_sync_hook_is_called(self, mock_role, mock_session):
        """Test that a sync hook is called when _call_hooks is invoked."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        hook_called = [False]

        def test_hook(data):
            hook_called[0] = True

        env = REPLExecutionEnvironment(chat_history, tool_manager, mock_role)
        env.register_hook("before_tool_execution", test_hook)

        # Manually call hooks
        result = await env._call_hooks("before_tool_execution", {"name": "test"})
        assert hook_called[0]

    @pytest.mark.asyncio
    async def test_hook_return_value_is_used(self, mock_role, mock_session):
        """Test that return value from sync hook is returned."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        env = REPLExecutionEnvironment(chat_history, tool_manager, mock_role)

        def blocking_hook(data):
            return (False, "Blocked!")

        env.register_hook("before_tool_execution", blocking_hook)

        result = await env._call_hooks("before_tool_execution", {"name": "test"})

        assert result == (False, "Blocked!")

    @pytest.mark.asyncio
    async def test_multiple_hooks_only_first_result_used(self, mock_role, mock_session):
        """Test that first hook returning a value stops hook chain."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        call_order = []

        def hook1(data):
            call_order.append(1)
            return (False, "First")

        def hook2(data):
            call_order.append(2)
            return (True, "Second")

        env = REPLExecutionEnvironment(chat_history, tool_manager, mock_role)
        env.register_hook("before_tool_execution", hook1)
        env.register_hook("before_tool_execution", hook2)

        result = await env._call_hooks("before_tool_execution", {"name": "test"})

        assert result == (False, "First")
        assert call_order == [1]  # hook2 should not be called
