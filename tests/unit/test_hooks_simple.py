"""Simple unit tests for ExecutionEnvironment hooks."""

import asyncio

import pytest

from peteos.chatbot import ChatHistory, Message, ContentPart
from peteos.replexecutionenvironment import REPLExecutionEnvironment
from peteos.role import Role
from peteos.toolmanager import ToolManager
from unittest.mock import MagicMock


@pytest.fixture
def mock_role():
    """Create a mock Role."""
    return Role(name="test", description="Test role")


@pytest.fixture
def mock_chatbot_manager():
    """Create a mock ChatBotManager."""
    manager = MagicMock()
    mock_chatbot = MagicMock()
    manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]
    return manager


@pytest.fixture
def mock_chatbot():
    """Create a mock ChatBot."""
    return MagicMock()


class TestHookRegistration:
    """Test hook registration and deregistration."""

    def test_register_hook_valid_point(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test registering a hook for a valid hook point."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        def test_hook():
            pass

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)
        env.register_hook("before_tool_execution", test_hook)

        # Check that the hook is registered (wrapped in partial with no extra args)
        hooks = env._hooks["before_tool_execution"]
        assert len(hooks) == 1
        assert hooks[0].func == test_hook

    def test_register_hook_invalid_point(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test registering a hook for an invalid hook point raises ValueError."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        def test_hook():
            pass

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)

        with pytest.raises(ValueError, match="Unknown hook point"):
            env.register_hook("invalid_hook", test_hook)

    def test_deregister_hook_valid(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test deregistering a valid hook."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        def test_hook():
            pass

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)
        env.register_hook("before_tool_execution", test_hook)
        env.deregister_hook("before_tool_execution", test_hook)

        assert test_hook not in env._hooks["before_tool_execution"]

    def test_deregister_all_hooks(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test deregistering all hooks for a point."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        def hook1():
            pass

        def hook2():
            pass

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)
        env.register_hook("before_tool_execution", hook1)
        env.register_hook("before_tool_execution", hook2)
        env.deregister_all_hooks("before_tool_execution")

        assert len(env._hooks["before_tool_execution"]) == 0

    def test_all_hook_points_exist(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test that all hook points are registered."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)

        assert "before_tool_execution" in env._hooks
        assert "after_tool_execution" in env._hooks
        assert "before_loop_continue" in env._hooks
        assert "before_loop_exit" in env._hooks


class TestHookCalling:
    """Test hook calling directly."""

    @pytest.mark.asyncio
    async def test_sync_hook_is_called(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test that a sync hook is called when _call_hooks is invoked."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        hook_called = [False]

        def test_hook(data):
            hook_called[0] = True

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)
        env.register_hook("before_tool_execution", test_hook)

        # Manually call hooks (not via loop)
        await env._call_hooks("before_tool_execution", {"name": "test"})

        assert hook_called[0]

    @pytest.mark.asyncio
    async def test_async_hook_is_called(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test that an async hook is called when _call_hooks is invoked."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        hook_called = [False]

        async def test_hook(data):
            hook_called[0] = True

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)
        env.register_hook("before_tool_execution", test_hook)

        await env._call_hooks("before_tool_execution", {"name": "test"})

        assert hook_called[0]

    @pytest.mark.asyncio
    async def test_hook_return_value_is_used(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test that return value from sync hook is returned."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)

        def blocking_hook(data):
            return (False, "Blocked!")

        env.register_hook("before_tool_execution", blocking_hook)

        result = await env._call_hooks("before_tool_execution", {"name": "test"})

        assert result == (False, "Blocked!")

    @pytest.mark.asyncio
    async def test_async_hook_return_value_is_used(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test that return value from async hook is returned."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)

        async def blocking_hook(data):
            return (False, "Async blocked!")

        env.register_hook("before_tool_execution", blocking_hook)

        result = await env._call_hooks("before_tool_execution", {"name": "test"})

        assert result == (False, "Async blocked!")

    @pytest.mark.asyncio
    async def test_multiple_hooks_only_first_result_used(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test that first hook returning a value stops hook chain."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        call_order = []

        def hook1(data):
            call_order.append(1)
            return (False, "First")

        def hook2(data):
            call_order.append(2)
            return (True, "Second")

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)
        env.register_hook("before_tool_execution", hook1)
        env.register_hook("before_tool_execution", hook2)

        result = await env._call_hooks("before_tool_execution", {"name": "test"})

        assert result == (False, "First")
        assert call_order == [1]  # hook2 should not be called

    @pytest.mark.asyncio
    async def test_before_loop_continue_hook_can_interrupt(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test that before_loop_continue hook can interrupt the loop."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]
        chat_history.append_message(Message(role="user", content=[ContentPart(part_type="text", text="Hello!")]))

        exit_calls = []

        async def exit_hook(reason):
            exit_calls.append(reason)

        async def interrupt_hook(delta_messages):
            return (True, "interrupted_by_hook")

        class MockResponse:
            def __init__(self):
                self._data = {
                    "role": "assistant",
                    "text": "",
                    "content": [{"type": "tool_use", "name": "test_tool", "arguments": "{}"}]
                }
            @property
            def data(self): return self._data
            def __aiter__(self): return self
            async def __anext__(self): raise StopAsyncIteration

        async def mock_send_message(history, streaming=True):
            return MockResponse()

        mock_chatbot.send_message = mock_send_message

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)
        env.register_hook("before_loop_continue", interrupt_hook)
        env.register_hook("before_loop_exit", exit_hook)

        await env.run()

        # Loop should have been interrupted by hook
        assert len(exit_calls) == 1
        assert exit_calls[0] == "interrupted_by_hook"
