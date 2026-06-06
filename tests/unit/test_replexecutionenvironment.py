"""Unit tests for REPLExecutionEnvironment."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from peteos.chatbot import ChatBot
from peteos.chatbot import ChatBotManager
from peteos.chatbot import ChatHistory
from peteos.chatbot import Message
from peteos.chatbot import ContentPart
from peteos.replexecutionenvironment import REPLExecutionEnvironment
from peteos.role import Role
from peteos.toolmanager import ToolManager
from peteos.executionenvironment import ExecStatus
from peteos.session import Session


@pytest.fixture
def mock_role():
    """Create a mock Role with default model regex."""
    return Role(name="test", description="Test role")


def _make_mock_session(chat_history, tool_manager, has_unfinished=False, has_pending=False):
    """Create a mock Session with proper pop_pending_tool_call."""
    mock = MagicMock(spec=Session)
    mock.has_unfinished_tool_call.return_value = has_unfinished
    mock.has_pending_tool_call.return_value = has_pending
    mock._pending_tool_calls = []
    mock.chat_history = chat_history
    mock.tool_manager = tool_manager

    def has_reviewed():
        for rec in mock._pending_tool_calls:
            if rec.approval_status != "pending":
                return True
            return False
        return False

    mock.has_reviewed_tool_call.side_effect = has_reviewed

    def pop_record():
        if mock._pending_tool_calls:
            return mock._pending_tool_calls.pop(0)
        return None

    mock.pop_pending_tool_call = pop_record

    async def append_and_notify(message):
        chat_history.append_message(message)

    mock.append_and_notify = append_and_notify
    return mock


@pytest.fixture
def mock_session():
    """Create a mock Session."""
    return _make_mock_session(ChatHistory(), ToolManager())


@pytest.fixture
def mock_chatbot():
    """Create a mock ChatBot."""
    return MagicMock()


@pytest.fixture(autouse=True)
def _setup_mock_chatbot(mock_chatbot):
    """Set up a mock ChatBot in the class-level ChatBotManager for tests."""
    ChatBotManager._backends = {"test-backend": MagicMock(models={"test_model": mock_chatbot})}
    yield
    ChatBotManager._backends.clear()


async def _run_loop(env, session, max_steps=10):
    """Simple step loop for tests (was previously run() / _run_impl)."""
    for _ in range(max_steps):
        status, _ = await env.step(session)
        if status in (ExecStatus.FINISHED, ExecStatus.INTERRUPTED, ExecStatus.ERROR):
            break
        elif status == ExecStatus.CONTINUE:
            continue
        elif status == ExecStatus.PENDING:
            break
        else:
            break


class TestREPLExecutionEnvironmentInit:
    """Test REPLExecutionEnvironment initialization."""

    def test_init(self, mock_role, mock_chatbot, mock_session):
        """Test basic initialization."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        env = REPLExecutionEnvironment(chat_history, tool_manager, mock_role)

        assert env.chatbot == mock_chatbot
        assert env._interrupt is False

    def test_interset_flag_default_false(self, mock_role, mock_chatbot, mock_session):
        """Test interrupt flag is False by default."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        env = REPLExecutionEnvironment(chat_history, tool_manager, mock_role)

        assert env._interrupt is False


class TestSetInterruptClearInterrupt:
    """Test interrupt flag methods."""

    def test_set_interrupt(self, mock_role, mock_chatbot, mock_session):
        """Test set_interrupt sets flag to True."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        env = REPLExecutionEnvironment(chat_history, tool_manager, mock_role)

        env.set_interrupt()

        assert env._interrupt is True

    def test_clear_interrupt(self, mock_role, mock_chatbot, mock_session):
        """Test clear_interrupt sets flag to False."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        env = REPLExecutionEnvironment(chat_history, tool_manager, mock_role)
        env.set_interrupt()

        env.clear_interrupt()

        assert env._interrupt is False


class TestREPLRunBasicConversation:
    """Test basic REPL loop without tool calls."""

    @pytest.mark.asyncio
    async def test_run_basic_conversation(self, mock_role,  mock_chatbot, mock_session):
        """Test single turn conversation with final answer."""
        chat_history = ChatHistory()
        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Hello!")]
        ))
        tool_manager = ToolManager()

        mock_session = _make_mock_session(chat_history, tool_manager, has_unfinished=False)

        class MockResponse:
            def __init__(self):
                self._data = {
                    "role": "assistant",
                    "content": [{"type": "text", "content": "Hello! How can I help?"}]
                }

            @property
            def data(self):
                return self._data

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        async def mock_send_message(history, streaming=True):
            return MockResponse()

        mock_chatbot.send_message = mock_send_message

        env = REPLExecutionEnvironment(chat_history, tool_manager, mock_role)
        await _run_loop(env, mock_session)

        # Should have 2 messages: user, assistant answer
        assert len(chat_history.messages) == 2
        assert chat_history.messages[0].content[0].type == "text"
        assert chat_history.messages[0].content[0].text == "Hello!"
        assert chat_history.messages[1].content[0].type == "text"
        assert chat_history.messages[1].content[0].text == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_run_with_thinking_content(self, mock_role,  mock_chatbot, mock_session):
        """Test response with reasoning content."""
        chat_history = ChatHistory()
        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Solve 2+2")]
        ))
        tool_manager = ToolManager()
        mock_session = _make_mock_session(chat_history, tool_manager, has_unfinished=False)

        reasoning_content = "Let me calculate this step by step..."
        text_content = "4"

        class MockResponse:
            def __init__(self):
                self._data = {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "content": reasoning_content},
                        {"type": "text", "content": text_content}
                    ]
                }

            @property
            def data(self):
                return self._data

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        async def mock_send_message(history, streaming=True):
            return MockResponse()

        mock_chatbot.send_message = mock_send_message

        env = REPLExecutionEnvironment(chat_history, tool_manager, mock_role)
        await _run_loop(env, mock_session)

        assert len(chat_history.messages) == 2
        assert chat_history.messages[0].content[0].type == "text"
        assert chat_history.messages[1].content[0].type == "reasoning"
        assert chat_history.messages[1].content[0].data["reasoning"] == reasoning_content
        assert chat_history.messages[1].content[1].type == "text"
        assert chat_history.messages[1].content[1].text == text_content

    @pytest.mark.asyncio
    async def test_run_loop_terminates_on_final_answer(self, mock_role,  mock_chatbot, mock_session):
        """Test that loop terminates when no tool_calls in response."""
        chat_history = ChatHistory()
        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Hello!")]
        ))
        tool_manager = ToolManager()
        mock_session = _make_mock_session(chat_history, tool_manager, has_unfinished=False)

        call_count = [0]

        class MockResponse:
            def __init__(self):
                self._data = {
                    "role": "assistant",
                    "content": [{"type": "text", "content": "Final answer"}]
                }

            @property
            def data(self):
                return self._data

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        async def mock_send_message(history, streaming=True):
            call_count[0] += 1
            return MockResponse()

        mock_chatbot.send_message = mock_send_message

        env = REPLExecutionEnvironment(chat_history, tool_manager, mock_role)
        await _run_loop(env, mock_session)

        # Should only call send_message once
        assert call_count[0] == 1


class TestREPLRunWithToolCalls:
    """Test REPL loop with tool calls."""

    @pytest.mark.asyncio
    async def test_run_tool_call_detected(self, mock_role,  mock_chatbot, mock_session):
        """Test that tool calls in response are executed."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        def get_weather(city: str) -> str:
            return f"Weather in {city}: sunny"

        tool_manager.register_tool(func=get_weather)
        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="What's the weather?")]
        ))

        final_answer = "It's sunny in London."

        class MockResponse:
            def __init__(self, data):
                self._data = data

            @property
            def data(self):
                return self._data

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        responses = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tc1", "name": "get_weather", "arguments": json.dumps({"city": "London"})}]},
            {"role": "assistant", "content": [{"type": "text", "content": final_answer}]}
        ]
        response_idx = [0]

        async def mock_send_message(history, streaming=True):
            idx = response_idx[0]
            response_idx[0] += 1
            return MockResponse(responses[idx])

        mock_chatbot.send_message = mock_send_message

        mock_session = _make_mock_session(chat_history, tool_manager, has_unfinished=False)
        pending_record = MagicMock()
        pending_record.tool_call = {"name": "get_weather", "id": "tc1", "arguments": json.dumps({"city": "London"})}
        pending_record.approval_status = "approved"
        mock_session._pending_tool_calls = [pending_record]
        mock_session.has_reviewed_tool_call.return_value = True
        mock_session.has_pending_tool_call.return_value = False

        env = REPLExecutionEnvironment(chat_history, tool_manager, mock_role)
        await _run_loop(env, mock_session)

        assert len(chat_history.messages) == 4
        assert chat_history.messages[0].get_role() == "user"
        assert chat_history.messages[1].get_role() == "assistant"
        tool_uses = [
            item for item in chat_history.messages[1].content
            if isinstance(item, ContentPart) and item.type == "tool_use"
        ]
        assert len(tool_uses) == 1
        assert tool_uses[0].data.get("name") == "get_weather"
        assert chat_history.messages[2].get_role() == "tool_result"
        assert chat_history.messages[2].content[0].type == "tool_result"
        assert chat_history.messages[3].get_role() == "assistant"
        assert chat_history.messages[3].content[0].text == final_answer

    @pytest.mark.asyncio
    async def test_run_tool_call_loop_continues(self, mock_role,  mock_chatbot, mock_session):
        """Test that loop continues after tool call with new response."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        def add(a: int, b: int) -> int:
            return a + b

        tool_manager.register_tool(func=add)
        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Add 2+2")]
        ))

        class MockResponse:
            def __init__(self, data):
                self._data = data

            @property
            def data(self):
                return self._data

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        responses = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tc1", "name": "add", "arguments": json.dumps({"a": 2, "b": 2})}]},
            {"role": "assistant", "content": [{"type": "text", "content": "The answer is 4"}]}
        ]
        response_idx = [0]

        async def mock_send_message(history, streaming=True):
            idx = response_idx[0]
            response_idx[0] += 1
            return MockResponse(responses[idx])

        mock_chatbot.send_message = mock_send_message

        mock_session = _make_mock_session(chat_history, tool_manager, has_unfinished=False)
        pending_record = MagicMock()
        pending_record.tool_call = {"name": "add", "id": "tc1", "arguments": json.dumps({"a": 2, "b": 2})}
        pending_record.approval_status = "approved"
        mock_session._pending_tool_calls = [pending_record]
        mock_session.has_reviewed_tool_call.return_value = True
        mock_session.has_pending_tool_call.return_value = False

        env = REPLExecutionEnvironment(chat_history, tool_manager, mock_role)
        await _run_loop(env, mock_session)

        assert response_idx[0] == 2
        assert len(chat_history.messages) == 4
        assert chat_history.messages[1].get_role() == "assistant"
        tool_uses = [
            item for item in chat_history.messages[1].content
            if isinstance(item, ContentPart) and item.type == "tool_use"
        ]
        assert len(tool_uses) == 1
        assert tool_uses[0].data.get("name") == "add"


class TestREPLRunInterrupt:
    """Test interrupt functionality."""

    @pytest.mark.asyncio
    async def test_run_interrupt_immediately(self, mock_role,  mock_chatbot, mock_session):
        """Test that interrupt stops the loop before processing."""
        chat_history = ChatHistory()
        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Hello!")]
        ))
        tool_manager = ToolManager()
        mock_session = _make_mock_session(chat_history, tool_manager, has_unfinished=False)

        class MockResponse:
            def __init__(self):
                self._data = {
                    "role": "assistant",
                    "content": [{"type": "text", "content": "Final answer"}]
                }

            @property
            def data(self):
                return self._data

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        async def mock_send_message(history, streaming=True):
            return MockResponse()

        mock_chatbot.send_message = mock_send_message

        env = REPLExecutionEnvironment(chat_history, tool_manager, mock_role)

        env.set_interrupt()

        await _run_loop(env, mock_session)

        assert env._interrupt is True


class TestContinuousBehaviorPolicy:
    """Test continuous behavior_policy changes step() return behavior."""

    @pytest.mark.asyncio
    async def test_continuous_no_yield_on_text(self, mock_chatbot):
        """Continuous role produces text but no yield_back — returns CONTINUE."""
        continuous_role = Role(name="continuous", description="Continuous agent", behavior_policy="continuous")
        chat_history = ChatHistory()
        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Hello")]
        ))
        tool_manager = ToolManager()

        class MockResponse:
            @property
            def data(self):
                return {"role": "assistant", "content": [{"type": "text", "content": "Hello!"}]}

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        async def mock_send(history, streaming=True):
            return MockResponse()

        mock_chatbot.send_message = mock_send

        mock_session = _make_mock_session(chat_history, tool_manager, has_unfinished=False)
        env = REPLExecutionEnvironment(chat_history, tool_manager, continuous_role)
        status, _ = await env.step(mock_session)

        assert status == ExecStatus.CONTINUE

    @pytest.mark.asyncio
    async def test_responsive_still_yields_on_text(self, mock_chatbot):
        """Responsive role produces text — returns FINISHED (regression)."""
        responsive_role = Role(name="responsive", description="Responsive agent", behavior_policy="responsive")
        chat_history = ChatHistory()
        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Hello")]
        ))
        tool_manager = ToolManager()

        class MockResponse:
            @property
            def data(self):
                return {"role": "assistant", "content": [{"type": "text", "content": "Hello!"}]}

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        async def mock_send(history, streaming=True):
            return MockResponse()

        mock_chatbot.send_message = mock_send

        mock_session = _make_mock_session(chat_history, tool_manager, has_unfinished=False)
        env = REPLExecutionEnvironment(chat_history, tool_manager, responsive_role)
        status, _ = await env.step(mock_session)

        assert status == ExecStatus.FINISHED

    @pytest.mark.asyncio
    async def test_continuous_yields_on_yield_back(self, mock_chatbot):
        """Continuous role calls yield_back — returns FINISHED."""
        continuous_role = Role(name="continuous", description="Continuous agent", behavior_policy="continuous")
        chat_history = ChatHistory()
        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Hello")]
        ))
        tool_manager = ToolManager()

        def _yield_back():
            pass

        tool_manager.register_tool(func=_yield_back, name="yield_back", description="Signal that you have finished your task and want to yield control back to the user/channel. Call this when you've completed all your work and no longer need to execute tools.")

        class MockResponse:
            @property
            def data(self):
                return {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "y1", "name": "yield_back", "arguments": "{}"}],
                }

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        async def mock_send(history, streaming=True):
            return MockResponse()

        mock_chatbot.send_message = mock_send

        pending_record = MagicMock()
        pending_record.tool_call = {"name": "yield_back", "id": "y1", "arguments": "{}"}
        pending_record.approval_status = "approved"
        mock_session = _make_mock_session(chat_history, tool_manager, has_unfinished=False)
        mock_session._pending_tool_calls = [pending_record]
        mock_session.has_reviewed_tool_call.return_value = True

        env = REPLExecutionEnvironment(chat_history, tool_manager, continuous_role)
        status, _ = await env.step(mock_session)

        assert status == ExecStatus.FINISHED
        # Chat history should contain user message + assistant tool_use + tool_result
        assert len(chat_history.messages) == 3
        assert chat_history.messages[2].get_role() == "tool_result"