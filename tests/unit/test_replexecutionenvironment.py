"""Unit tests for REPLExecutionEnvironment."""

import asyncio
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


@pytest.fixture
def mock_role():
    """Create a mock Role with default model regex."""
    return Role(name="test", description="Test role")


@pytest.fixture
def mock_chatbot_manager():
    """Create a mock ChatBotManager with a default ChatBot."""
    manager = MagicMock(spec=ChatBotManager)
    mock_chatbot = MagicMock(spec=ChatBot)
    manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]
    return manager


@pytest.fixture
def mock_chatbot():
    """Create a mock ChatBot for direct assignment."""
    return MagicMock(spec=ChatBot)


class TestREPLExecutionEnvironmentInit:
    """Test REPLExecutionEnvironment initialization."""

    def test_init(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test basic initialization."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)

        assert env.chatbot == mock_chatbot
        assert env.chat_history == chat_history
        assert env.tool_manager == tool_manager
        assert env.role == mock_role
        assert env._interrupt is False

    def test_interset_flag_default_false(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test interrupt flag is False by default."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)

        assert env._interrupt is False


class TestSetInterruptClearInterrupt:
    """Test interrupt flag methods."""

    def test_set_interrupt(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test set_interrupt sets flag to True."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)

        env.set_interrupt()

        assert env._interrupt is True

    def test_clear_interrupt(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test clear_interrupt sets flag to False."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)
        env.set_interrupt()

        env.clear_interrupt()

        assert env._interrupt is False


class TestREPLRunBasicConversation:
    """Test basic REPL loop without tool calls."""

    @pytest.mark.asyncio
    async def test_run_basic_conversation(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test single turn conversation with final answer."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Hello!")]
        ))

        class MockResponse:
            def __init__(self):
                self._data = {"role": "assistant", "text": "Hello! How can I help?", "reasoning": ""}

            @property
            def data(self):
                return self._data

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        call_count = [0]

        async def mock_send_message(history, streaming=True):
            call_count[0] += 1
            return MockResponse()

        mock_chatbot.send_message = mock_send_message
        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)

        await env.run()

        # Should have 2 messages: user, assistant answer
        assert len(chat_history.messages) == 2
        assert chat_history.messages[0].content[0].type == "text"
        assert chat_history.messages[0].content[0].text == "Hello!"
        assert chat_history.messages[1].content[0].type == "text"
        assert chat_history.messages[1].content[0].text == "Hello! How can I help?"
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_run_with_thinking_content(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test response with reasoning content."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Solve 2+2")]
        ))

        reasoning_content = "Let me calculate this step by step..."
        text_content = "4"

        class MockResponse:
            def __init__(self):
                self._data = {"role": "assistant", "reasoning": reasoning_content, "text": text_content}

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
        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)

        await env.run()

        # Should have: user, assistant (with reasoning and text in one message)
        assert len(chat_history.messages) == 2
        assert chat_history.messages[0].content[0].type == "text"
        # Reasoning is added first, text second in content_parts order
        assert chat_history.messages[1].content[0].type == "reasoning"
        assert chat_history.messages[1].content[0].data.get("reasoning") == reasoning_content
        assert chat_history.messages[1].content[1].type == "text"
        assert chat_history.messages[1].content[1].text == text_content

    @pytest.mark.asyncio
    async def test_run_loop_terminates_on_final_answer(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test that loop terminates when no tool_calls in response."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Hello!")]
        ))

        call_count = [0]

        class MockResponse:
            def __init__(self):
                self._data = {"role": "assistant", "text": "Final answer", "reasoning": ""}

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
        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)

        await env.run()

        # Should only call send_message once (loop terminates on final answer)
        assert call_count[0] == 1


class TestREPLRunWithToolCalls:
    """Test REPL loop with tool calls."""

    @pytest.mark.asyncio
    async def test_run_tool_call_detected(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test that tool calls in response.data['tool_calls'] are executed."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        # Register a test tool
        def get_weather(city: str) -> str:
            return f"Weather in {city}: sunny"

        tool_manager.register_tool(func=get_weather)

        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="What's the weather?")]
        ))

        final_answer = "It's sunny in London."

        call_count = [0]

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

        # First call: tool_calls in content array, second call: final answer
        # Arguments must be a JSON string (as accumulated during streaming)
        responses = [
            {"role": "assistant", "text": "", "reasoning": "", "content": [{"type": "tool_use", "name": "get_weather", "arguments": "{\"city\":\"London\"}"}]},
            {"role": "assistant", "text": final_answer, "reasoning": ""}
        ]
        response_idx = [0]

        async def mock_send_message(history, streaming=True):
            idx = response_idx[0]
            response_idx[0] += 1
            return MockResponse(responses[idx])

        mock_chatbot.send_message = mock_send_message
        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)

        await env.run()

        # Should have: user, assistant (with tool_calls), tool (result), assistant (final answer)
        assert len(chat_history.messages) == 4
        # Message 0 is user
        assert chat_history.messages[0].get_role() == "user"
        # Message 1 is assistant response with tool_calls in content array
        assert chat_history.messages[1].get_role() == "assistant"
        # Tool calls are in content array with type="tool_use"
        tool_calls = [
            item for item in chat_history.messages[1].content
            if isinstance(item, ContentPart) and item.type == "tool_calls"
        ]
        assert len(tool_calls) == 1
        assert tool_calls[0].data["tool_calls"][0]["name"] == "get_weather"
        # Message 2 is tool result
        assert chat_history.messages[2].get_role() == "tool_result"
        assert chat_history.messages[2].content[0].type == "tool_result"
        assert chat_history.messages[2].content[0].data["name"] == "get_weather"
        # Message 3 is final answer
        assert chat_history.messages[3].get_role() == "assistant"
        assert chat_history.messages[3].content[0].text == final_answer

    @pytest.mark.asyncio
    async def test_run_tool_call_loop_continues(self, mock_role, mock_chatbot_manager, mock_chatbot):
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

        call_count = [0]

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
            {"role": "assistant", "text": "", "reasoning": "", "content": [{"type": "tool_use", "name": "add", "arguments": "{\"a\":2,\"b\":2}"}]},  # First: tool call
            {"role": "assistant", "text": "The answer is 4", "reasoning": ""}  # Second: final answer
        ]
        response_idx = [0]

        async def mock_send_message(history, streaming=True):
            idx = response_idx[0]
            response_idx[0] += 1
            return MockResponse(responses[idx])

        mock_chatbot.send_message = mock_send_message
        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)

        await env.run()

        # Should call send_message twice: once for tool, once for final answer
        assert response_idx[0] == 2
        # user, assistant (tool), tool (result), assistant (answer)
        assert len(chat_history.messages) == 4
        # Verify assistant response has tool_calls in content array
        assert chat_history.messages[1].get_role() == "assistant"
        tool_calls = [
            item for item in chat_history.messages[1].content
            if isinstance(item, ContentPart) and item.type == "tool_calls"
        ]
        assert len(tool_calls) == 1
        assert tool_calls[0].data["tool_calls"][0]["name"] == "add"


class TestREPLRunInterrupt:
    """Test interrupt functionality."""

    @pytest.mark.asyncio
    async def test_run_interrupt_immediately(self, mock_role, mock_chatbot_manager, mock_chatbot):
        """Test that interrupt stops the loop before processing."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        chat_history.append_message(Message(
            role="user",
            content=[ContentPart(part_type="text", text="Hello!")]
        ))

        class MockResponse:
            def __init__(self):
                self._data = {"role": "assistant", "text": "Final answer", "reasoning": ""}

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
        mock_chatbot_manager.list_chatbots.return_value = [("mock_model", mock_chatbot)]

        env = REPLExecutionEnvironment(mock_chatbot_manager, chat_history, tool_manager, mock_role)

        # Set interrupt immediately
        env.set_interrupt()

        await env.run()

        # Should exit immediately without processing
        assert env._interrupt is True


