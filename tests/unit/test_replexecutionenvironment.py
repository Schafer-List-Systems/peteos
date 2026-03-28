"""Unit tests for REPLExecutionEnvironment."""

import asyncio
from unittest.mock import MagicMock

import pytest

from peteos.chatbot import ChatBot
from peteos.chathistory import ChatHistory
from peteos.message import Message
from peteos.replexecutionenvironment import REPLExecutionEnvironment
from peteos.toolmanager import ToolManager


class TestREPLExecutionEnvironmentInit:
    """Test REPLExecutionEnvironment initialization."""

    def test_init(self):
        """Test basic initialization."""
        chatbot = MagicMock(spec=ChatBot)
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        env = REPLExecutionEnvironment(chatbot, chat_history, tool_manager)

        assert env.chatbot == chatbot
        assert env.chat_history == chat_history
        assert env.tool_manager == tool_manager
        assert env._interrupt is False

    def test_interset_flag_default_false(self):
        """Test interrupt flag is False by default."""
        chatbot = MagicMock(spec=ChatBot)
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        env = REPLExecutionEnvironment(chatbot, chat_history, tool_manager)

        assert env._interrupt is False


class TestSetInterruptClearInterrupt:
    """Test interrupt flag methods."""

    def test_set_interrupt(self):
        """Test set_interrupt sets flag to True."""
        chatbot = MagicMock(spec=ChatBot)
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        env = REPLExecutionEnvironment(chatbot, chat_history, tool_manager)

        env.set_interrupt()

        assert env._interrupt is True

    def test_clear_interrupt(self):
        """Test clear_interrupt sets flag to False."""
        chatbot = MagicMock(spec=ChatBot)
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        env = REPLExecutionEnvironment(chatbot, chat_history, tool_manager)
        env.set_interrupt()

        env.clear_interrupt()

        assert env._interrupt is False


class TestREPLRunBasicConversation:
    """Test basic REPL loop without tool calls."""

    @pytest.mark.asyncio
    async def test_run_basic_conversation(self):
        """Test single turn conversation with final answer."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        chat_history.append_message(
            Message(content={"role": "user", "content": "Hello!"})
        )

        class MockResponse:
            def __init__(self):
                self.thinking_content = ""
                self.text_content = "Hello! How can I help?"

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        call_count = [0]

        async def mock_send_message(history, streaming=True):
            call_count[0] += 1
            return MockResponse()

        chatbot = MagicMock(spec=ChatBot)
        chatbot.send_message = mock_send_message

        env = REPLExecutionEnvironment(chatbot, chat_history, tool_manager)

        await env.run()

        # Should have 2 messages: user, assistant answer
        assert len(chat_history.messages) == 2
        assert chat_history.messages[0].content["role"] == "user"
        assert chat_history.messages[1].content["role"] == "assistant"
        assert chat_history.messages[1].content["content"] == "Hello! How can I help?"
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_run_with_thinking_content(self):
        """Test response with thinking content."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        chat_history.append_message(
            Message(content={"role": "user", "content": "Solve 2+2"})
        )

        thinking_content = "Let me calculate this step by step..."
        text_content = "4"

        class MockResponse:
            def __init__(self):
                self.thinking_content = thinking_content
                self.text_content = text_content

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        async def mock_send_message(history, streaming=True):
            return MockResponse()

        chatbot = MagicMock(spec=ChatBot)
        chatbot.send_message = mock_send_message

        env = REPLExecutionEnvironment(chatbot, chat_history, tool_manager)

        await env.run()

        # Should have: user, assistant (thinking), assistant (answer)
        assert len(chat_history.messages) == 3
        assert chat_history.messages[0].content["role"] == "user"
        assert chat_history.messages[1].content["role"] == "assistant"
        assert "[Thinking]" in chat_history.messages[1].content["content"]
        assert chat_history.messages[1].content["content"].endswith(thinking_content)
        assert chat_history.messages[2].content["content"] == text_content

    @pytest.mark.asyncio
    async def test_run_loop_terminates_on_final_answer(self):
        """Test that loop terminates when no tool calls in response."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        chat_history.append_message(
            Message(content={"role": "user", "content": "Hello!"})
        )

        call_count = [0]

        class MockResponse:
            def __init__(self):
                self.thinking_content = ""
                self.text_content = "Final answer"

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        async def mock_send_message(history, streaming=True):
            call_count[0] += 1
            return MockResponse()

        chatbot = MagicMock(spec=ChatBot)
        chatbot.send_message = mock_send_message

        env = REPLExecutionEnvironment(chatbot, chat_history, tool_manager)

        await env.run()

        # Should only call send_message once (loop terminates on final answer)
        assert call_count[0] == 1


class TestREPLRunWithToolCalls:
    """Test REPL loop with tool calls."""

    @pytest.mark.asyncio
    async def test_run_tool_call_detected(self):
        """Test that tool calls are correctly parsed from response."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        # Register a test tool
        def get_weather(city: str) -> str:
            return f"Weather in {city}: sunny"

        tool_manager.register_tool(func=get_weather)

        chat_history.append_message(
            Message(content={"role": "user", "content": "What's the weather?"})
        )

        # Tool call as direct JSON in text_content
        tool_call_json = '{"name": "get_weather", "arguments": {"city": "London"}}'
        final_answer = "It's sunny in London."

        class MockResponse:
            def __init__(self, text_content):
                self.thinking_content = ""
                self.text_content = text_content

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        response_order = [tool_call_json, final_answer]
        response_idx = [0]

        async def mock_send_message(history, streaming=True):
            idx = response_idx[0]
            response_idx[0] += 1
            if idx < len(response_order):
                return MockResponse(response_order[idx])
            else:
                return MockResponse(final_answer)

        chatbot = MagicMock(spec=ChatBot)
        chatbot.send_message = mock_send_message

        env = REPLExecutionEnvironment(chatbot, chat_history, tool_manager)

        await env.run()

        # Should have: user, assistant (tool call), tool (result), assistant (final)
        # send_message called twice: tool call then final answer
        assert len(chat_history.messages) == 4
        assert chat_history.messages[0].content["role"] == "user"
        assert chat_history.messages[1].content["role"] == "assistant"
        assert "get_weather" in chat_history.messages[1].content["content"]
        assert chat_history.messages[2].content["role"] == "tool"
        assert chat_history.messages[2].content["name"] == "get_weather"
        assert chat_history.messages[3].content["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_run_tool_call_loop_continues(self):
        """Test that loop continues after tool call with new response."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        def add(a: int, b: int) -> int:
            return a + b

        tool_manager.register_tool(func=add)

        chat_history.append_message(
            Message(content={"role": "user", "content": "Add 2+2"})
        )

        class MockResponse:
            def __init__(self, text_content):
                self.text_content = text_content
                self.thinking_content = ""

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        responses = [
            '{"name": "add", "arguments": {"a": 2, "b": 2}}',  # First: tool call
            "The answer is 4",  # Second: final answer
        ]
        response_idx = [0]

        async def mock_send_message(history, streaming=True):
            idx = response_idx[0]
            response_idx[0] += 1
            if idx < len(responses):
                return MockResponse(responses[idx])
            else:
                return MockResponse(responses[-1])  # Default to last

        chatbot = MagicMock(spec=ChatBot)
        chatbot.send_message = mock_send_message

        env = REPLExecutionEnvironment(chatbot, chat_history, tool_manager)

        await env.run()

        # Should call send_message twice: once for tool, once for final answer
        assert response_idx[0] == 2
        # user, assistant (tool), tool (result), assistant (answer)
        assert len(chat_history.messages) == 4


class TestREPLRunInterrupt:
    """Test interrupt functionality."""

    @pytest.mark.asyncio
    async def test_run_interrupt_immediately(self):
        """Test that interrupt stops the loop before processing."""
        chat_history = ChatHistory()
        tool_manager = ToolManager()

        chat_history.append_message(
            Message(content={"role": "user", "content": "Hello!"})
        )

        class MockResponse:
            def __init__(self):
                self.thinking_content = ""
                self.text_content = "Final answer"

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        async def mock_send_message(history, streaming=True):
            return MockResponse()

        chatbot = MagicMock(spec=ChatBot)
        chatbot.send_message = mock_send_message

        env = REPLExecutionEnvironment(chatbot, chat_history, tool_manager)

        # Set interrupt immediately
        env.set_interrupt()

        await env.run()

        # Should exit immediately without processing
        assert env._interrupt is True


class TestParseToolCalls:
    """Test _parse_tool_calls method."""

    def setup_method(self):
        """Set up test fixtures."""
        chatbot = MagicMock(spec=ChatBot)
        chat_history = ChatHistory()
        tool_manager = ToolManager()
        self.env = REPLExecutionEnvironment(chatbot, chat_history, tool_manager)

    def test_parse_tool_calls_direct_json(self):
        """Test parsing direct JSON tool call."""
        content = '{"name": "get_weather", "arguments": {"city": "London"}}'

        tool_calls = self.env._parse_tool_calls(content)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[0]["arguments"]["city"] == "London"

    def test_parse_tool_calls_with_tool_calls_array(self):
        """Test parsing tool_calls array format."""
        content = '{"tool_calls": [{"name": "get_weather", "arguments": {"city": "London"}}]}'

        tool_calls = self.env._parse_tool_calls(content)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"

    def test_parse_tool_calls_list_format(self):
        """Test parsing list of tool calls."""
        content = '[{"name": "tool1", "arguments": {}}, {"name": "tool2", "arguments": {}}]'

        tool_calls = self.env._parse_tool_calls(content)

        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "tool1"
        assert tool_calls[1]["name"] == "tool2"

    def test_parse_tool_calls_embedded_in_text(self):
        """Test parsing JSON embedded in text response."""
        content = 'Here is the tool: {"name": "add", "arguments": {"a": 1, "b": 2}} more text'

        tool_calls = self.env._parse_tool_calls(content)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "add"
        assert tool_calls[0]["arguments"]["a"] == 1
        assert tool_calls[0]["arguments"]["b"] == 2

    def test_parse_tool_calls_nested_braces(self):
        """Test parsing JSON with nested objects."""
        content = '{"name": "search", "arguments": {"query": "weather", "location": {"city": "NYC"}}}'

        tool_calls = self.env._parse_tool_calls(content)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "search"
        assert tool_calls[0]["arguments"]["location"]["city"] == "NYC"

    def test_parse_tool_calls_no_tool_calls(self):
        """Test parsing response without tool calls."""
        content = "This is just a normal response without any tool calls."

        tool_calls = self.env._parse_tool_calls(content)

        assert tool_calls == []

    def test_parse_tool_calls_invalid_json(self):
        """Test parsing invalid JSON."""
        content = "not valid json"

        tool_calls = self.env._parse_tool_calls(content)

        assert tool_calls == []

    def test_parse_tool_calls_empty_string(self):
        """Test parsing empty string."""
        content = ""

        tool_calls = self.env._parse_tool_calls(content)

        assert tool_calls == []
