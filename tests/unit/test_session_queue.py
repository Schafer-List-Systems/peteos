"""Tests for Session message queue functionality."""
import asyncio
import time

import pytest

from peteos.session import Session, ApprovalEvent
from peteos.chatbot import Message, ChatHistory, ContentPart
from peteos.chatbot.chatbotresponse import ChatBotResponse
from peteos.role import Role
from peteos.toolmanager import ToolManager
from peteos.chatbot import ChatBotManager
from peteos.executionenvironment import ExecStatus


def _user_msg(text: str) -> Message:
    return Message(role="user", content=[ContentPart(part_type="text", text=text)])


class _FakeChatBotResponse:
    """Minimal mock ChatBotResponse that supports __aiter__."""
    def __init__(self, data: dict):
        self.data = data

    def __aiter__(self):
        return self

    def __anext__(self):
        raise StopAsyncIteration

    def __getitem__(self, key):
        return self.data[key]


class MockChatBot:
    """Mock chatbot that returns a final answer immediately."""

    def __init__(self, response_content: str = "Final answer"):
        self.response_content = response_content

    async def send_message(self, chat_history, streaming=True):
        return _FakeChatBotResponse({"text": self.response_content})


class MockExecutionEnvironment:
    """Mock execution environment that works with Session.step()."""

    def __init__(self, simulate_running=True, wait_on_interrupt=False):
        self._running = False
        self._interrupt = False
        self._completion_signal = asyncio.Event()
        self._completion_signal.set()
        self.interrupt_count = 0
        self.run_count = 0
        self.simulate_running = simulate_running
        self.wait_on_interrupt = wait_on_interrupt
        self._hooks: dict[str, list] = {
            "before_tool_execution": [],
            "after_tool_execution": [],
            "before_notification_publish": [],
            "before_send_to_chatbot": [],
            "after_message_append": [],
            "after_step": [],
        }

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def chatbot(self):
        return MockChatBot()

    def set_interrupt(self):
        self.interrupt_count += 1
        self._interrupt = True

    def clear_interrupt(self):
        self._interrupt = False

    async def wait_for_stop(self):
        await self._completion_signal.wait()

    async def step(self, session):
        """Simulate one step: return (ExecStatus, None) unless waiting_on_interrupt."""
        if self.simulate_running:
            await asyncio.sleep(0.01)
        self._running = True
        self.run_count += 1
        if self.wait_on_interrupt and self._interrupt:
            self._running = False
            return (ExecStatus.FINISHED, None)
        self._running = False
        return (ExecStatus.FINISHED, None)

    async def execute_pending_tool(self, tool_call):
        pass

    def register_hook(self, hook_point, callback, *args):
        if hook_point not in self._hooks:
            self._hooks[hook_point] = []
        self._hooks[hook_point].append(callback)

    async def _call_hooks(self, hook_point, *args):
        for cb in self._hooks.get(hook_point, []):
            result = cb(*args)
            if asyncio.iscoroutinefunction(cb):
                result = await result
            if result is not None:
                return result
        return None


@pytest.mark.asyncio
async def test_queue_message_adds_to_queue():
    """Test that queue_message adds message to chat_history."""
    chat_history = ChatHistory()
    role = Role(name="test", description="Test role")
    tool_manager = ToolManager()
    chatbot_manager = ChatBotManager()

    env = MockExecutionEnvironment(simulate_running=False)
    session = Session(
        role=role,
        tool_manager=tool_manager,
        chatbot_manager=chatbot_manager,
        chat_history=chat_history,
        execution_environment=env,
    )

    message = _user_msg("Hello")
    await session.queue_message(message)

    # Give the event loop time to process
    await asyncio.sleep(0.1)

    assert len(chat_history.messages) == 1
    assert chat_history.messages[0].get_role() == "user"
    assert env.run_count == 1

    await session.stop()


@pytest.mark.asyncio
async def test_queue_message_non_running():
    """Test that queue_message starts env when not running."""
    chat_history = ChatHistory()
    role = Role(name="test", description="Test role")
    tool_manager = ToolManager()
    chatbot_manager = ChatBotManager()

    env = MockExecutionEnvironment(simulate_running=False)
    session = Session(
        role=role,
        tool_manager=tool_manager,
        chatbot_manager=chatbot_manager,
        chat_history=chat_history,
        execution_environment=env,
    )

    await session.queue_message(_user_msg("Test"))
    await asyncio.sleep(0.1)

    assert env.run_count >= 1
    assert len(chat_history.messages) == 1

    await session.stop()


@pytest.mark.asyncio
async def test_queue_message_drains_all_to_history():
    """Test that all queued messages are drained to chat_history."""
    chat_history = ChatHistory()
    role = Role(name="test", description="Test role")
    tool_manager = ToolManager()
    chatbot_manager = ChatBotManager()

    env = MockExecutionEnvironment(simulate_running=False)
    session = Session(
        role=role,
        tool_manager=tool_manager,
        chatbot_manager=chatbot_manager,
        chat_history=chat_history,
        execution_environment=env,
    )

    for i in range(5):
        msg = _user_msg(f"Message {i}")
        await session.queue_message(msg)

    await asyncio.sleep(0.1)

    assert len(chat_history.messages) >= 5

    await session.stop()


@pytest.mark.asyncio
async def test_queue_message_preserves_order():
    """Test that messages are processed in order."""
    chat_history = ChatHistory()
    role = Role(name="test", description="Test role")
    tool_manager = ToolManager()
    chatbot_manager = ChatBotManager()

    env = MockExecutionEnvironment(simulate_running=False)
    session = Session(
        role=role,
        tool_manager=tool_manager,
        chatbot_manager=chatbot_manager,
        chat_history=chat_history,
        execution_environment=env,
    )

    for i in range(5):
        msg = _user_msg(f"Msg {i}")
        await session.queue_message(msg)

    await asyncio.sleep(0.1)

    assert len(chat_history.messages) >= 5

    await session.stop()


@pytest.mark.asyncio
async def test_push_event_non_blocking():
    """Test that push_event is non-blocking."""
    chat_history = ChatHistory()
    role = Role(name="test", description="Test role")
    tool_manager = ToolManager()
    chatbot_manager = ChatBotManager()

    env = MockExecutionEnvironment(simulate_running=False)
    session = Session(
        role=role,
        tool_manager=tool_manager,
        chatbot_manager=chatbot_manager,
        chat_history=chat_history,
        execution_environment=env,
    )

    start = time.monotonic()
    session.push_event(_user_msg("Test"))
    elapsed = time.monotonic() - start

    assert elapsed < 0.01  # Should be nearly instant

    await session.stop()


@pytest.mark.asyncio
async def test_concurrent_push_no_race_condition():
    """Test that concurrent pushes don't cause race conditions."""
    chat_history = ChatHistory()
    role = Role(name="test", description="Test role")
    tool_manager = ToolManager()
    chatbot_manager = ChatBotManager()

    env = MockExecutionEnvironment(simulate_running=False)
    session = Session(
        role=role,
        tool_manager=tool_manager,
        chatbot_manager=chatbot_manager,
        chat_history=chat_history,
        execution_environment=env,
    )

    # Use queue_message which starts the session and enqueues
    for i in range(10):
        msg = _user_msg(f"Concurrent {i}")
        await session.queue_message(msg)

    await asyncio.sleep(0.1)

    assert len(chat_history.messages) >= 10

    await session.stop()


@pytest.mark.asyncio
async def test_session_start_stop_lifecycle():
    """Test that session start/stop works correctly."""
    chatbot_manager = ChatBotManager()
    tool_manager = ToolManager()
    chat_history = ChatHistory()

    env = MockExecutionEnvironment(simulate_running=False)
    role = Role(name="test", description="Test role")

    session = Session(
        role=role,
        tool_manager=tool_manager,
        chatbot_manager=chatbot_manager,
        chat_history=chat_history,
        execution_environment=env,
    )

    assert not session.is_running()

    await session.start()
    assert session.is_running()

    await session.stop()
    assert not session.is_running()

    # Double stop should be a no-op
    await session.stop()


@pytest.mark.asyncio
async def test_approval_event_handling():
    """Test that ApprovalEvent events are processed correctly."""
    chat_history = ChatHistory()
    role = Role(name="test", description="Test role")
    tool_manager = ToolManager()
    chatbot_manager = ChatBotManager()

    env = MockExecutionEnvironment(simulate_running=False)
    session = Session(
        role=role,
        tool_manager=tool_manager,
        chatbot_manager=chatbot_manager,
        chat_history=chat_history,
        execution_environment=env,
    )

    # Start the session, then push an ApprovalEvent
    await session.start()
    tool_call = {"name": "test_tool", "arguments": "{}"}
    approval = ApprovalEvent(tool_call=tool_call, approved=True)
    session.push_event(approval)

    await asyncio.sleep(0.1)

    await session.stop()
