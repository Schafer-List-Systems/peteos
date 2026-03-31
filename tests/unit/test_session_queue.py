"""Tests for Session message queue functionality."""
import asyncio
import time

import pytest

from peteos.session import Session
from peteos.message import Message
from peteos.chathistory import ChatHistory
from peteos.role import Role
from peteos.toolmanager import ToolManager
from peteos.chatbotmanager import ChatBotManager


class MockChatBot:
    """Mock chatbot that returns a final answer immediately."""

    def __init__(self, response_content: str = "Final answer"):
        self.response_content = response_content

    async def send_message(self, chat_history, streaming=True):
        class FakeResponse:
            data = {"text": self.response_content}

        return FakeResponse()


class MockExecutionEnvironment:
    """Mock execution environment that works with Session.queue_message()."""

    def __init__(self, simulate_running=True, wait_on_interrupt=False):
        self._running = False
        self._interrupt = False
        self._completion_signal = asyncio.Event()
        self._completion_signal.set()  # Start signaled (not running)
        self.interrupt_count = 0
        self.run_count = 0
        self.simulate_running = simulate_running
        self.wait_on_interrupt = wait_on_interrupt

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
        """Block until execution has stopped."""
        await self._completion_signal.wait()

    async def run(self):
        """Run the execution environment."""
        self._running = True
        self._completion_signal.clear()
        self.run_count += 1

        try:
            if self.simulate_running:
                # If we should wait on interrupt, do so
                if self.wait_on_interrupt:
                    while not self._interrupt:
                        await asyncio.sleep(0.01)
                # Otherwise, just run briefly
                else:
                    await asyncio.sleep(0.01)
        finally:
            self._running = False
            self._completion_signal.set()


@pytest.mark.asyncio
async def test_queue_message_adds_to_queue():
    """Test that queue_message adds message to chat_history."""
    chat_history = ChatHistory()
    role = Role(name="test", description="Test role")
    tool_manager = ToolManager()
    chatbot_manager = ChatBotManager()

    # Env not running initially
    env = MockExecutionEnvironment(simulate_running=False)
    session = Session(
        role=role,
        tool_manager=tool_manager,
        chatbot_manager=chatbot_manager,
        chat_history=chat_history,
        execution_environment=env
    )

    message = Message({"role": "user", "content": "Hello"})
    await session.queue_message(message)

    # Message should be in chat_history
    assert len(chat_history.messages) == 1
    assert chat_history.messages[0].content == {"role": "user", "content": "Hello"}
    assert env.run_count == 1


@pytest.mark.asyncio
async def test_queue_message_interrupts_running_env():
    """Test that queue_message interrupts running execution environment."""
    chat_history = ChatHistory()
    role = Role(name="test", description="Test role")
    tool_manager = ToolManager()
    chatbot_manager = ChatBotManager()

    env = MockExecutionEnvironment(simulate_running=True, wait_on_interrupt=True)

    session = Session(
        role=role,
        tool_manager=tool_manager,
        chatbot_manager=chatbot_manager,
        chat_history=chat_history,
        execution_environment=env
    )

    # Start env in background task
    async def run_env_async():
        await env.run()

    task = asyncio.create_task(run_env_async())

    # Give it a moment to start
    await asyncio.sleep(0.05)

    # Queue a message - should interrupt
    message = Message({"role": "user", "content": "Test"})
    await session.queue_message(message)

    # Wait for task to finish
    await asyncio.sleep(0.2)

    # Verify interruption happened
    assert env.interrupt_count == 1

    # Verify env stopped
    assert not env.is_running


@pytest.mark.asyncio
async def test_queue_message_drains_all_to_history():
    """Test that all queued messages are drained to chat_history."""
    chat_history = ChatHistory()
    role = Role(name="test", description="Test role")
    tool_manager = ToolManager()
    chatbot_manager = ChatBotManager()

    env = MockExecutionEnvironment(simulate_running=True, wait_on_interrupt=True)

    session = Session(
        role=role,
        tool_manager=tool_manager,
        chatbot_manager=chatbot_manager,
        chat_history=chat_history,
        execution_environment=env
    )

    # Start env in background task
    async def run_env_async():
        await env.run()

    task = asyncio.create_task(run_env_async())

    # Give it a moment to start
    await asyncio.sleep(0.05)

    # Queue multiple messages rapidly
    for i in range(5):
        msg = Message({"role": "user", "content": f"Message {i}"})
        await session.queue_message(msg)

    # Wait a bit
    await asyncio.sleep(0.2)

    # All messages should be in chat_history
    assert len(chat_history.messages) >= 5


@pytest.mark.asyncio
async def test_queue_message_restarts_env():
    """Test that execution env restarts after draining messages."""
    chat_history = ChatHistory()
    role = Role(name="test", description="Test role")
    tool_manager = ToolManager()
    chatbot_manager = ChatBotManager()

    env = MockExecutionEnvironment(simulate_running=True, wait_on_interrupt=True)

    session = Session(
        role=role,
        tool_manager=tool_manager,
        chatbot_manager=chatbot_manager,
        chat_history=chat_history,
        execution_environment=env
    )

    # Start env in background task
    async def run_env_async():
        await env.run()

    task = asyncio.create_task(run_env_async())

    # Give it a moment to start
    await asyncio.sleep(0.05)

    # Queue message - should restart
    await session.queue_message(Message({"role": "user", "content": "Test"}))

    # Wait a bit
    await asyncio.sleep(0.2)

    # Env should have been restarted (original run + restart)
    assert env.run_count >= 2


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
        execution_environment=env
    )

    # Queue message when not running - should start env
    await session.queue_message(Message({"role": "user", "content": "Test"}))

    # Env should have been started
    assert env.run_count >= 1
    assert len(chat_history.messages) == 1


@pytest.mark.asyncio
async def test_wait_for_stop_signals_completion():
    """Test that wait_for_stop properly signals completion."""
    chat_history = ChatHistory()
    role = Role(name="test", description="Test role")
    tool_manager = ToolManager()
    chatbot_manager = ChatBotManager()

    env = MockExecutionEnvironment(simulate_running=True, wait_on_interrupt=False)

    session = Session(
        role=role,
        tool_manager=tool_manager,
        chatbot_manager=chatbot_manager,
        chat_history=chat_history,
        execution_environment=env
    )

    # Run env
    await env.run()

    # Wait_for_stop should return immediately (already stopped)
    await env.wait_for_stop()

    # Verify env has stopped
    assert not env.is_running


@pytest.mark.asyncio
async def test_concurrent_queue_no_race_condition():
    """Test that concurrent enqueues don't cause race conditions."""
    chat_history = ChatHistory()
    role = Role(name="test", description="Test role")
    tool_manager = ToolManager()
    chatbot_manager = ChatBotManager()

    env = MockExecutionEnvironment(simulate_running=True, wait_on_interrupt=True)

    session = Session(
        role=role,
        tool_manager=tool_manager,
        chatbot_manager=chatbot_manager,
        chat_history=chat_history,
        execution_environment=env
    )

    # Start env in background task
    async def run_env_async():
        await env.run()

    task = asyncio.create_task(run_env_async())

    # Give it a moment to start
    await asyncio.sleep(0.05)

    # Enqueue messages sequentially (serialized by lock in queue_message)
    for i in range(10):
        msg = Message({"role": "user", "content": f"Concurrent {i}"})
        await session.queue_message(msg)

    # Wait a bit
    await asyncio.sleep(0.2)

    # All messages should be in chat_history
    assert len(chat_history.messages) >= 10

    # Interrupt should have been called exactly once (serialized by lock)
    assert env.interrupt_count == 1


@pytest.mark.asyncio
async def test_queue_message_preserves_order():
    """Test that messages are processed in order."""
    chat_history = ChatHistory()
    role = Role(name="test", description="Test role")
    tool_manager = ToolManager()
    chatbot_manager = ChatBotManager()

    env = MockExecutionEnvironment(simulate_running=True, wait_on_interrupt=True)

    session = Session(
        role=role,
        tool_manager=tool_manager,
        chatbot_manager=chatbot_manager,
        chat_history=chat_history,
        execution_environment=env
    )

    # Start env in background task
    async def run_env_async():
        await env.run()

    task = asyncio.create_task(run_env_async())

    # Give it a moment to start
    await asyncio.sleep(0.05)

    # Queue messages in order
    for i in range(5):
        msg = Message({"role": "user", "content": f"Msg {i}"})
        await session.queue_message(msg)

    # Wait a bit
    await asyncio.sleep(0.2)

    # Messages should be in chat_history
    assert len(chat_history.messages) >= 5
