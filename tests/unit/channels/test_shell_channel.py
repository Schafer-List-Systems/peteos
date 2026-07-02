"""Unit tests for InteractiveShellChannel."""

import asyncio
import uuid

import pytest
from unittest.mock import MagicMock, AsyncMock

from peteos.engine.channel import Channel
from peteos.channels import InteractiveShellChannel
from peteos.chatbot import Message, ContentPart
from peteos.engine.runner import Runner


@pytest.fixture(autouse=True)
def _setup_mock_chatbot():
    """Set up a mock ChatBot in the class-level ChatBotManager for tests."""
    from peteos.chatbot.manager import ChatBotManager
    ChatBotManager._backends = {"test-backend": MagicMock(models={"test_model": MagicMock()})}
    yield
    ChatBotManager._backends.clear()


def _cleanup_channels():
    """Clean up channel registry and mock runner call tracking."""
    for name in list(Channel._registry.keys()):
        Channel._registry.pop(name)


def _make_mock_runner():
    """Create a mock runner suitable for shell channel tests."""
    runner = MagicMock(spec=Runner)
    runner.role = MagicMock()
    runner.role.name = "test"
    runner.subscribe = MagicMock(return_value=True)
    runner._execution_environment = MagicMock()
    runner._execution_environment.get_pending_tool_calls = MagicMock(return_value=[])
    runner.push_event = MagicMock()
    runner.uuid = MagicMock()
    runner.uuid.hex = "abcd1234"
    return runner


def _reset_mock_runner(runner):
    """Reset mock call tracking to prevent reference leaks."""
    runner.reset_mock()
    runner.role.reset_mock()


class TestShellChannelInit:
    """Test InteractiveShellChannel initialization."""

    def setup_method(self):
        _cleanup_channels()

    def teardown_method(self):
        _cleanup_channels()

    @pytest.mark.asyncio
    async def test_shell_channel_creation(self):
        """Test creating a shell channel."""
        runner = _make_mock_runner()
        shell_channel = InteractiveShellChannel("shell", runner)

        assert shell_channel.name == "shell"
        assert shell_channel._running is False
        assert shell_channel._runner is runner

        shell_channel._running = False

    def test_shell_channel_subscribes_to_runner(self):
        """Test shell channel subscribes to its runner on init."""
        runner = _make_mock_runner()
        channel = InteractiveShellChannel("shell", runner)

        runner.subscribe.assert_called_once_with(channel)


class TestShellChannelSend:
    """Test send() method."""

    def setup_method(self):
        _cleanup_channels()
        self.runner = _make_mock_runner()

    def teardown_method(self):
        _cleanup_channels()

    def test_send_string_prints(self):
        channel = InteractiveShellChannel("shell", self.runner)
        output = []

        def mock_print(msg):
            output.append(msg)

        import builtins
        original_print = builtins.print
        builtins.print = mock_print
        try:
            asyncio.run(channel.send("hello world"))
        finally:
            builtins.print = original_print

        assert output == ["hello world"]

    @pytest.mark.asyncio
    async def test_send_message_calls_printable(self):
        channel = InteractiveShellChannel("shell", self.runner)
        output = []

        class MockSend:
            def __init__(self, callback):
                self.callback = callback
            async def __call__(self, message, session_uuid=None):
                self.callback(message)

        channel.send = MockSend(lambda msg: output.append(msg))

        msg = Message.create(
            role="assistant",
            content_parts=[ContentPart.create_text("Hello")],
        )
        await channel.send(msg)
        assert len(output) == 1


class TestShellChannelPrompt:
    """Test _get_prompt()."""

    def setup_method(self):
        _cleanup_channels()
        self.runner = _make_mock_runner()

    def teardown_method(self):
        _cleanup_channels()

    def test_prompt_shows_runner_uuid(self):
        channel = InteractiveShellChannel("shell", self.runner)
        prompt = channel._get_prompt()
        assert "@test" in prompt
        assert ">> " in prompt


class TestShellChannelCommands:
    """Test shell channel command handling."""

    def setup_method(self):
        _cleanup_channels()
        self.runner = _make_mock_runner()

    def teardown_method(self):
        _cleanup_channels()

    def test_command_quit(self):
        channel = InteractiveShellChannel("shell", self.runner)
        should_continue, output = channel.handle_command("/quit")

        assert should_continue is False
        assert "Goodbye" in output
        assert channel._running is False

    def test_command_unknown(self):
        channel = InteractiveShellChannel("shell", self.runner)
        should_continue, output = channel.handle_command("/unknown")

        assert should_continue is True
        assert "Unknown command" in output

    def test_command_case_insensitive(self):
        channel = InteractiveShellChannel("shell", self.runner)
        should_continue, output = channel.handle_command("/QUIT")

        assert should_continue is False
        assert channel._running is False


class TestShellChannelApprovalCommands:
    """Test /approve, /deny, /pending commands."""

    def setup_method(self):
        _cleanup_channels()
        self.runner = _make_mock_runner()
        self.channel = InteractiveShellChannel("shell", self.runner)

    def teardown_method(self):
        _cleanup_channels()

    def test_approve_no_pending(self):
        self.runner._execution_environment.get_pending_tool_calls.return_value = []
        should_continue, output = self.channel.handle_command("/approve")
        assert should_continue is True
        assert "No pending tool calls" in output

    def test_deny_no_pending(self):
        self.runner._execution_environment.get_pending_tool_calls.return_value = []
        should_continue, output = self.channel.handle_command("/deny")
        assert should_continue is True
        assert "No pending tool calls" in output

    def test_pending_shows_list(self):
        record = MagicMock()
        record.tool_call_id = "tc-1"
        record.tool_call.name = "test_tool"
        record.approval_status.value = "pending"
        self.runner._execution_environment.get_pending_tool_calls.return_value = [record]
        should_continue, output = self.channel.handle_command("/pending")
        assert should_continue is True
        assert "Pending tool calls:" in output
        assert "test_tool" in output

    def test_approve_sends_event(self):
        record = MagicMock()
        record.tool_call_id = "tc-1"
        record.tool_call.name = "test_tool"
        record.approval_status.value = "pending"
        self.runner._execution_environment.get_pending_tool_calls.return_value = [record]
        should_continue, output = self.channel.handle_command("/approve")
        assert should_continue is True
        assert "Approved" in output
        self.runner.push_event.assert_called_once()

    def test_deny_sends_event(self):
        record = MagicMock()
        record.tool_call_id = "tc-1"
        record.tool_call.name = "test_tool"
        record.approval_status.value = "pending"
        self.runner._execution_environment.get_pending_tool_calls.return_value = [record]
        should_continue, output = self.channel.handle_command("/deny")
        assert should_continue is True
        assert "Denied" in output
        self.runner.push_event.assert_called_once()


class TestShellChannelRun:
    """Test shell channel run loop."""

    def setup_method(self):
        _cleanup_channels()
        self.runner = _make_mock_runner()

    def teardown_method(self):
        _cleanup_channels()

    def test_run_displays_welcome(self):
        channel = InteractiveShellChannel("shell", self.runner)
        output = []

        def mock_send(msg):
            output.append(msg)

        channel.send = mock_send

        asyncio.run(channel.start())

        assert any("Connected" in str(msg) or "Commands" in str(msg) for msg in output)

        asyncio.run(channel.stop())

    def test_run_handles_quit(self):
        channel = InteractiveShellChannel("shell", self.runner)
        channel._running = True

        should_continue, _ = channel.handle_command("/quit")
        assert should_continue is False
        assert channel._running is False
