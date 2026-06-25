"""InteractiveShellChannel - REPL-style interaction with agents via queues."""

import asyncio
import sys
import uuid
from typing import Optional

from peteos.chatbot import Message, ContentPart
from peteos.engine.channel import Channel
from peteos.engine.executionenvironment import (
    ApprovalEvent,
    ToolApprovalStatus,
)
from peteos.utils import get_logger

_logger = get_logger(__name__)


class InteractiveShellChannel(Channel):
    """Interactive shell channel for REPL-style interaction with agents.

    This channel connects directly to a Runner:
    - User input (non-command lines) is posted to the Runner's message queue
    - Notifications are pushed to the channel's notification queue
    - Channel consumes notifications and displays them to user

    Session switching is the application's responsibility — the app
    creates a new runner and constructs a new shell channel for it.

    Example:
        >>> agent = Agent(role, tool_manager)
        >>> session = await agent.create_session()
        >>> runner = Runner(agent, session.uuid)
        >>> await runner.start()
        >>> shell = InteractiveShellChannel("shell", runner)
        >>> await shell.start()
        >>> # User types: Hello!
        >>> # Agent processes and notifications appear
        >>> await shell.stop()
        >>> await runner.stop()
    """

    def __init__(self, name: str, runner):
        """Initialize InteractiveShellChannel.

        Args:
            name: Unique identifier for this channel.
            runner: The Runner instance this channel connects to.
        """
        super().__init__(name, runner)
        self._input_task: asyncio.Task | None = None

    async def send(self, message: Message | str, session_uuid: uuid.UUID | None = None) -> None:
        """Send a message to the shell.

        Args:
            message: The Message to display, or a plain string.
            session_uuid: Unused for shell (message is printed regardless).
        """
        if isinstance(message, str):
            print(message)
        else:
            print(message.printable())

    def _post_message_to_runner(self, content: str) -> None:
        """Post a message directly to the runner's event queue.

        Non-blocking. The runner's event loop processes the message.
        Since this method is sync (called from the input loop), uses
        asyncio.create_task to call the async queue_message.

        Args:
            content: The message content from the user.
        """
        message = Message.create(
            role="user",
            content_parts=[ContentPart.create_text(content)]
        )
        self._runner.push_event(message)

    def _handle_file(self, args: str) -> tuple[bool, str]:
        """Handle /image or /file command."""
        parts = args.split(maxsplit=1)
        src = parts[0]
        text = parts[1] if len(parts) > 1 else ""

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._queue_file_message(src, text, "auto"))
                return (True, "")
            else:
                loop.run_until_complete(self._queue_file_message(src, text, "auto"))
                return (True, "")
        except (FileNotFoundError, RuntimeError) as e:
            return (True, f"Error: {e}")
        except Exception as e:
            return (True, f"Error: {type(e).__name__}: {e}")

    async def _queue_file_message(
        self, src: str, text: str, file_type: str = "auto"
    ) -> None:
        """Queue a file message to the current session."""
        from peteos.conversation.media import create_media_content_part_async

        file_part = await create_media_content_part_async(src)

        parts = [ContentPart.create_text(text)] if text else []
        parts.append(file_part)
        message = Message.create(role="user", content_parts=parts)
        self._runner.push_event(message)
        _logger.debug("%s message queued from %s", file_type, src)

    def _get_input_line(self) -> Optional[str]:
        """Synchronous input reader for use with run_in_executor."""
        try:
            line = sys.stdin.readline()
            return line if line else None
        except EOFError:
            return None

    async def start(self) -> None:
        """Start the shell channel.

        Starts the background notification consumption loop via super().start()
        and the input reading loop via asyncio.create_task().
        """
        await super().start()
        self._input_task = asyncio.create_task(self.read_input_loop())
        self._input_task.add_done_callback(self._on_input_task_done)

    @staticmethod
    def _on_input_task_done(task: asyncio.Task) -> None:
        """Suppress CancelledError from input task to avoid 'exception never retrieved' warnings."""
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            _logger.error("[InteractiveShellChannel]: input task finished with exception: %s: %r", type(e).__name__, e)

    async def stop(self) -> None:
        """Stop the shell channel gracefully."""
        if self._input_task is not None:
            self._input_task.cancel()
            self._input_task = None
        await super().stop()

    def _get_prompt(self) -> str:
        """Get the prompt string for the shell."""
        return f"{self._runner.uuid.hex[:8]} @{self._runner.role.name} >> "

    def handle_command(self, line: str) -> tuple[bool, str]:
        """Handle a shell command.

        Args:
            line: The command line (starting with /).

        Returns:
            Tuple of (should_continue: bool, output: str).
        """
        parts = line.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command == "/approve":
            pending = self._runner._execution_environment.get_pending_tool_calls()
            if not pending:
                return (True, "No pending tool calls")
            record = pending[0]
            approval_event = ApprovalEvent(
                tool_call_id=record.tool_call_id,
                tool_call=record.tool_call,
                approved=True,
            )
            self._runner.push_event(approval_event)
            return (True, f"Approved tool call: {record.tool_call.name}")

        elif command == "/deny":
            pending = self._runner._execution_environment.get_pending_tool_calls()
            if not pending:
                return (True, "No pending tool calls")
            record = pending[0]
            approval_event = ApprovalEvent(
                tool_call_id=record.tool_call_id,
                tool_call=record.tool_call,
                approved=False,
            )
            self._runner.push_event(approval_event)
            return (True, f"Denied tool call: {record.tool_call.name}")

        elif command == "/pending":
            pending = self._runner._execution_environment.get_pending_tool_calls()
            if not pending:
                return (True, "No pending tool calls")
            output = "Pending tool calls:"
            for record in pending:
                output += f"\n  [{record.tool_call_id}] {record.tool_call.name} ({record.approval_status.value})"
            return (True, output)

        elif command in ("/image", "/file"):
            if not args:
                return (True, "Usage: /file <filepath or url> [text]")
            return self._handle_file(args)

        elif command == "/quit":
            self._running = False
            return (False, "Goodbye!")

        else:
            return (True, f"Unknown command: {command}. Use /approve, /deny, /pending, /image (or /file), /quit.")

    async def read_input_loop(self) -> None:
        """Read user input and send events to the runner's queue."""
        await self.send("Connected. Commands: /approve, /deny, /pending, /image (or /file), /quit")

        while self.is_running():
            # Print prompt before reading input
            prompt = self._get_prompt()
            print(prompt, end="", flush=True)

            # Read input in parallel with notifications
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, self._get_input_line
                )
            except Exception:
                break

            if line is None:
                break

            line = line.strip()
            if not line:
                continue

            if line.startswith("/"):
                should_continue, output = self.handle_command(line)
                await self.send(output)
                if not should_continue:
                    break
            else:
                try:
                    self._post_message_to_runner(line)
                except Exception as e:
                    await self.send(f"Error: {type(e).__name__}: {str(e)}")
