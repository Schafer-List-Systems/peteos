"""InteractiveShellChannel - REPL-style interaction with agents via queues."""

import asyncio
import sys
import uuid
from typing import Optional

from peteos.channels.channel import Channel
from peteos.chatbot import Message, ContentPart
from peteos.logger import get_logger

_logger = get_logger(__name__)


class InteractiveShellChannel(Channel):
    """Interactive shell channel for REPL-style interaction with agents.

    This channel uses the Agent's queue-based architecture:
    - User input (non-command lines) is posted to the Agent's message queue
    - Agent processes messages and triggers session execution
    - Notifications are pushed to the channel's notification queue
    - Channel consumes notifications and displays them to user

    The shell runs in a separate thread from the Agent's event loop,
    so blocking input() doesn't block session processing.

    Example:
        >>> agent = Agent(role_manager, chatbot_manager, tool_manager)
        >>> await agent.start()
        >>> shell = InteractiveShellChannel("shell", agent)
        >>> await shell.start()
        >>> # User types: /new test
        >>> # User types: Hello!
        >>> # Agent processes and notifications appear
        >>> await shell.stop()
        >>> await agent.stop()
    """

    def __init__(self, name: str, agent):
        """Initialize InteractiveShellChannel.

        Args:
            name: Unique identifier for this channel.
            agent: The Agent instance this channel connects to.
        """
        super().__init__(name, agent)

    def send(self, message: Message | str) -> None:
        """Send a message to the shell.

        Args:
            message: The Message to display, or a plain string.
        """
        if isinstance(message, str):
            print(message)
        else:
            print(message.text)

    def _post_message_to_agent(self, session_uuid: uuid.UUID, content: str) -> None:
        """Post a message to the Agent's message queue.

        Non-blocking queue put. If queue is full, blocks until space available.

        Args:
            session_uuid: The UUID of the target session.
            content: The message content from the user.
        """
        if self._agent:
            message = Message(
                role="user",
                content=[ContentPart(part_type="text", text=content)]
            )
            self._agent.post_message(session_uuid, message)

    def _get_input_line(self) -> Optional[str]:
        """Synchronous input reader for use with run_in_executor."""
        try:
            line = sys.stdin.readline()
            return line if line else None
        except EOFError:
            return None

    def receive(self) -> str | None:
        """Receive method for Channel compatibility.

        Returns None since run() uses run_in_executor instead.
        """
        return None

    async def start(self) -> None:
        """Start the shell channel.

        Starts the background notification consumption loop via super().start()
        and the input reading loop via asyncio.create_task().
        """
        await super().start()
        asyncio.create_task(self.read_input_loop())

    async def stop(self) -> None:
        """Stop the shell channel gracefully."""
        await super().stop()

    def select_session(self, session_uuid: uuid.UUID) -> None:
        """
        Select a session as the active session for this channel.

        Also subscribes to notifications for this session via the base class.

        Args:
            session_uuid: The UUID of the session to select.
        """
        super().select_session(session_uuid)
        self.subscribe_to_session(session_uuid)

    def _get_prompt(self) -> str:
        """Get the prompt string for the shell."""
        if self._active_session_uuid:
            short_uuid = str(self._active_session_uuid)[:8]
            session = self._agent.get_session(self._active_session_uuid)
            role_name = session.role.name if session else "agent"
            return f"{short_uuid} @{role_name} >> "
        return ">> "

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

        if command == "/new":
            if not args:
                return (True, "Usage: /new <role>")
            try:
                session = self._agent.create_session(args.strip())
                self.select_session(session.uuid)
                _logger.debug(f"Session created: {session.uuid}")
                return (True, "")
            except ValueError as e:
                return (True, str(e))

        elif command == "/list":
            sessions = self._agent.list_sessions()
            if not sessions:
                return (True, "No sessions available")
            output = "Sessions:"
            for uuid_, session in sessions.items():
                active_marker = " (active)" if uuid_ == self._active_session_uuid else ""
                output += f"\n  {uuid_}{active_marker} - Role: {session.role.name}"
            return (True, output)

        elif command == "/select":
            if not args:
                return (True, "Usage: /select <uuid>")
            try:
                session_uuid = uuid.UUID(args.strip())
                session = self._agent.get_session(session_uuid)
                if session is None:
                    return (True, f"Session not found: {session_uuid}")
                self.select_session(session_uuid)
                _logger.debug(f"Active session: {session_uuid}")
                return (True, "")
            except ValueError:
                return (True, f"Invalid UUID: {args}")

        elif command == "/messages":
            if self._active_session_uuid is None:
                return (True, "No session selected")
            session = self._agent.get_session(self._active_session_uuid)
            if session is None:
                return (True, "Session not found")
            history = session.chat_history.messages
            if not history:
                return (True, "No messages in session history")
            output = f"Recent messages ({min(10, len(history))} of {len(history)}):"
            for msg in history[-10:]:
                content = msg.content
                role = content.get("role", "unknown")
                text = content.get("text", content.get("content", ""))[:100]
                output += f"\n  [{role}] {text}"
            return (True, output)

        elif command == "/quit":
            self._running = False
            return (False, "Goodbye!")

        else:
            return (True, f"Unknown command: {command}. Use /list for available commands.")

    async def read_input_loop(self) -> None:
        """Read user input and send events to the channel's queue."""
        self.send("Connected. Commands: /new, /list, /select, /messages, /quit")

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
                self.send(output)
                if not should_continue:
                    break
            else:
                # Push user input as an event to the channel queue.
                # The Channel.run() loop consumes events via _wait() and
                # processes them by calling send(). For non-command input,
                # we need to forward to the agent's message queue directly
                # since this is a shell-specific flow.
                if self._active_session_uuid is None:
                    self.send("No session selected. Use /new <role> or /select <uuid>.")
                    continue

                try:
                    self._post_message_to_agent(self._active_session_uuid, line)
                except Exception as e:
                    self.send(f"Error: {type(e).__name__}: {str(e)}")
