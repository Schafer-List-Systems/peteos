"""InteractiveShellChannel - REPL-style interaction with agents via queues."""

import asyncio
import sys
import uuid
from typing import Optional

from peteos.channel import Channel
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
        self._running = False  # Explicit start/stop lifecycle
        self._notification_queues: dict = {}

    def send(self, message: str) -> None:
        """Send a message to the shell.

        Args:
            message: The message to display.
        """
        print(message)

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

    async def _consume_notifications(self, session_uuid: uuid.UUID) -> None:
        """Consume notifications from the Agent's notification queue.

        Runs until the Agent is stopped. Blocks on queue get(), so non-blocking.

        Args:
            session_uuid: The UUID of the session to subscribe to.
        """
        while self._running:
            try:
                # Get notifications for this channel/session
                queue_key = (self.name, session_uuid)
                if queue_key not in self._agent._notification_queues:
                    # Subscribe to this channel's notifications for this session
                    self._agent._notification_queues[queue_key] = asyncio.Queue()
                    if session_uuid not in self._agent._session_channels:
                        self._agent._session_channels[session_uuid] = set()

                queue = self._agent._notification_queues[queue_key]
                if not queue.empty():
                    notification = queue.get_nowait()
                    self.send(notification)
                else:
                    # Wait a bit before checking again
                    await asyncio.sleep(0.1)
            except Exception:
                # Queue might be closed, break
                break

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

        Initializes notification consumption for the active session.
        """
        if self._running:
            return

        self._running = True
        self.send(f"Connected. Commands: /new, /list, /select, /messages, /quit")

    async def stop(self) -> None:
        """Stop the shell channel gracefully."""
        if not self._running:
            return

        self._running = False

    def select_session(self, session_uuid: uuid.UUID) -> None:
        """
        Select a session as the active session for this channel.

        Also subscribes to notifications for this session.

        Args:
            session_uuid: The UUID of the session to select.
        """
        super().select_session(session_uuid)

        # Subscribe to notifications for this session
        queue_key = (self.name, session_uuid)
        if queue_key not in self._agent._notification_queues:
            self._agent._notification_queues[queue_key] = asyncio.Queue()
        if session_uuid not in self._agent._session_channels:
            self._agent._session_channels[session_uuid] = set()
        # Also register the channel so it's tracked
        if session_uuid not in self._agent._session_channels:
            self._agent._session_channels[session_uuid] = set()
        self._agent._session_channels[session_uuid].add(self)

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

    async def run(self) -> None:
        """Run the shell interaction loop.

        This is the main async loop that:
        1. Reads user input via blocking input() (in executor)
        2. Processes commands
        3. Posts messages to Agent for non-command input
        4. Consumes notifications from Agent's queue in parallel
        """
        await self.start()

        # Notification task for active session
        notification_task: Optional[asyncio.Task] = None

        try:
            while self._running:
                # Start notification consumer if we have an active session
                if self._active_session_uuid is not None and notification_task is None:
                    notification_task = asyncio.create_task(
                        self._consume_notifications(self._active_session_uuid)
                    )

                # Read input in parallel with notifications
                try:
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, self._get_input_line
                    )
                except Exception:
                    self._running = False
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
                    # Forward message to active session via Agent's queue
                    if self._active_session_uuid is None:
                        self.send("No session selected. Use /new <role> or /select <uuid>.")
                        continue

                    try:
                        session_uuid = self._active_session_uuid
                        self._post_message_to_agent(session_uuid, line)
                    except Exception as e:
                        self.send(f"Error: {type(e).__name__}: {str(e)}")

        finally:
            await self.stop()
            if notification_task and not notification_task.done():
                notification_task.cancel()
                try:
                    await notification_task
                except asyncio.CancelledError:
                    pass
