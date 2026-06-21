"""InteractiveShellChannel - REPL-style interaction with agents via queues."""

import asyncio
import sys
import uuid
from typing import Optional

from peteos.channels.channel import Channel
from peteos.chatbot import Message, ContentPart
from peteos.logger import get_logger
from peteos.session import ApprovalEvent

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
        >>> role = Role(name="test", description="Test role")
        >>> agent = Agent(role, tool_manager)
        >>> await agent.start()
        >>> shell = InteractiveShellChannel("shell", agent)
        >>> await shell.start()
        >>> # User types: /new
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
        self._active_session_uuid: uuid.UUID | None = None

    @property
    def active_session_uuid(self) -> uuid.UUID | None:
        """Get the currently active session UUID for this shell."""
        return self._active_session_uuid

    @active_session_uuid.setter
    def active_session_uuid(self, value: uuid.UUID | None) -> None:
        self._active_session_uuid = value

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

    def _post_message_to_agent(self, session_uuid: uuid.UUID, content: str) -> None:
        """Post a message directly to the session's event queue.

        Non-blocking. The session's event loop processes the message.
        Since this method is sync (called from the input loop), uses
        asyncio.create_task to call the async queue_message.

        Args:
            session_uuid: The UUID of the target session.
            content: The message content from the user.
        """
        session = self._agent.get_session(session_uuid)
        if session:
            message = Message(
                role="user",
                content=[ContentPart(part_type="text", text=content)]
            )
            asyncio.create_task(session.queue_message(message))

    def _handle_file(self, args: str) -> tuple[bool, str]:
        """Handle /image or /file command."""
        if self._active_session_uuid is None:
            return (True, "No session selected")
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

        parts = [ContentPart(part_type="text", text=text)] if text else []
        parts.append(file_part)
        message = Message(role="user", content=parts)
        session = self._agent.get_session(self._active_session_uuid)
        if session:
            await session.queue_message(message)
            _logger.debug(
                "%s message queued for session %s from %s",
                file_type, self._active_session_uuid, src,
            )

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
        asyncio.create_task(self.read_input_loop())

    async def stop(self) -> None:
        """Stop the shell channel gracefully."""
        await super().stop()

    def select_session(self, session_uuid: uuid.UUID) -> None:
        """Select a session as the active session for this channel.

        Also subscribes to notifications for this session via the base class.

        Args:
            session_uuid: The UUID of the session to select.
        """
        self._active_session_uuid = session_uuid
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
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._handle_new_session())
                    return (True, "")
                else:
                    session = loop.run_until_complete(
                        self._agent.create_session()
                    )
                    self.select_session(session.uuid)
                    _logger.debug("Session created: %s", session.uuid)
                    return (True, "")
            except Exception as e:
                return (True, str(e))

        elif command == "/list":
            sessions = self._agent.list_sessions()
            if not sessions:
                role_name = self._agent.role.name if hasattr(self._agent, "role") else "??"
                return (True, f"No sessions available (role: {role_name})")
            output = f"Sessions ({self._agent.role.name}):"
            for uuid_, session in sessions.items():
                active_marker = " (active)" if uuid_ == self._active_session_uuid else ""
                output += f"\n  {uuid_}{active_marker}"
            return (True, output)

        elif command == "/switch":
            if not args:
                return (True, "Usage: /switch <uuid>")
            try:
                session_uuid = uuid.UUID(args.strip())
                session = self._agent.get_session(session_uuid)
                if session is None:
                    return (True, f"Session not found: {session_uuid}")
                self.select_session(session_uuid)
                _logger.debug("Active session: %s", session_uuid)
                return (True, "")
            except ValueError:
                return (True, f"Invalid UUID: {args}")

        elif command == "/approve":
            if self._active_session_uuid is None:
                return (True, "No session selected")
            session = self._agent.get_session(self._active_session_uuid)
            if session is None:
                return (True, "Session not found")
            pending = session.get_pending_tool_calls()
            if not pending:
                return (True, "No pending tool calls")
            record = pending[0]
            approval_event = ApprovalEvent(
                tool_call_id=record.tool_call_id,
                tool_call=record.tool_call,
                approved=True,
            )
            session.push_event(approval_event)
            return (True, f"Approved tool call: {record.tool_call.get('name')}")

        elif command == "/deny":
            if self._active_session_uuid is None:
                return (True, "No session selected")
            session = self._agent.get_session(self._active_session_uuid)
            if session is None:
                return (True, "Session not found")
            pending = session.get_pending_tool_calls()
            if not pending:
                return (True, "No pending tool calls")
            record = pending[0]
            approval_event = ApprovalEvent(
                tool_call_id=record.tool_call_id,
                tool_call=record.tool_call,
                approved=False,
            )
            session.push_event(approval_event)
            return (True, f"Denied tool call: {record.tool_call.get('name')}")

        elif command == "/pending":
            if self._active_session_uuid is None:
                return (True, "No session selected")
            session = self._agent.get_session(self._active_session_uuid)
            if session is None:
                return (True, "Session not found")
            pending = session.get_pending_tool_calls()
            if not pending:
                return (True, "No pending tool calls")
            output = "Pending tool calls:"
            for record in pending:
                output += f"\n  [{record.tool_call_id}] {record.tool_call.get('name', '?')} ({record.approval_status.value})"
            return (True, output)

        elif command in ("/image", "/file"):
            if not args or self._active_session_uuid is None:
                return (True, "Usage: /file <filepath or url> [text]")
            return self._handle_file(args)

        elif command == "/quit":
            self._running = False
            return (False, "Goodbye!")

        else:
            return (True, f"Unknown command: {command}. Use /list for available commands.")

    async def _handle_new_session(self) -> None:
        """Async helper for /new command."""
        session = await self._agent.create_session()
        self.select_session(session.uuid)
        _logger.debug("Session created: %s", session.uuid)

    async def read_input_loop(self) -> None:
        """Read user input and send events to the channel's queue."""
        await self.send("Connected. Commands: /new, /list, /switch, /approve, /deny, /pending, /image (or /file), /quit")

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
                # Push user input as an event to the channel queue.
                # The Channel.run() loop consumes events via _wait() and
                # processes them by calling send(). For non-command input,
                # we need to forward to the agent's message queue directly
                # since this is a shell-specific flow.
                if self._active_session_uuid is None:
                    await self.send("No session selected. Use /new or /switch <uuid>.")
                    continue

                try:
                    self._post_message_to_agent(self._active_session_uuid, line)
                except Exception as e:
                    await self.send(f"Error: {type(e).__name__}: {str(e)}")
