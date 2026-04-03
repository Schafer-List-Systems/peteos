import uuid

from peteos.channel import Channel
from peteos.chatbot import Message


class InteractiveShellChannel(Channel):
    """Interactive shell channel for REPL-style interaction with agents."""

    def __init__(self, name: str, agent):
        """
        Initialize InteractiveShellChannel.

        Args:
            name: Unique identifier for this channel.
            agent: The Agent instance this channel connects to.
        """
        super().__init__(name, agent)
        self._running = True

    def send(self, message: str) -> None:
        """Send a message to the shell."""
        print(f"[{self.name}] {message}")

    def receive(self) -> str | None:
        """
        Receive a message from user input.

        Returns:
            The user input, or None if channel is closed.
        """
        if not self._running:
            return None
        try:
            return input()
        except EOFError:
            self._running = False
            return None

    def handle_command(self, line: str) -> tuple[bool, str]:
        """
        Handle a shell command.

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
                return (True, f"Session created: {session.uuid}")
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
                return (True, f"Active session: {session_uuid}")
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

    def run(self) -> None:
        """Run the shell interaction loop."""
        self.send(f"Connected. Commands: /new, /list, /select, /messages, /quit")
        self.send("")

        while self._running:
            line = self.receive()
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
                # Forward message to active session
                if self._active_session_uuid is None:
                    self.send("No session selected. Use /new <role> or /select <uuid>.")
                    continue

                try:
                    session_uuid = self._active_session_uuid
                    session = self._agent.get_session(session_uuid)
                    if session is None:
                        self.send(f"Session {session_uuid} not found.")
                        continue

                    message = Message(content={"role": "user", "content": line})
                    self.send(f"Sending message to session {session_uuid}...")
                    import asyncio
                    asyncio.run(session.queue_message(message))
                except Exception as e:
                    self.send(f"Error: {type(e).__name__}: {str(e)}")

    def close(self) -> None:
        """Close the channel."""
        self._running = False
