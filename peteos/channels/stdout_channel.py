"""ReadStdoutChannel - Read stdout from a process and forward matching lines to a session."""

import asyncio
import json
import re
import uuid
from typing import Optional

from peteos.chatbot import Message, ContentPart
from peteos.channels.channel import Channel
from peteos.logger import get_logger

_logger = get_logger(__name__)


class ReadStdoutChannel(Channel):
    """A read-only channel that monitors a subprocess's stdout.

    Matches each line against a regex. Matching lines become messages
    pushed into the session's queue. No send() is needed - the
    channel is a one-way source, not a destination.

    Example::

        channel = ReadStdoutChannel(
            name="docker-logs",
            agent=agent,
            config={"session_uuid": ..., "command": [...], "pattern": r"ERROR|WARNING"},
        )
        await channel.start()
        # messages matching the pattern are queued to the session
        await channel.stop()
    """

    def __init__(
        self,
        name: str,
        agent,
        config: dict,
    ):
        """Initialize the channel.

        Args:
            name: Unique identifier for this channel.
            agent: The Agent instance to queue messages to.
            config: Dict with session_uuid, command, and optional pattern.
        """
        super().__init__(name, agent)
        self._config = config
        self._process: asyncio.subprocess.Process | None = None

    def send(self, message, session_uuid: uuid.UUID | None = None) -> None:
        """No-op - this channel is read-only."""
        pass


    @staticmethod
    def load_config(config_file: str) -> dict:
        """Load and validate ReadStdoutChannel config from JSON file.

        Required fields: session_uuid, command.
        Optional fields: pattern (regex string, defaults to ".*").

        Args:
            config_file: Path to the JSON configuration file.

        Returns:
            Dict with validated config values and defaults applied.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            KeyError: If required fields are missing.
        """
        with open(config_file, "r") as f:
            config = json.load(f)
        config.setdefault("pattern", ".*")
        required = ["session_uuid", "command"]
        missing = [k for k in required if k not in config]
        if missing:
            raise KeyError(f"Missing required config fields: {', '.join(missing)}")
        # Parse session_uuid string to UUID
        config["session_uuid"] = uuid.UUID(config["session_uuid"])
        return config

    async def start(self) -> None:
        """Start the subprocess and begin reading stdout."""
        if self._running:
            return

        self._running = True
        _logger.info("Starting %s: %s", self.name, self._command)
        self._process = await asyncio.create_subprocess_exec(
            *self._command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        asyncio.create_task(self._read_loop())

    async def stop(self) -> None:
        """Stop the subprocess."""
        self._running = False
        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
            _logger.info("Stopped %s: %s", self.name, self._command)

    async def _read_loop(self) -> None:
        """Read stdout line by line and queue matching messages."""
        assert self._process is not None
        stdout_reader = self._process.stdout
        if stdout_reader is None:
            return

        try:
            while self.is_running():
                raw = await stdout_reader.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if not self._pattern.search(line):
                    continue

                message = Message(
                    role="user",
                    content=[ContentPart(part_type="text", text=line)],
                )
                try:
                    await self._agent.get_session(self._config["session_uuid"]).queue_message(message)
                    _logger.debug("Queued to session %s: %s", self._config["session_uuid"], line[:80])
                except (KeyError, AttributeError):
                    _logger.warning("Session %s no longer exists, dropping message", self._config["session_uuid"])
        except asyncio.CancelledError:
            pass
        except Exception as e:
            _logger.error("Read loop error in %s: %s", self.name, e)
