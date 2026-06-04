"""ReadStdoutChannel - Read stdout from a process and forward matching lines to a session."""

import asyncio
import json
import re
import uuid
from typing import Optional

from peteos.chatbot import Message, ContentPart
from peteos.channels.channel import Channel
from peteos.logger import get_logger, truncate

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
            config={"command": ["journalctl", "-f"], "pattern": r"ERROR|WARNING"},
        )
        channel.subscribe_to_session(some_uuid)
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
            config: Dict with command and optional pattern.
        """
        super().__init__(name, agent)
        self._config = config
        self._prefix = config.get("prefix", "")
        self._process: asyncio.subprocess.Process | None = None
        self._exclude_patterns: list[str] = []
        # Populate exclude patterns from config
        self._exclude_patterns.extend(self._config.get("exclude", []))
        # Load persisted patterns from file if configured
        patterns_file = self._config.get("patterns_file")
        if patterns_file:
            self._exclude_patterns.extend(ReadStdoutChannel.load_patterns(patterns_file))
        self._patterns_file = patterns_file  # Keep for save on stop

    def send(self, message, session_uuid: uuid.UUID | None = None) -> None:
        """No-op - this channel is read-only."""
        pass

    def add_exclude_pattern(self, pattern: str) -> int | None:
        """Add a regex pattern to the exclusion list.

        Lines matching any exclude pattern will be dropped before
        the inclusive pattern check.

        Args:
            pattern: Regular expression pattern to exclude.

        Returns:
            Index where the pattern was added, or None if it already existed.
        """
        if pattern in self._exclude_patterns:
            return None
        self._exclude_patterns.append(pattern)
        return len(self._exclude_patterns) - 1

    def remove_exclude_pattern(self, index: int) -> bool:
        """Remove an exclude pattern by its index in the list.

        Args:
            index: The index of the pattern to remove.

        Returns:
            True if a pattern was removed, False if the index was out of range.
        """
        try:
            self._exclude_patterns.pop(index)
            return True
        except IndexError:
            return False

    def list_exclude_patterns(self) -> list[str]:
        """List all current exclude patterns.

        Returns:
            List of active exclude regex patterns.
        """
        return list(self._exclude_patterns)

    def save_patterns(self, path: str) -> None:
        """Save current exclude patterns to a JSON file.

        Args:
            path: File path to save patterns to.
        """
        with open(path, "w") as f:
            json.dump(self._exclude_patterns, f, indent=2)
        _logger.info("Saved %d exclude patterns to %s", len(self._exclude_patterns), path)

    @staticmethod
    def load_patterns(path: str) -> list[str]:
        """Load exclude patterns from a JSON file.

        Args:
            path: File path to load patterns from.

        Returns:
            List of loaded patterns, or empty list if file doesn't exist.
        """
        import os
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            patterns = json.load(f)
        _logger.info("Loaded %d exclude patterns from %s", len(patterns), path)
        return patterns


    @staticmethod
    def load_config(config_file: str) -> dict:
        """Load and validate ReadStdoutChannel config from JSON file.

        Required fields: command.
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
        config.setdefault("exclude", [])
        config.setdefault("patterns_file", None)
        config.setdefault("prefix", "")
        required = ["command", "process_terminate_timeout"]
        missing = [k for k in required if k not in config]
        if missing:
            raise KeyError(f"Missing required config fields: {', '.join(missing)}")
        return config

    async def start(self) -> None:
        """Start the subprocess and begin reading stdout."""
        if self._running:
            return

        if self._session_uuid is None:
            raise RuntimeError(f"{self.name} is not subscribed to a session. Call subscribe_to_session() before start().")

        self._running = True
        _logger.info("Starting %s: %s", self.name, self._config["command"])
        self._process = await asyncio.create_subprocess_exec(
            *self._config["command"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        asyncio.create_task(self._read_loop())

    async def stop(self) -> None:
        """Stop the subprocess and save exclude patterns."""
        self._running = False
        # Save patterns before stopping
        if self._patterns_file:
            try:
                self.save_patterns(self._patterns_file)
            except Exception as e:
                _logger.error("Failed to save exclude patterns: %s", e)
        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=self._config["process_terminate_timeout"])
            except asyncio.TimeoutError:
                _logger.error(
                    "Process %s did not terminate within %ss, killing",
                    self._config["command"],
                    self._config["process_terminate_timeout"],
                )
                self._process.kill()
            _logger.info("Stopped %s: %s", self.name, self._config["command"])

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
                if not re.search(self._config["pattern"], line):
                    continue
                for ex_pattern in self._exclude_patterns:
                    if re.search(ex_pattern, line):
                        _logger.debug("Excluded by pattern '%s': %s", ex_pattern, truncate(line))
                        break
                else:
                    message = Message(
                        role="user",
                        content=[ContentPart(part_type="text", text=f"{self._prefix}{line}")],
                    )
                    try:
                        await self._agent.get_session(self._session_uuid).queue_message(message)
                        _logger.debug("Queued to session %s: %s", self._session_uuid, truncate(line))
                    except (KeyError, AttributeError):
                        _logger.warning("Session %s no longer exists, dropping message", self._session_uuid)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            _logger.error("Read loop error in %s: %s", self.name, e)
