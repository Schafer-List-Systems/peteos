import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Set

from peteos.chatbot import Message


class Channel(ABC):
    """Abstract base class for channels connecting users to agents."""

    _registry: Dict[str, "Channel"] = {}

    def __init__(self, name: str, agent):
        """
        Initialize Channel.

        Args:
            name: Unique identifier for this channel.
            agent: The Agent instance this channel connects to.
        """
        self.name = name
        self._agent = agent
        self._active_session_uuid: uuid.UUID | None = None
        self._session_consumer_tasks: Dict[uuid.UUID, asyncio.Task] = {}
        self._running: bool = False
        self._show_reasoning: bool = True
        self._show_tool_calls: bool = True
        self._show_tool_results: bool = True
        Channel._registry[name] = self
        agent.register_channel(self)

    @property
    def active_session_uuid(self) -> uuid.UUID | None:
        """Get the currently active session UUID for this channel."""
        return self._active_session_uuid

    def select_session(self, session_uuid: uuid.UUID) -> None:
        """
        Select a session as the active session for this channel.

        Args:
            session_uuid: The UUID of the session to select.
        """
        self._active_session_uuid = session_uuid

    @abstractmethod
    def send(self, message: Message) -> None:
        """
        Send a message to the user through this channel.

        Each channel decides how to format or serialize the Message
        for its specific medium (terminal, HTTP, chat platform, etc.).

        Args:
            message: The Message to send.
        """
        pass

    @abstractmethod
    def receive(self) -> str | None:
        """
        Receive a message from the user.

        Returns:
            The received message, or None if channel is closed.
        """
        pass

    @classmethod
    def deregister_all(cls) -> None:
        """Remove all channels from the registry."""
        cls._registry.clear()

    @classmethod
    def get_by_name(cls, name: str) -> "Channel | None":
        """
        Get a channel by name.

        Args:
            name: The channel name.

        Returns:
            The Channel instance, or None if not found.
        """
        return cls._registry.get(name)

    @classmethod
    def list_all(cls) -> Dict[str, "Channel"]:
        """
        List all registered channels.

        Returns:
            Dictionary mapping channel names to Channel instances.
        """
        return dict(cls._registry)

    def subscribe_to_session(self, session_uuid: uuid.UUID) -> None:
        """Subscribe this channel to notifications for a session.

        Creates the notification queue, registers the channel with the
        agent's _session_channels, and starts the notification consumer
        for the given session.

        Args:
            session_uuid: The session to subscribe to.
        """
        if not self._running:
            return

        queue_key = (self.name, session_uuid)

        if queue_key not in self._agent._notification_queues:
            self._agent._notification_queues[queue_key] = asyncio.Queue()
        if session_uuid not in self._agent._session_channels:
            self._agent._session_channels[session_uuid] = set()
        self._agent._session_channels[session_uuid].add(self)

        # Start notification consumer if not already running for this session
        if session_uuid not in self._session_consumer_tasks or self._session_consumer_tasks[session_uuid].done():
            self._session_consumer_tasks[session_uuid] = asyncio.create_task(
                self._consume_notifications(session_uuid)
            )

    def unsubscribe_from_session(self, session_uuid: uuid.UUID) -> None:
        """Unsubscribe this channel from notifications for a session.

        Cancels the notification consumer task for the given session.

        Args:
            session_uuid: The session to unsubscribe from.
        """
        task = self._session_consumer_tasks.pop(session_uuid, None)
        if task and not task.done():
            task.cancel()

    async def _consume_notifications(self, session_uuid: uuid.UUID) -> None:
        """Poll the notification queue and forward messages via send().

        Runs until the channel is stopped. Subclasses may override to customize
        notification delivery behavior.

        Args:
            session_uuid: The session to consume notifications for.
        """
        queue_key = (self.name, session_uuid)
        try:
            while self._running:
                queue = self._agent._notification_queues.get(queue_key)
                if not queue:
                    await asyncio.sleep(0.1)
                    continue
                if not queue.empty():
                    try:
                        message = queue.get_nowait()
                        if message.role == "reasoning" and not self._show_reasoning:
                            continue
                        elif message.role == "tool" and not self._show_tool_calls:
                            continue
                        elif message.role == "tool_result" and not self._show_tool_results:
                            continue
                        self._active_session_uuid = session_uuid
                        self.send(message)
                    except Exception:
                        pass
                else:
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    def enable_reasoning(self, on: bool) -> None:
        self._show_reasoning = on

    def enable_tool_calls(self, on: bool) -> None:
        self._show_tool_calls = on

    def enable_tool_results(self, on: bool) -> None:
        self._show_tool_results = on
