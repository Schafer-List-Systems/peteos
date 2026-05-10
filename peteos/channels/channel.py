import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional, Set

from peteos.activeclass import ActiveClass
from peteos.chatbot import Message


@dataclass(frozen=True)
class NotificationEvent:
    session_uuid: uuid.UUID
    message: Message


class Channel(ActiveClass, ABC):
    """Abstract base class for channels connecting users to agents."""

    _registry: Dict[str, "Channel"] = {}

    def __init__(self, name: str, agent):
        """
        Initialize Channel.

        Args:
            name: Unique identifier for this channel.
            agent: The Agent instance this channel connects to.
        """
        super().__init__()
        self.name = name
        self._agent = agent
        self._show_reasoning: bool = True
        self._show_tool_calls: bool = True
        self._show_tool_results: bool = True
        Channel._registry[name] = self
        agent.register_channel(self)

    @abstractmethod
    def send(self, message: Message, session_uuid: uuid.UUID | None = None) -> None:
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

    async def run(self) -> None:
        """Main notification consumption loop.

        Consumes events pushed via push_event() and delivers them via send().
        Runs until the channel is stopped.

        Call ``await channel.start()`` to begin the background loop,
        and ``await channel.stop()`` to end it.
        """
        try:
            while self.is_running():
                event = await self._wait()
                if event is None:
                    break
                if isinstance(event, NotificationEvent):
                    message = event.message
                    session_uuid: uuid.UUID | None = event.session_uuid
                else:
                    # Legacy: plain Message pushed directly (e.g., by tests)
                    message = event
                    session_uuid = None
                if message is None:
                    break
                if message.role == "reasoning" and not self._show_reasoning:
                    continue
                elif message.role == "tool" and not self._show_tool_calls:
                    continue
                elif message.role == "tool_result" and not self._show_tool_results:
                    continue
                self.send(message, session_uuid=session_uuid)
        finally:
            await self.stop()

    def subscribe_to_session(self, session_uuid: uuid.UUID) -> None:
        """Subscribe this channel to notifications for a session.

        Registers the channel with the agent's _session_channels and
        starts the notification consumer (self.run()).

        Args:
            session_uuid: The session to subscribe to.
        """
        if session_uuid not in self._agent._session_channels:
            self._agent._session_channels[session_uuid] = set()
        self._agent._session_channels[session_uuid].add(self)

    def unsubscribe_from_session(self, session_uuid: uuid.UUID) -> None:
        """Unsubscribe this channel from notifications for a session.

        Removes the channel from the session's subscribed channels.

        Args:
            session_uuid: The session to unsubscribe from.
        """
        if session_uuid in self._agent._session_channels:
            if self in self._agent._session_channels[session_uuid]:
                self._agent._session_channels[session_uuid].remove(self)

    def enable_reasoning(self, on: bool) -> None:
        self._show_reasoning = on

    def enable_tool_calls(self, on: bool) -> None:
        self._show_tool_calls = on

    def enable_tool_results(self, on: bool) -> None:
        self._show_tool_results = on
