import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Dict, Optional, Set

from peteos.utils.activeclass import ActiveClass
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
        self._session_uuid: uuid.UUID | None = None
        self._show_reasoning: bool = True
        self._show_tool_calls: bool = True
        self._show_tool_results: bool = True
        Channel._registry[name] = self
        agent.register_channel(self)

    @abstractmethod
    async def send(self, message: Message, session_uuid: uuid.UUID | None = None) -> None:
        """
        Send a message to the user through this channel.

        Each channel decides how to format or serialize the Message
        for its specific medium (terminal, HTTP, chat platform, etc.).

        Args:
            message: The Message to send.
            session_uuid: The session UUID to route to.
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
                if message.get_role() == "reasoning" and not self._show_reasoning:
                    continue
                elif message.get_role() == "tool" and not self._show_tool_calls:
                    continue
                elif message.get_role() == "tool_result" and not self._show_tool_results:
                    continue
                await self.send(message, session_uuid=session_uuid)
        finally:
            await self.stop()

    def subscribe_to_session(self, session_uuid: uuid.UUID) -> bool:
        """Subscribe this channel to notifications for a session.

        Raises no exception but returns False if already subscribed to
        a different session, or if the session doesn't exist.

        Args:
            session_uuid: The session to subscribe to.
        """
        if session_uuid not in self._agent._sessions:
            return False
        if self._session_uuid is not None and self._session_uuid != session_uuid:
            return False  # already subscribed to a different session
        self._session_uuid = session_uuid
        session = self._agent.get_session(session_uuid)
        if session:
            return session.subscribe(self)
        return False

    def unsubscribe_from_session(self, session_uuid: uuid.UUID) -> bool:
        """Unsubscribe this channel from notifications for a session.

        Returns False if the session doesn't exist or the channel is
        not subscribed to it.

        Args:
            session_uuid: The session to unsubscribe from.
        """
        if session_uuid not in self._agent._sessions:
            return False
        session = self._agent.get_session(session_uuid)
        if session:
            result = session.unsubscribe(self)
            if result:
                self._session_uuid = None
            return result
        return False

    def enable_reasoning(self, on: bool) -> None:
        self._show_reasoning = on

    def enable_tool_calls(self, on: bool) -> None:
        self._show_tool_calls = on

    def enable_tool_results(self, on: bool) -> None:
        self._show_tool_results = on
