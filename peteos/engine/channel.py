"""Channel — abstract base class for connecting runners to users."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from peteos.utils.activeclass import ActiveClass
from peteos.chatbot import Message


@dataclass(frozen=True)
class NotificationEvent:
    session_uuid: uuid.UUID
    message: Message


class Channel(ActiveClass, ABC):
    """Abstract base class for channels connecting runners to users."""

    _registry: Dict[str, "Channel"] = {}

    def __init__(self, name: str, runner):
        """
        Initialize Channel.

        Args:
            name: Unique identifier for this channel.
            runner: The Runner instance this channel connects to.
        """
        super().__init__()
        self.name = name
        self._runner = runner
        self._session_uuid: uuid.UUID | None = None
        self._show_reasoning: bool = True
        self._show_tool_calls: bool = True
        self._show_tool_results: bool = True
        Channel._registry[name] = self
        runner.subscribe(self)

    @abstractmethod
    async def send(self, message: Message, session_uuid: uuid.UUID | None = None) -> None:
        """
        Send a message to the user through this channel.

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
                await self.send(message, session_uuid=session_uuid)
        finally:
            await self.stop()

    def subscribe_to_session(self, session_uuid: uuid.UUID) -> bool:
        """Mark this channel as subscribed to a session.

        The channel is already registered with its runner via the
        constructor. This method simply stores the session UUID for
        routing purposes.

        Args:
            session_uuid: The session to subscribe to.
        """
        if self._session_uuid is not None and self._session_uuid != session_uuid:
            return False
        self._session_uuid = session_uuid
        return True

    def unsubscribe_from_session(self, session_uuid: uuid.UUID) -> bool:
        """Unsubscribe this channel from a session.

        Args:
            session_uuid: The session to unsubscribe from.
        """
        if self._session_uuid is not None:
            self._session_uuid = None
            return True
        return False

    def enable_reasoning(self, on: bool) -> None:
        self._show_reasoning = on

    def enable_tool_calls(self, on: bool) -> None:
        self._show_tool_calls = on

    def enable_tool_results(self, on: bool) -> None:
        self._show_tool_results = on
