import uuid
from abc import ABC, abstractmethod
from typing import Dict, Set


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
        self._agent.register_channel(self)
        self._active_session_uuid: uuid.UUID | None = None
        Channel._registry[name] = self

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
    def send(self, message: str) -> None:
        """
        Send a message to the user through this channel.

        Args:
            message: The message to send.
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
