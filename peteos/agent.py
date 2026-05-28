"""Agent - Central hub for session management and notifications."""

import asyncio
import uuid
from typing import Dict, Optional, Set

from peteos.channels.channel import Channel
from peteos.chatbot import Message, ContentPart
from peteos.logger import get_logger
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.session import Session
from peteos.toolmanager import ToolManager

_logger = get_logger(__name__)


class Agent:
    """Central hub for session management and event loop management.

    The Agent manages sessions and channels with full decoupling:
    - Channels post messages directly to the Session's queue (non-blocking)
    - Session's event loop processes messages and drives the execution environment
    - Sessions trigger hooks that publish notifications to per-channel queues
    - Channels consume notifications from their own notification queues

    The session's event loop runs in a background task (created by start()).
    """

    def __init__(
        self,
        role_manager: RoleManager,
        tool_manager: ToolManager
    ):
        self._role_manager = role_manager
        self._tool_manager = tool_manager

        # Session management
        self._sessions: Dict[uuid.UUID, Session] = {}

        # Track which channels are subscribed to which sessions

        # Track registered channels (for Channel base class compatibility)
        self._channels: Dict[str, Channel] = {}

    def register_channel(self, channel: Channel) -> None:
        """Register a channel with the agent."""
        self._channels[channel.name] = channel

    def deregister_channel(self, name: str) -> None:
        """Deregister a channel from the agent."""
        if name in self._channels:
            del self._channels[name]

    def get_channel(self, name: str) -> Channel | None:
        """Get a channel by name."""
        return self._channels.get(name)

    def list_channels(self) -> Dict[str, Channel]:
        """List all registered channels."""
        return dict(self._channels)

    async def create_session(self, role_name: str) -> Session:
        """Create a new session with the specified role.

        Creates per-session queue and registers hooks for notifications.
        Also starts the session's event loop.

        Args:
            role_name: The name of the role to use for this session.

        Returns:
            A new Session instance.

        Raises:
            ValueError: If the role is not found in RoleManager.
        """
        role = self._role_manager.get_role(role_name)
        if role is None:
            raise ValueError(f"Role '{role_name}' not found in RoleManager")

        session = Session(
            role=role,
            tool_manager=self._tool_manager,
        )
        self._sessions[session.uuid] = session

        # Register hooks on the session's execution environment
        env = session.execution_environment
        env.register_hook("before_tool_execution",
                          self._on_before_tool_execution)
        env.register_hook("after_tool_execution",
                          self._on_after_tool_execution)
        env.register_hook("before_notification_publish",
                          self._on_before_notification_publish)

        # Start the session's event loop
        await session.start()

        return session

    def get_session(self, session_uuid: uuid.UUID) -> Session | None:
        """Get a session by its UUID."""
        return self._sessions.get(session_uuid)

    def list_sessions(self) -> Dict[uuid.UUID, Session]:
        """List all sessions."""
        return dict(self._sessions)

    async def destroy_session(self, session_uuid: uuid.UUID) -> bool:
        """Destroy a session by its UUID.

        Stops the session's event loop.
        """
        if session_uuid in self._sessions:
            session = self._sessions[session_uuid]
            await session.stop()

            del self._sessions[session_uuid]
            return True
        return False

    def _on_before_tool_execution(
        self,
        session: Session,
        tool_call: dict
    ) -> tuple:
        """Hook callback fired before each tool execution.

        Auto-approves tools listed in the session's auto_approve_tools.

        Returns:
            Tuple (allow: bool | None, message: str | None) - Returns
            (True, None) for auto-approved tools, ("pending", None) otherwise.
        """
        tool_name = tool_call.get("name", "")
        if tool_name in session.auto_approve_tools:
            return (True, None)

        return ("pending", None)

    def _on_after_tool_execution(
        self,
        session: Session,
        tool_call: dict,
        result: str,
        success: bool
    ) -> None:
        """Hook callback fired after tool execution completes."""
        pass

    def _on_before_notification_publish(
        self,
        session: Session,
        message: Message
    ) -> None:
        """Hook callback fired before a notification is published to channels."""
