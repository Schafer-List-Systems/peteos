"""Agent - Central hub for session management and notifications."""

import asyncio
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Set

from peteos.channels.channel import Channel, NotificationEvent
from peteos.chatbot import ChatBotManager, Message, ContentPart
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
        chatbot_manager: ChatBotManager,
        tool_manager: ToolManager
    ):
        self._role_manager = role_manager
        self._chatbot_manager = chatbot_manager
        self._tool_manager = tool_manager

        # Session management
        self._sessions: Dict[uuid.UUID, Session] = {}

        # Track which channels are subscribed to which sessions
        self._session_channels: Dict[uuid.UUID, Set[Channel]] = {}

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
            chatbot_manager=self._chatbot_manager
        )
        self._sessions[session.uuid] = session

        # Initialize channel tracking
        self._session_channels[session.uuid] = set()

        # Register hooks on the session's execution environment
        env = session.execution_environment
        env.register_hook("before_tool_execution",
                          self._on_before_tool_execution,
                          session.uuid)
        env.register_hook("after_tool_execution",
                          self._on_after_tool_execution,
                          session.uuid)
        env.register_hook("before_loop_continue",
                          self._on_before_loop_continue,
                          session.uuid)
        env.register_hook("before_loop_exit",
                          self._on_before_loop_exit,
                          session.uuid)

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

        Cleans up channel tracking and stops the session's event loop.
        """
        if session_uuid in self._sessions:
            session = self._sessions[session_uuid]
            await session.stop()

            # Clean up channel tracking
            if session_uuid in self._session_channels:
                del self._session_channels[session_uuid]

            del self._sessions[session_uuid]
            return True
        return False

    def _on_before_tool_execution(
        self,
        session_uuid: uuid.UUID,
        tool_call: dict
    ) -> tuple:
        """Hook callback fired before each tool execution.

        Publishes notification to all subscribed channels. Auto-approves
        tools listed in the session's auto_approve_tools.

        Returns:
            Tuple (allow: bool | None, message: str | None) - Returns
            (True, None) for auto-approved tools, ("pending", None) otherwise.
        """
        msg = Message(
            role="tool",
            content=[ContentPart(part_type="tool_call", tool_call=tool_call)],
        )
        self._publish_notification(session_uuid, msg)

        session = self.get_session(session_uuid)
        if session is not None:
            tool_name = tool_call.get("name", "")
            if tool_name in session.auto_approve_tools:
                return (True, None)

        return ("pending", None)

    def _on_after_tool_execution(
        self,
        session_uuid: uuid.UUID,
        tool_call: dict,
        result: str,
        success: bool
    ) -> None:
        """Hook callback fired after tool execution completes."""
        status = "error" if not success else "ok"
        msg = Message(
            role="tool_result",
            content=[ContentPart(part_type="tool_result", content=result)],
            metadata={"tool_status": status},
        )
        self._publish_notification(session_uuid, msg)

    def _on_before_loop_continue(
        self,
        session_uuid: uuid.UUID,
        delta_messages: List[Message]
    ) -> None:
        """Hook callback fired when the loop continues after tool calls."""
        for msg in delta_messages:
            if msg.get_role() == "assistant" and msg.text:
                self._publish_notification(session_uuid, msg)

    def _on_before_loop_exit(
        self,
        session_uuid: uuid.UUID,
        reason: str
    ) -> None:
        """Hook callback fired when the loop exits."""
        session = self.get_session(session_uuid)
        if session:
            history = session.chat_history.messages
            # Search for last assistant message (anchored messages sit at edges)
            last_assistant = None
            for msg in reversed(history):
                if msg.get_role() == "assistant":
                    last_assistant = msg
                    break
            if last_assistant is not None:
                last_assistant.metadata["finish"] = True
                self._publish_notification(session_uuid, last_assistant)

    def _publish_notification(
        self,
        session_uuid: uuid.UUID,
        message: Message
    ) -> None:
        """Publish a notification to all subscribed channels."""
        channels = self._session_channels.get(session_uuid, set())
        for channel in channels:
            if channel:
                channel.push_event(NotificationEvent(session_uuid, message))
