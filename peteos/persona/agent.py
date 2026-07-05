"""Agent - Central hub for session management and notifications."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Dict, Optional

from peteos.utils import get_logger

from peteos.conversation.system_prompt_message import SystemPromptMessage
from peteos.conversation.tool_definitions_message import ToolDefinitionsMessage
from peteos.conversation.session import Session
from peteos.conversation.message import ContentPart

from .role import Role
from .toolmanager import ToolManager

_logger = get_logger(__name__)


class Agent:
    """Central hub for session management and event loop management.

    The Agent manages sessions and channels with full decoupling:
    - Channels post messages directly to the Session's queue (non-blocking)
    - Session's event loop processes messages and drives the execution environment
    - Sessions trigger hooks that publish notifications to per-channel queues
    - Channels consume notifications from their own notification queues

    The session's event loop runs in a background task (created by start()).

    The ``agent_base`` class attribute must be set by ``ConfigManager.init()``
    before any Agent instances are created.
    """

    agent_base: str = ""

    def __init__(
        self,
        role: Role,
        tool_manager: ToolManager,
        agent_base: str | None = None,
    ):
        self._role = role
        self._tool_manager = tool_manager
        base = agent_base or self.__class__.agent_base
        self._agent_dir = str(Path(base) / role.name)
        Path(self._agent_dir).mkdir(parents=True, exist_ok=True)

        # Session management
        self._sessions: Dict[str, Session] = {}

    async def create_session(self) -> Session:
        """Create a new session for this agent."""
        system_prompt_msg = SystemPromptMessage.create(self._role.system_prompt)
        tool_defs_msg = ToolDefinitionsMessage()
        session = Session.create(self._agent_dir, system_prompt_msg, tool_defs_msg)
        session.session_dir.mkdir(exist_ok=True)
        self._sessions[session.uuid] = session

        session.register_hook(tool_defs_msg, ToolDefinitionsMessage.TOOL_LIST_HOOK_NAME, self._tool_list_hook)
        session.register_hook(tool_defs_msg, ToolDefinitionsMessage.TOOL_FILTER_HOOK_NAME, self._tool_filter_hook)

        # TODO: Register hooks (before_tool_execution, after_tool_execution,
        # before_notification_publish) on the environment

        return session

    def get_session(self, session_uuid: str) -> Session | None:
        """Get a session by its UUID."""
        return self._sessions.get(session_uuid)

    def list_sessions(self) -> Dict[str, Session]:
        """List all sessions."""
        return dict(self._sessions)

    @property
    def agent_dir(self) -> str:
        """Return the agent's directory path."""
        return self._agent_dir

    @property
    def role(self) -> Role:
        """Return this agent's role."""
        return self._role

    @property
    def tool_manager(self) -> ToolManager:
        """Return this agent's tool manager."""
        return self._tool_manager

    def _tool_list_hook(self) -> str:
        """Hook callback that returns the tool list as a JSON string."""
        tool_list = self._tool_manager.get_tool_list()
        _logger.debug(
            "Agent._tool_list_hook: %d tools available for role '%s': %s",
            len(tool_list), self._role.name, [t.name for t in tool_list],
        )
        return json.dumps([
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tool_list
        ])

    def _tool_filter_hook(self) -> str:
        """Hook callback that returns the tool filter as a JSON string."""
        return json.dumps(self._role.tool_filter or [])

    def load(self, session_uuid: str) -> Session:
        """Load a session from disk.

        The session is loaded from ``{agent_dir}/{session_uuid}/session.json``.

        Args:
            session_uuid: The session's UUID or subdirectory name.

        Returns:
            A new Session instance.
        """
        session = Session.load(self._agent_dir, session_uuid)
        self._sessions[session.uuid] = session

        # Register tool hooks
        tool_defs_msg = session.active_context.tool_definitions_message
        session.register_hook(tool_defs_msg, ToolDefinitionsMessage.TOOL_LIST_HOOK_NAME, self._tool_list_hook)
        session.register_hook(tool_defs_msg, ToolDefinitionsMessage.TOOL_FILTER_HOOK_NAME, self._tool_filter_hook)

        # hooks can resolve tool definitions. Options: session setter or
        # agent method like session.register_tool_hook(tool_manager).
        return session

    async def destroy_session(self, session_uuid: uuid.UUID) -> bool:
        """Destroy a session by its UUID.

        Stops the session's event loop.
        """
        if session_uuid in self._sessions:
            session = self._sessions[session_uuid]
            # TODO: this is inconsistent: the persona package should not know about engine properties
            await session.stop()

            del self._sessions[session_uuid]
            return True
        return False

    def _on_before_tool_execution(
        self,
        session: Session,
        tool_call: ContentPart
    ) -> tuple:
        """Hook callback fired before each tool execution.

        Auto-approves tools listed in the session's auto_approve_tools.

        Returns:
            Tuple (allow: bool | None, message: str | None) - Returns
            (True, None) for auto-approved tools, ("pending", None) otherwise.
        """
        tool_name = tool_call.name
        if tool_name in session.auto_approve_tools:
            return (True, None)

        return ("pending", None)
