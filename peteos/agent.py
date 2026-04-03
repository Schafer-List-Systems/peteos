import uuid
from typing import Dict

from peteos.chatbot import ChatBotManager
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.session import Session
from peteos.toolmanager import ToolManager


class Agent:
    """Manages concurrent sessions.

    Each session runs in its own thread. The agent also manages the channels
    to the sessions.
    """

    def __init__(
        self,
        role_manager: RoleManager,
        chatbot_manager: ChatBotManager,
        tool_manager: ToolManager
    ):
        """
        Initialize Agent.

        Args:
            role_manager: The RoleManager instance to use (obligatory).
            chatbot_manager: The ChatBotManager instance to use (obligatory).
            tool_manager: The ToolManager instance to use (obligatory).
        """
        self._role_manager = role_manager
        self._chatbot_manager = chatbot_manager
        self._tool_manager = tool_manager
        self._sessions: Dict[uuid.UUID, Session] = {}

    def create_session(self, role_name: str) -> Session:
        """Create a new session with the specified role.

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
        return session

    def get_session(self, session_uuid: uuid.UUID) -> Session | None:
        """Get a session by its UUID.

        Args:
            session_uuid: The UUID of the session to retrieve.

        Returns:
            The Session instance if found, None otherwise.
        """
        return self._sessions.get(session_uuid)

    def list_sessions(self) -> Dict[uuid.UUID, Session]:
        """List all sessions.

        Returns:
            A dictionary mapping UUIDs to Session instances.
        """
        return dict(self._sessions)

    def destroy_session(self, session_uuid: uuid.UUID) -> bool:
        """Destroy a session by its UUID.

        Args:
            session_uuid: The UUID of the session to destroy.

        Returns:
            True if the session was found and destroyed, False otherwise.
        """
        if session_uuid in self._sessions:
            del self._sessions[session_uuid]
            return True
        return False
