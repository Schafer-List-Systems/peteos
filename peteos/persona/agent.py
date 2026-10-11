"""Agent - Central hub for session management and notifications."""

from peteos.utils import get_logger, json

from peteos.conversation.system_prompt_message import SystemPromptMessage
from peteos.conversation.session import Session
from peteos.conversation.tool_definitions_message import ToolDefinitionsMessage


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

    Session storage is the responsibility of the caller, not the Agent class.
    """

    def __init__(
        self,
        role: Role,
        tool_manager: ToolManager,
    ):
        self._role = role
        self._tool_manager = tool_manager

    async def create_session(self) -> Session:
        """Create a new session for this agent.

        The session resolves its own session directory via ConfigManager.
        The caller is responsible for storing the session reference.
        """
        system_prompt_msg = SystemPromptMessage.create(self._role.system_prompt)
        tool_defs_msg = ToolDefinitionsMessage()
        session = Session.create(system_prompt_msg, tool_defs_msg)

        session.register_hook(tool_defs_msg, ToolDefinitionsMessage.TOOL_LIST_HOOK_NAME, self._tool_list_hook)
        session.register_hook(tool_defs_msg, ToolDefinitionsMessage.TOOL_FILTER_HOOK_NAME, self._tool_filter_hook)

        # TODO: Register hooks (before_tool_execution, after_tool_execution,
        # before_notification_publish) on the environment

        return session

    @property
    def role(self) -> Role:
        """Return this agent's role."""
        return self._role

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

