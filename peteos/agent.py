import uuid
from typing import Any, Dict, List, Set

from peteos.channel import Channel
from peteos.chatbot import ChatBotManager, Message
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.session import Session
from peteos.toolmanager import ToolManager


class Agent:
    """Manages concurrent sessions and channels.

    The agent serves as a hub connecting channels to sessions. Commands are
    handled at the agent level, and outgoing messages from sessions are
    forwarded to subscribed channels via hooks.

    Each channel can select an active session. Messages from channels are
    forwarded to the active session. Session responses are sent to all
    channels that have the session selected as active.
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
        self._channels: Dict[str, Channel] = {}
        self._session_channels: Dict[uuid.UUID, Set[Channel]] = {}

    def register_channel(self, channel: Channel) -> None:
        """Register a channel with the agent.

        Args:
            channel: The channel to register.
        """
        self._channels[channel.name] = channel

    def deregister_channel(self, name: str) -> None:
        """Deregister a channel from the agent.

        Args:
            name: The channel name to deregister.
        """
        if name in self._channels:
            del self._channels[name]

    def get_channel(self, name: str) -> Channel | None:
        """Get a channel by name.

        Args:
            name: The channel name.

        Returns:
            The Channel instance, or None if not found.
        """
        return self._channels.get(name)

    def list_channels(self) -> Dict[str, Channel]:
        """List all registered channels.

        Returns:
            A dictionary mapping channel names to Channel instances.
        """
        return dict(self._channels)

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
            if session_uuid in self._session_channels:
                del self._session_channels[session_uuid]
            return True
        return False

    def _on_before_tool_execution(self, session_uuid: uuid.UUID, tool_call: dict) -> tuple:
        """Hook callback fired before each tool execution.

        Args:
            session_uuid: The UUID of the session making the tool call.
            tool_call: Dict containing 'name' and 'arguments' of the tool.

        Returns:
            Tuple (allow: bool, message: str) - For now, always allows.
        """
        tool_name = tool_call.get("name", "unknown")
        args = tool_call.get("arguments", {})
        message = f"Tool called: {tool_name} with args: {args}"
        self._notify_channels(session_uuid, f"[Agent] {message}")
        # For now, always allow tool execution
        return (True, "")

    def _on_after_tool_execution(self, session_uuid: uuid.UUID, tool_call: dict, result: str, success: bool) -> None:
        """Hook callback fired after tool execution completes.

        Args:
            session_uuid: The UUID of the session that executed the tool.
            tool_call: Dict containing 'name' and 'arguments' of the tool.
            result: The tool result as a string.
            success: Whether the tool execution succeeded.
        """
        tool_name = tool_call.get("name", "unknown")
        status = "success" if success else "failed"
        message = f"Tool '{tool_name}' {status}: {result}"
        self._notify_channels(session_uuid, f"[Agent] {message}")

    def _on_before_loop_continue(self, session_uuid: uuid.UUID, delta_messages: List[Message]) -> None:
        """Hook callback fired when the loop continues after tool calls.

        Args:
            session_uuid: The UUID of the session.
            delta_messages: List of messages added during this iteration.
        """
        # Forward tool results and intermediate messages to channels
        for msg in delta_messages:
            content = msg.content
            role = content.get("role", "unknown")
            if role == "tool":
                tool_name = content.get("name", "unknown")
                tool_content = content.get("content", "")
                success = content.get("success", False)
                status = "success" if success else "failed"
                message = f"Tool '{tool_name}' result ({status}): {tool_content}"
                self._notify_channels(session_uuid, f"[Agent] {message}")
            elif role == "assistant":
                text = content.get("text", content.get("content", ""))
                if text:
                    self._notify_channels(session_uuid, f"[Agent] {text}")

    def _on_before_loop_exit(self, session_uuid: uuid.UUID, reason: str) -> None:
        """Hook callback fired when the loop exits.

        Args:
            session_uuid: The UUID of the session.
            reason: The reason for exit (e.g., "final_answer" or "interrupt").
        """
        # Forward the final answer to channels
        session = self.get_session(session_uuid)
        if session:
            history = session.chat_history.messages
            if history:
                last_msg = history[-1]
                content = last_msg.content
                role = content.get("role", "unknown")
                if role == "assistant":
                    text = content.get("text", content.get("content", ""))
                    if text:
                        self._notify_channels(session_uuid, f"[Agent] {text}")
                    elif "reasoning" in content:
                        reasoning = content.get("reasoning", "")
                        self._notify_channels(session_uuid,
                                              f"[Agent] Reasoning: {reasoning}")

    def _notify_channels(self, session_uuid: uuid.UUID, message: str) -> None:
        """Notify all channels subscribed to a session.

        Args:
            session_uuid: The UUID of the session.
            message: The message to send.
        """
        channels = self._session_channels.get(session_uuid, set())
        for channel in channels:
            channel.send(message)
