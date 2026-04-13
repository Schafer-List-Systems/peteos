"""Agent - Central hub for message routing and event loop management."""

import asyncio
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, AsyncIterator

from peteos.channel import Channel
from peteos.chatbot import ChatBotManager, Message
from peteos.logger import get_logger
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.session import Session
from peteos.toolmanager import ToolManager

_logger = get_logger(__name__)


class Agent:
    """Central hub for message routing and session management.

    The Agent manages concurrent sessions and channels with full decoupling:
    - Channels post messages to the Agent's per-session message queues
    - Agent's event loop processes messages and routes them to sessions
    - Sessions trigger hooks that publish notifications to per-channel queues
    - Channels consume notifications from their own notification queues

    The event loop runs in a background thread to prevent blocking user input
    from blocking session processing.

    Thread Model:
        - Agent event loop: runs in dedicated background thread
        - Channels: each runs in its own thread (shell blocking input, REST async)

    Threading Guarantees:
        - All queue operations are thread-safe via asyncio.Queue
        - No shared mutable state between components
        - Each Agent instance is completely isolated

    Example:
        >>> agent = Agent(role_manager, chatbot_manager, tool_manager)
        >>> await agent.start()
        >>> session = agent.create_session("test")
        >>> shell = InteractiveShellChannel("shell", agent)
        >>> await shell.start()
        >>> shell.send_message_to_session(session.uuid, "Hello!")
        >>> # ... process messages ...
        >>> await shell.stop()
        >>> await agent.stop()
    """

    def __init__(
        self,
        role_manager: RoleManager,
        chatbot_manager: ChatBotManager,
        tool_manager: ToolManager
    ):
        """Initialize Agent.

        Args:
            role_manager: The RoleManager instance to use.
            chatbot_manager: The ChatBotManager instance to use.
            tool_manager: The ToolManager instance to use.
        """
        self._role_manager = role_manager
        self._chatbot_manager = chatbot_manager
        self._tool_manager = tool_manager

        # Session management
        self._sessions: Dict[uuid.UUID, Session] = {}

        # Per-session message queues for incoming messages
        self._message_queues: Dict[uuid.UUID, asyncio.Queue] = {}

        # Per-channel notification queues for outgoing messages
        # Key: (channel_name, session_uuid) -> Queue[str]
        self._notification_queues: Dict[Tuple[str, uuid.UUID], asyncio.Queue] = {}

        # Track which channels are subscribed to which sessions
        self._session_channels: Dict[uuid.UUID, Set[Channel]] = {}

        # Track registered channels (for Channel base class compatibility)
        self._channels: Dict[str, Channel] = {}

        # Event loop lifecycle
        self._loop_task: Optional[asyncio.Task] = None
        self._running = False

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

    async def start(self) -> None:
        """Start the Agent's event loop in a background thread.

        The event loop processes message queues and handles hook notifications.
        Must be called before creating sessions or posting messages.

        Raises:
            RuntimeError: If agent is already running.
        """
        if self._running:
            raise RuntimeError("Agent is already running")

        self._running = True
        self._loop_task = asyncio.create_task(self._main_loop())

    async def stop(self) -> None:
        """Stop the Agent's event loop gracefully.

        Waits for pending operations to complete before shutting down.
        Should be called after all channels have been stopped.

        Raises:
            RuntimeError: If agent is not running.
        """
        if not self._running:
            raise RuntimeError("Agent is not running")

        self._running = False

        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

    async def _main_loop(self) -> None:
        """Main event loop that processes messages and notifications.

        This loop runs in a background thread and:
        1. Polls message queues for pending messages to process
        2. Routes messages to their respective sessions
        3. Handles hook notifications from sessions
        4. Publishes notifications to channel queues
        """
        while self._running:
            # Process pending messages
            for session_uuid, queue in list(self._message_queues.items()):
                if not queue.empty():
                    try:
                        message = queue.get_nowait()
                        session = self._sessions.get(session_uuid)
                        if session:
                            await session.queue_message(message)
                    except asyncio.QueueEmpty:
                        pass

            # Small sleep to prevent busy waiting
            await asyncio.sleep(0.01)

    def create_session(self, role_name: str) -> Session:
        """Create a new session with the specified role.

        Creates per-session message queue and registers hooks for notifications.

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

        # Create message queue for this session
        self._message_queues[session.uuid] = asyncio.Queue()

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

        Cleans up message queue and notification queues for this session.

        Args:
            session_uuid: The UUID of the session to destroy.

        Returns:
            True if the session was found and destroyed, False otherwise.
        """
        if session_uuid in self._sessions:
            # Clean up message queue
            if session_uuid in self._message_queues:
                del self._message_queues[session_uuid]

            # Clean up all notification queues for this session
            keys_to_delete = [
                key for key in self._notification_queues
                if key[1] == session_uuid
            ]
            for key in keys_to_delete:
                del self._notification_queues[key]

            # Clean up channel tracking
            if session_uuid in self._session_channels:
                del self._session_channels[session_uuid]

            del self._sessions[session_uuid]
            return True
        return False

    def post_message(self, session_uuid: uuid.UUID, message: Message) -> None:
        """Post a message to a session's message queue.

        Non-blocking queue put. If the queue is full, this will block
        until space is available (default behavior of asyncio.Queue).

        Args:
            session_uuid: The UUID of the target session.
            message: The message to post.

        Raises:
            KeyError: If the session does not exist.
        """
        queue = self._message_queues.get(session_uuid)
        if queue is None:
            raise KeyError(f"Session {session_uuid} not found")
        queue.put_nowait(message)

    def subscribe_notifications(
        self,
        channel_name: str,
        session_uuid: uuid.UUID
    ) -> AsyncIterator[str]:
        """Subscribe to notifications for a session from a channel.

        Returns an async iterator that yields notification messages.
        The iterator runs until the Agent is stopped.

        Args:
            channel_name: The name of the subscribing channel.
            session_uuid: The UUID of the session to subscribe to.

        Yields:
            Notification messages from the session.
        """
        queue_key = (channel_name, session_uuid)

        # Create notification queue if it doesn't exist
        if queue_key not in self._notification_queues:
            self._notification_queues[queue_key] = asyncio.Queue()

        # Track this channel as subscribed to this session
        if session_uuid not in self._session_channels:
            self._session_channels[session_uuid] = set()
        self._session_channels[session_uuid].add(
            Channel.get_by_name(channel_name) if Channel.get_by_name(channel_name) else None
        )

        # Create async generator
        async def notification_generator() -> AsyncIterator[str]:
            while self._running:
                try:
                    msg = await self._notification_queues[queue_key].get()
                    yield msg
                except KeyError:
                    break

        return notification_generator()

    def unsubscribe_notifications(
        self,
        channel_name: str,
        session_uuid: uuid.UUID
    ) -> None:
        """Unsubscribe from notifications for a session.

        Cleans up the notification queue for this channel/session pair.

        Args:
            channel_name: The name of the unsubscribing channel.
            session_uuid: The UUID of the session to unsubscribe from.
        """
        queue_key = (channel_name, session_uuid)
        if queue_key in self._notification_queues:
            del self._notification_queues[queue_key]

        # Remove channel from session's subscribed channels
        if session_uuid in self._session_channels:
            channel = Channel.get_by_name(channel_name)
            if channel and channel in self._session_channels[session_uuid]:
                self._session_channels[session_uuid].remove(channel)

    def _on_before_tool_execution(
        self,
        session_uuid: uuid.UUID,
        tool_call: dict
    ) -> tuple:
        """Hook callback fired before each tool execution.

        Publishes notification to all subscribed channels.

        Args:
            session_uuid: The UUID of the session making the tool call.
            tool_call: Dict containing 'name' and 'arguments' of the tool.

        Returns:
            Tuple (allow: bool, message: str) - For now, always allows.
        """
        tool_name = tool_call.get("name", "unknown")
        args = tool_call.get("arguments", {})
        message = f"[Agent] Tool called: {tool_name} with args: {args}"
        self._publish_notification(session_uuid, message)
        return (True, "")

    def _on_after_tool_execution(
        self,
        session_uuid: uuid.UUID,
        tool_call: dict,
        result: str,
        success: bool
    ) -> None:
        """Hook callback fired after tool execution completes.

        Publishes notification to all subscribed channels.

        Args:
            session_uuid: The UUID of the session that executed the tool.
            tool_call: Dict containing 'name' and 'arguments' of the tool.
            result: The tool result as a string.
            success: Whether the tool execution succeeded.
        """
        tool_name = tool_call.get("name", "unknown")
        status = "success" if success else "failed"
        message = f"[Agent] Tool '{tool_name}' {status}: {result}"
        self._publish_notification(session_uuid, message)

    def _on_before_loop_continue(
        self,
        session_uuid: uuid.UUID,
        delta_messages: List[Message]
    ) -> None:
        """Hook callback fired when the loop continues after tool calls.

        Publishes tool results and intermediate messages to all subscribed
        channels. Tool results are already reported by after_tool_execution,
        so we skip them here to avoid duplicates.

        Args:
            session_uuid: The UUID of the session.
            delta_messages: List of messages added during this iteration.
        """
        for msg in delta_messages:
            role = msg.role
            if role == "tool_result":
                # Tool results already reported by _on_after_tool_execution
                continue
            elif role == "assistant":
                text = ""
                for part in msg.content:
                    if part.type == "text" and part.data.get("text"):
                        text = part.data.get("text")
                        break
                if text:
                    self._publish_notification(session_uuid, f"[Agent] {text}")

    def _on_before_loop_exit(
        self,
        session_uuid: uuid.UUID,
        reason: str
    ) -> None:
        """Hook callback fired when the loop exits.

        Publishes the final answer to all subscribed channels.

        Args:
            session_uuid: The UUID of the session.
            reason: The reason for exit (e.g., "final_answer" or "interrupt").
        """
        session = self.get_session(session_uuid)
        if session:
            history = session.chat_history.messages
            if history:
                last_msg = history[-1]
                if last_msg.role == "assistant":
                    text = ""
                    reasoning = ""
                    for part in last_msg.content:
                        if part.type == "text" and part.data.get("text"):
                            text = part.data.get("text")
                        elif part.type == "reasoning" and part.data.get("reasoning"):
                            reasoning = part.data.get("reasoning")
                    if text:
                        self._publish_notification(session_uuid, f"[Agent] {text}")
                    elif reasoning:
                        self._publish_notification(
                            session_uuid,
                            f"[Agent] Reasoning: {reasoning}"
                        )

    def _publish_notification(
        self,
        session_uuid: uuid.UUID,
        message: str
    ) -> None:
        """Publish a notification to all subscribed channels.

        Thread-safe notification broadcasting to all channels subscribed
        to this session.

        Args:
            session_uuid: The UUID of the session.
            message: The notification message to publish.
        """
        channels = self._session_channels.get(session_uuid, set())
        for channel in channels:
            if channel:
                queue_key = (channel.name, session_uuid)
                queue = self._notification_queues.get(queue_key)
                if queue:
                    try:
                        queue.put_nowait(message)
                    except Exception:
                        # Queue might be closed, ignore
                        pass
