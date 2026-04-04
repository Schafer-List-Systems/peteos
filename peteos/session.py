import asyncio
import json
from collections import deque
from datetime import datetime
import uuid
from typing import Optional

from peteos.chatbot import ChatBotManager, ChatHistory, Message, ContentPart
from peteos.replexecutionenvironment import REPLExecutionEnvironment
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager


class Session:
    """A session with an execution environment."""

    @staticmethod
    def _initialize_chat_history(role: Role, tool_manager: ToolManager) -> ChatHistory:
        """
        Initialize chat history with role system prompt and tool list.

        This is a central place for creating chat history with context from
        the role system prompt and available tools. The system prompt and
        tool list are prepended to the chat history as the first messages.

        Args:
            role: The Role instance containing system prompt and required tools.
            tool_manager: The ToolManager instance with available tools.

        Returns:
            ChatHistory with system prompt and tool list prepended.
        """
        chat_history = ChatHistory()

        # Add system prompt from role
        if role.system_prompt:
            chat_history.append_message(Message(
                role="system",
                content=[ContentPart(part_type="text", text=role.system_prompt)]
            ))

        # Add tool definitions from tool manager
        tool_list = tool_manager.get_tool_list()
        for tool in tool_list:
            chat_history.append_message(Message(
                role="tool",
                content=[ContentPart(
                    part_type="tool",
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters
                )]
            ))

        return chat_history

    def __init__(
        self,
        role: Role,
        tool_manager: ToolManager,
        chatbot_manager: ChatBotManager,
        chat_history: Optional[ChatHistory] = None,
        session_uuid: Optional[uuid.UUID] = None,
        execution_environment: Optional[REPLExecutionEnvironment] = None
    ):
        """
        Initialize Session.

        Note: Use load_from_json() or load_from_file() to create Session instances.
        The constructor is intentionally kept flexible for internal use.

        Args:
            role: The Role instance to use (obligatory).
            tool_manager: The ToolManager instance to use.
            chatbot_manager: The ChatBotManager instance to use.
            chat_history: Optional ChatHistory instance. Creates one with system
                prompt and tool list if None.
            session_uuid: Optional UUID. Generates one if None.
            execution_environment: Optional execution environment. Creates REPL one if None.
        """
        self.uuid = session_uuid if session_uuid is not None else uuid.uuid4()
        self.role = role
        self.chat_history = chat_history if chat_history is not None else self._initialize_chat_history(role, tool_manager)
        self.chatbot_manager = chatbot_manager

        self.execution_environment = execution_environment if execution_environment is not None else REPLExecutionEnvironment(
            chatbot_manager=chatbot_manager,
            chat_history=self.chat_history,
            tool_manager=tool_manager,
            role=role
        )

        self._message_queue: deque[Message] = deque()
        self._queue_lock = asyncio.Lock()

    async def queue_message(self, message: Message) -> None:
        """
        Queue a message for processing.

        If the execution environment is running, this function:
        1. Adds the message to the queue
        2. Interrupts the execution environment
        3. Waits for it to stop
        4. Drains ALL queued messages to chat_history
        5. Restarts the execution environment

        If the execution environment is NOT running, the message is added
        directly to chat_history and the env is started.

        Uses a lock to ensure thread-safe concurrent enqueues.

        Args:
            message: The message to queue.
        """
        async with self._queue_lock:
            # Add message to queue
            self._message_queue.append(message)

            if self.execution_environment.is_running:
                # Interrupt
                self.execution_environment.set_interrupt()

            # Drain ALL queued messages to chat_history
            while self._message_queue:
                msg = self._message_queue.popleft()
                self.chat_history.append_message(msg)

            # Start/restart execution env
            await self.execution_environment.run()

    @staticmethod
    def load_from_json(
        json_data: dict,
        chatbot_manager: ChatBotManager,
        role_manager: RoleManager,
        tool_manager: ToolManager
    ) -> "Session":
        """
        Creates a new session from a JSON dict.

        Args:
            json_data: JSON dict with session data (uuid, role name, chat_history).
            chatbot_manager: ChatBotManager instance for session construction.
            role_manager: RoleManager instance to lookup Role by name.
            tool_manager: ToolManager instance to validate required tools.

        Returns:
            A new Session instance.

        Raises:
            ValueError: If required tools are missing from tool_manager.
        """
        uuid_str = json_data.get("uuid")
        role_name = json_data["role"]
        chat_history_data = json_data.get("chat_history", [])

        role = role_manager.get_role(role_name)
        if role is None:
            raise ValueError(f"Role '{role_name}' not found in RoleManager")

        for tool_name in role.required_tools:
            if tool_manager.get_tool(tool_name) is None:
                raise ValueError(
                    f"Role '{role_name}' requires tool '{tool_name}', "
                    f"but it's not registered in tool_manager"
                )

        messages = [
            Message.from_dict(
                msg["content"],
                creation_timestamp=datetime.fromisoformat(msg["creation_timestamp"]) if "creation_timestamp" in msg else None,
                message_id=msg.get("id")
            )
            for msg in chat_history_data
        ]
        chat_history = ChatHistory()
        for msg in messages:
            chat_history.append_message(msg)

        session_uuid = uuid.UUID(uuid_str) if uuid_str else None

        # Determine execution environment from role config (default to REPL)
        env = REPLExecutionEnvironment(
            chatbot_manager=chatbot_manager,
            chat_history=chat_history,
            tool_manager=tool_manager,
            role=role
        )

        return Session(
            role=role,
            tool_manager=tool_manager,
            chatbot_manager=chatbot_manager,
            chat_history=chat_history,
            session_uuid=session_uuid,
            execution_environment=env
        )

    @staticmethod
    def load_from_file(
        file_path: str,
        chatbot_manager: ChatBotManager,
        role_manager: RoleManager,
        tool_manager: ToolManager
    ) -> "Session":
        """
        Creates a new session from a JSON file.

        Args:
            file_path: Path to the JSON file to load the session from.
            chatbot_manager: ChatBotManager instance for session construction.
            role_manager: RoleManager instance to lookup Role by name.
            tool_manager: ToolManager instance to validate required tools.

        Returns:
            A new Session instance.
        """
        with open(file_path, "r") as f:
            json_data = json.load(f)
        return Session.load_from_json(
            json_data,
            chatbot_manager,
            role_manager,
            tool_manager
        )
