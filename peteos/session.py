import json
from datetime import datetime
import uuid
from typing import Optional

from peteos.chatbotmanager import ChatBotManager
from peteos.chathistory import ChatHistory
from peteos.message import Message
from peteos.replexecutionenvironment import REPLExecutionEnvironment
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager


class Session:
    """A session with an execution environment."""

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
            chat_history: Optional ChatHistory instance. Creates one if None.
            session_uuid: Optional UUID. Generates one if None.
            execution_environment: Optional execution environment. Creates REPL one if None.
        """
        self.uuid = session_uuid if session_uuid is not None else uuid.uuid4()
        self.role = role
        self.chat_history = chat_history if chat_history is not None else ChatHistory()
        self.chatbot_manager = chatbot_manager

        self.execution_environment = execution_environment if execution_environment is not None else REPLExecutionEnvironment(
            chatbot_manager=chatbot_manager,
            chat_history=self.chat_history,
            tool_manager=tool_manager,
            role=role
        )

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
            Message(
                content=msg["content"],
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
