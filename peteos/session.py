import uuid
from typing import Optional

from peteos.chatbot import ChatBot
from peteos.chatbotmanager import ChatBotManager
from peteos.chathistory import ChatHistory
from peteos.executionenvironment import ExecutionEnvironment
from peteos.role import Role
from peteos.toolmanager import ToolManager


class Session:
    """A session with an execution environment."""

    def __init__(
        self,
        role: Role,
        tool_manager: ToolManager,
        chatbot_manager: ChatBotManager,
        chat_history: Optional[ChatHistory] = None,
        session_uuid: Optional[uuid.UUID] = None
    ):
        """
        Initialize Session.

        Args:
            role: The Role instance to use (obligatory).
            tool_manager: The ToolManager instance to use.
            chatbot_manager: The ChatBotManager instance to use.
            chat_history: Optional ChatHistory instance. Creates one if None.
            session_uuid: Optional UUID. Generates one if None.
        """
        self.uuid = session_uuid if session_uuid is not None else uuid.uuid4()
        self.role = role
        self.chat_history = chat_history if chat_history is not None else ChatHistory()
        self.chatbot_manager = chatbot_manager
        self.execution_environment = ExecutionEnvironment(
            chatbot_manager=chatbot_manager,
            chat_history=self.chat_history,
            tool_manager=tool_manager,
            role=role
        )

    @staticmethod
    def load_from_string(data: str) -> "Session":
        """
        Creates a new session from a string.

        The UUID is extracted from the loaded data.

        Args:
            data: The string data to load the session from.

        Returns:
            A new Session instance.
        """
        pass

    @staticmethod
    def load_from_file(file_path: str) -> "Session":
        """
        Creates a new session from a file.

        The UUID is extracted from the loaded data.

        Args:
            file_path: Path to the file to load the session from.

        Returns:
            A new Session instance.
        """
        pass
