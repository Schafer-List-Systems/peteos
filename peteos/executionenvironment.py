from abc import ABC, abstractmethod
from collections import deque

from peteos.chatbot import ChatBot
from peteos.chathistory import ChatHistory
from peteos.message import Message
from peteos.toolmanager import ToolManager


class ExecutionEnvironment(ABC):
    """Environment for executing agent actions."""

    def __init__(
        self,
        chatbot: ChatBot,
        chat_history: ChatHistory,
        tool_manager: ToolManager
    ):
        """
        Initialize ExecutionEnvironment.

        Args:
            chatbot: The ChatBot instance to use (obligatory).
            chat_history: The ChatHistory instance to use (obligatory).
            tool_manager: The ToolManager instance to use.
        """
        self.tool_manager = tool_manager
        self.chat_history = chat_history
        self.chatbot = chatbot
        self._message_queue = deque()

    def queue_message(self, message: Message) -> None:
        """
        Add a message to the internal queue.

        Args:
            message: The message to queue.
        """
        self._message_queue.append(message)

    def get_chat_history(self) -> ChatHistory:
        """
        Get the internal chat history.

        Returns:
            The ChatHistory instance.
        """
        return self.chat_history

    @abstractmethod
    def run(self) -> None:
        """
        Run the agentic loop until the LLM responds with a final answer or the loop is interrupted.

        This method is abstract and must be overridden by derived classes.
        """
        pass
