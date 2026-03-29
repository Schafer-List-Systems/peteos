from abc import ABC, abstractmethod

from peteos.chatbot import ChatBot
from peteos.chathistory import ChatHistory
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
        self._interrupt = False
        self._running = False

    @property
    def is_running(self) -> bool:
        """Check if the execution environment is currently running."""
        return self._running

    def set_interrupt(self) -> None:
        """Request interruption of the execution loop."""
        self._interrupt = True

    def clear_interrupt(self) -> None:
        """Clear the interrupt flag."""
        self._interrupt = False

    def get_chat_history(self) -> ChatHistory:
        """Get the internal chat history."""
        return self.chat_history

    @abstractmethod
    def run(self) -> None:
        """
        Run the agentic loop until the LLM responds with a final answer or the loop is interrupted.

        This method is abstract and must be overridden by derived classes.
        """
        pass
