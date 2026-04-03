import asyncio
from abc import ABC, abstractmethod

from peteos.chatbot import ChatBot, ChatBotManager, ChatHistory
from peteos.role import Role
from peteos.toolmanager import ToolManager


class ExecutionEnvironment(ABC):
    """Environment for executing agent actions."""

    def __init__(
        self,
        chatbot_manager: ChatBotManager,
        chat_history: ChatHistory,
        tool_manager: ToolManager,
        role: Role
    ):
        """
        Initialize ExecutionEnvironment.

        Args:
            chatbot_manager: The ChatBotManager instance to use (obligatory).
            chat_history: The ChatHistory instance to use (obligatory).
            tool_manager: The ToolManager instance to use (obligatory).
            role: The Role instance to use (for model_regex and future properties).
        """
        self.tool_manager = tool_manager
        self.chat_history = chat_history
        self.chatbot_manager = chatbot_manager
        self.role = role
        self._chatbot: ChatBot = self._select_chatbot()
        self._interrupt = False
        self._completion_signal: asyncio.Event = asyncio.Event()
        self._completion_signal.set()  # Start as signaled (not running)

    @property
    def chatbot(self) -> ChatBot:
        """Get current ChatBot, selecting from manager if available."""
        return self._chatbot

    def _select_chatbot(self) -> ChatBot:
        """Select a ChatBot from the manager based on role.model."""
        chatbots = self.chatbot_manager.list_chatbots(self.role.model)
        if not chatbots:
            raise ValueError(
                f"No ChatBot found matching model pattern '{self.role.model}' "
                f"for role '{self.role.name}'"
            )
        return chatbots[0][1]

    @property
    def is_running(self) -> bool:
        """Check if the execution environment is currently running."""
        return not self._completion_signal.is_set()

    def set_interrupt(self) -> None:
        """Request interruption of the execution loop."""
        self._interrupt = True

    def clear_interrupt(self) -> None:
        """Clear the interrupt flag."""
        self._interrupt = False

    def get_chat_history(self) -> ChatHistory:
        """Get the internal chat history."""
        return self.chat_history

    async def run(self) -> None:
        """
        Run the agentic loop until the LLM responds with a final answer or the loop is interrupted.

        This is a concrete implementation that wraps the abstract _run_impl() method
        to provide completion signaling. Derived classes should override _run_impl().
        """
        self._completion_signal.clear()
        try:
            await self._run_impl()
        finally:
            self._completion_signal.set()

    @abstractmethod
    async def _run_impl(self) -> None:
        """
        Actual implementation of the agentic loop.

        This method must be overridden by derived classes.
        """
        pass

    async def wait_for_stop(self) -> None:
        """
        Wait for the execution loop to complete.

        This blocks until the loop finishes (either naturally or via interruption).
        """
        await self._completion_signal.wait()
