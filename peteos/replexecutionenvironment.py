from peteos.chatbot import ChatBot
from peteos.chathistory import ChatHistory
from peteos.executionenvironment import ExecutionEnvironment
from peteos.toolmanager import ToolManager


class REPLExecutionEnvironment(ExecutionEnvironment):
    """REPL (Read-Eval-Print Loop) execution environment."""

    def __init__(
        self,
        chatbot: ChatBot,
        chat_history: ChatHistory,
        tool_manager: ToolManager
    ):
        """
        Initialize REPLExecutionEnvironment.

        Args:
            chatbot: The ChatBot instance to use.
            chat_history: The ChatHistory instance to use.
            tool_manager: The ToolManager instance to use.
        """
        super().__init__(chatbot=chatbot, chat_history=chat_history, tool_manager=tool_manager)

    def run(self) -> None:
        """
        Run the REPL loop.

        Reads input, processes it through the chatbot, and appends output to ChatHistory.
        """
        pass
