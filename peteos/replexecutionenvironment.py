from peteos.chatbotmanager import ChatBotManager
from peteos.chathistory import ChatHistory
from peteos.executionenvironment import ExecutionEnvironment
from peteos.message import Message
from peteos.role import Role
from peteos.toolmanager import ToolManager


class REPLExecutionEnvironment(ExecutionEnvironment):
    """REPL (Read-Eval-Print Loop) execution environment."""

    def __init__(
        self,
        chatbot_manager: ChatBotManager,
        chat_history: ChatHistory,
        tool_manager: ToolManager,
        role: Role
    ):
        """
        Initialize REPLExecutionEnvironment.

        Args:
            chatbot_manager: The ChatBotManager instance to use.
            chat_history: The ChatHistory instance to use.
            tool_manager: The ToolManager instance to use.
            role: The Role instance to use.
        """
        super().__init__(
            chatbot_manager=chatbot_manager,
            chat_history=chat_history,
            tool_manager=tool_manager,
            role=role
        )

    async def _run_impl(self) -> None:
        """
        Run the REPL loop.

        Reads input, processes it through the chatbot, and appends output to ChatHistory.
        Loops until final answer is received or interrupt flag is set.
        """
        while not self._interrupt:
            # Send chat history to chatbot
            response = await self.chatbot.send_message(
                self.chat_history,
                streaming=True
            )

            # Collect accumulated response with interrupt checks
            async for _ in response:
                if self._interrupt:
                    break

            if self._interrupt:
                # Request dropped mid-stream, exit loop
                break

            # Append the full response as a Message to ChatHistory
            self.chat_history.append_message(Message(content=response.data))

            # Check if response contains tool calls
            tool_calls = response.data.get("tool_calls")
            tool_calls_list = tool_calls if isinstance(tool_calls, list) else []

            if tool_calls_list:
                # Execute tool calls
                for tool_call in tool_calls_list:
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name")
                        args = tool_call.get("arguments", {})

                        tool = self.tool_manager.get_tool(tool_name)
                        if tool:
                            try:
                                result = tool.execute(**args)
                                self.chat_history.append_message(
                                    Message(content={
                                        "role": "tool",
                                        "name": tool_name,
                                        "content": str(result),
                                        "success": True
                                    })
                                )
                            except Exception as e:
                                self.chat_history.append_message(
                                    Message(content={
                                        "role": "tool",
                                        "name": tool_name,
                                        "content": f"Error: {type(e).__name__}: {str(e)}",
                                        "success": False
                                    })
                                )
                        else:
                            self.chat_history.append_message(
                                Message(content={
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": f"Error: Tool '{tool_name}' not found",
                                    "success": False
                                })
                            )
                # Loop continues - sends history with tool results back to LLM
            else:
                # Final answer - exit loop
                break
