from peteos.chatbot import ChatBot
from peteos.chathistory import ChatHistory
from peteos.executionenvironment import ExecutionEnvironment
from peteos.message import Message
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

    async def run(self) -> None:
        """
        Run the REPL loop.

        Reads input, processes it through the chatbot, and appends output to ChatHistory.
        Loops until final answer is received or interrupt flag is set.
        """
        self._running = True
        try:
            while not self._interrupt:
                # Send chat history to chatbot
                response = await self.chatbot.send_message(
                    self.chat_history,
                    streaming=True
                )

                # Collect accumulated response with interrupt checks
                interrupted = False
                async for _ in response:
                    if self._interrupt:
                        interrupted = True
                        break

                if interrupted:
                    # Request dropped mid-stream, exit loop
                    break

                # Check if we're interrupted
                if self._interrupt:
                    break

                # Extract response data
                response_data = response.data
                text = response_data.get("text", "")
                reasoning = response_data.get("reasoning", "")
                tool_calls = response_data.get("tool_calls")

                # Append reasoning as separate field if present
                if reasoning:
                    self.chat_history.append_message(
                        Message(content={
                            "role": "assistant",
                            "reasoning": reasoning
                        })
                    )

                # tool_calls is already a list from response.data["tool_calls"]
                tool_calls_list = tool_calls if isinstance(tool_calls, list) else []

                if tool_calls_list:
                    # Append the response containing tool calls as assistant message
                    self.chat_history.append_message(
                        Message(content={
                            "role": "assistant",
                            "content": text
                        })
                    )
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
                    # Final answer - append and exit loop
                    self.chat_history.append_message(
                        Message(content={
                            "role": "assistant",
                            "content": text
                        })
                    )
                    break
        finally:
            self._running = False
