from peteos.chatbot import ChatBotManager, ChatHistory, Message
from peteos.executionenvironment import ExecutionEnvironment
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
                await self._call_hooks("before_loop_exit", "interrupt")
                break

            # Append the full response as a Message to ChatHistory
            self.chat_history.append_message(Message(content=response.data))

            # Check if response contains tool calls
            tool_calls = response.data.get("tool_calls")
            tool_calls_list = tool_calls if isinstance(tool_calls, list) else []

            if tool_calls_list:
                # Track history length before tool execution
                history_length_before = len(self.chat_history.messages)
                # Execute tool calls
                for tool_call in tool_calls_list:
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name")
                        args = tool_call.get("arguments", {})

                        tool = self.tool_manager.get_tool(tool_name)
                        if tool:
                            # Check if tool execution should be allowed
                            hook_result = await self._call_hooks("before_tool_execution", tool_call)
                            if hook_result is not None:
                                allow, message = hook_result
                                if not allow:
                                    # Tool execution disallowed by hook
                                    self.chat_history.append_message(Message(content={
                                        "role": "tool",
                                        "name": tool_name,
                                        "content": message,
                                        "success": False
                                    }))
                                    await self._call_hooks("after_tool_execution", tool_call, message, False)
                                    continue

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
                                await self._call_hooks("after_tool_execution", tool_call, str(result), True)
                            except Exception as e:
                                self.chat_history.append_message(
                                    Message(content={
                                        "role": "tool",
                                        "name": tool_name,
                                        "content": f"Error: {type(e).__name__}: {str(e)}",
                                        "success": False
                                    })
                                )
                                await self._call_hooks("after_tool_execution", tool_call, str(e), False)
                        else:
                            self.chat_history.append_message(
                                Message(content={
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": f"Error: Tool '{tool_name}' not found",
                                    "success": False
                                })
                            )
                            await self._call_hooks("after_tool_execution", tool_call, f"Error: Tool '{tool_name}' not found", False)
                # Loop continues - sends history with tool results back to LLM
                # Track delta messages (messages added during this iteration)
                new_message_count = len(self.chat_history.messages) - history_length_before
                delta_messages = self.chat_history.messages[-new_message_count:] if new_message_count > 0 else []
                hook_result = await self._call_hooks("before_loop_continue", delta_messages)
                if hook_result is not None:
                    should_exit, reason = hook_result
                    if should_exit:
                        await self._call_hooks("before_loop_exit", reason)
                        break
            else:
                # Check if response has text content
                text = response.data.get("text")
                if text is not None and text:
                    # Final answer - exit loop
                    await self._call_hooks("before_loop_exit", "final_answer")
                    break
                # No tool calls and no text - only reasoning, continue loop
                await self._call_hooks("before_loop_continue", [])
