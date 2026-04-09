import json

from peteos.chatbot import ChatBotManager, ChatHistory, Message, ContentPart
from peteos.executionenvironment import ExecutionEnvironment
from peteos.logger import get_logger
from peteos.role import Role
from peteos.toolmanager import ToolManager

_logger = get_logger(__name__)


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
            # response.data has format from translation: {text: "...", reasoning: "...", tool_calls: [...]}
            # Role is required - must be present (ChatBot ensures this)
            # If there's an error from the chatbot, skip appending and exit loop
            if "error" in response.data:
                _logger.warning("Chatbot returned error, skipping response: %s", response.data["error"])
                break
            assert "role" in response.data, f"ChatBot response missing 'role' field: {response.data.keys()}"

            # Generic mapping: response.data keys → ContentPart types
            # Any key becomes a ContentPart with part_type=key and the value
            content_parts = []
            for key, value in response.data.items():
                if key == "role":
                    continue  # Role is handled separately
                if value:  # Only add non-empty values
                    content_parts.append(ContentPart(part_type=key, **{key: value}))

            self.chat_history.append_message(Message(
                role=response.data["role"],
                content=content_parts
            ))

            # Check if response contains tool calls
            raw_tool_calls = response.data.get("tool_calls")
            # tool_calls is accumulated as a string during streaming - parse it as JSON
            if isinstance(raw_tool_calls, str):
                _logger.debug("raw_tool_calls type: %s", type(raw_tool_calls))
                _logger.debug("raw_tool_calls value: %s", raw_tool_calls[:200])
                try:
                    tool_calls_list = json.loads(raw_tool_calls)
                    if not isinstance(tool_calls_list, list):
                        tool_calls_list = []
                except (json.JSONDecodeError, TypeError) as e:
                    _logger.debug("JSON parse error: %s", e)
                    tool_calls_list = []
            else:
                tool_calls_list = raw_tool_calls if isinstance(raw_tool_calls, list) else []

            _logger.debug("Tool calls detected: %s", tool_calls_list)
            _logger.debug("response.data keys: %s", list(response.data.keys()))
            if "tool_calls" in response.data:
                _logger.debug("response.tool_calls type: %s", type(response.data["tool_calls"]))
                _logger.debug("response.tool_calls value: %s", response.data["tool_calls"])

            if tool_calls_list:
                # Track history length before tool execution
                history_length_before = len(self.chat_history.messages)
                _logger.debug("History length before tool execution: %d", history_length_before)
                # Execute tool calls
                for tool_call in tool_calls_list:
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name")
                        # Parse arguments from JSON string (accumulated during streaming)
                        args = json.loads(tool_call.get("arguments", "{}"))

                        tool = self.tool_manager.get_tool(tool_name)
                        if tool:
                            # Check if tool execution should be allowed
                            hook_result = await self._call_hooks("before_tool_execution", tool_call)
                            if hook_result is not None:
                                allow, message = hook_result
                                if not allow:
                                    # Tool execution disallowed by hook
                                    self.chat_history.append_message(Message(
                                        role="tool_result",
                                        content=[
                                            ContentPart(part_type="tool_result", name=tool_name, content=message),
                                            ContentPart(part_type="bool", success=False)
                                        ]
                                    ))
                                    await self._call_hooks("after_tool_execution", tool_call, message, False)
                                    continue

                            try:
                                result = tool.execute(**args)
                                self.chat_history.append_message(Message(
                                    role="tool_result",
                                    content=[
                                        ContentPart(part_type="tool_result", name=tool_name, content=str(result)),
                                        ContentPart(part_type="bool", success=True)
                                    ]
                                ))
                                await self._call_hooks("after_tool_execution", tool_call, str(result), True)
                            except Exception as e:
                                self.chat_history.append_message(Message(
                                    role="tool_result",
                                    content=[
                                        ContentPart(part_type="tool_result", name=tool_name, content=f"Error: {type(e).__name__}: {str(e)}"),
                                        ContentPart(part_type="bool", success=False)
                                    ]
                                ))
                                await self._call_hooks("after_tool_execution", tool_call, str(e), False)
                        else:
                            self.chat_history.append_message(Message(
                                role="tool_result",
                                content=[
                                    ContentPart(part_type="tool_result", name=tool_name, content=f"Error: Tool '{tool_name}' not found"),
                                    ContentPart(part_type="bool", success=False)
                                ]
                            ))
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
