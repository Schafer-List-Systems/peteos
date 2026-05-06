import json

import inspect
from typing import Any, Callable, Dict

from peteos.chatbot import ChatBotManager, ChatHistory, Message, ContentPart
from peteos.executionenvironment import ExecutionEnvironment
from peteos.logger import get_logger
from peteos.role import Role
from peteos.toolmanager import ToolManager, Tool

_logger = get_logger(__name__)


def _cast_args_to_types(func: Callable, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cast arguments to correct types based on function signature.

    This handles type mismatches between LLM-returned JSON strings and
    Python type annotations or default values (e.g., "1" -> 1, "true" -> True).

    Args:
        func: The function to cast arguments for.
        args: Dictionary of arguments to cast.

    Returns:
        Dictionary with arguments cast to correct types.
    """
    signature = inspect.signature(func)
    casted_args = {}

    for param_name, param in signature.parameters.items():
        if param_name not in args:
            # Use default value if parameter has one
            if param.default != inspect.Parameter.empty:
                casted_args[param_name] = param.default
            continue

        value = args[param_name]
        annotation = param.annotation

        # Skip casting for Any or empty annotations
        if annotation == inspect.Parameter.empty or annotation == Any:
            # Try to infer type from default value
            if param.default != inspect.Parameter.empty:
                target_type = type(param.default)
                # If default is not the same type as value, try conversion
                if type(value) != target_type:
                    value = _try_cast_value(value, target_type)
            casted_args[param_name] = value
            continue

        # Get the target type from annotation
        target_type = annotation

        # Handle special cases
        if target_type == int:
            if isinstance(value, str):
                try:
                    value = int(value)
                except ValueError:
                    pass
        elif target_type == float:
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    pass
        elif target_type == bool:
            if isinstance(value, str):
                value = value.lower() in ("true", "1", "yes")
        elif target_type == str:
            if not isinstance(value, str):
                value = str(value)
        else:
            # For other types, try to cast using the type
            value = _try_cast_value(value, target_type)

        casted_args[param_name] = value

    return casted_args


def _try_cast_value(value: Any, target_type: type) -> Any:
    """
    Try to cast a value to the target type.

    Args:
        value: The value to cast.
        target_type: The target type.

    Returns:
        The casted value, or the original value if casting fails.
    """
    if target_type == int and isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    elif target_type == float and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    elif target_type == bool and isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    elif target_type == str:
        return str(value)
    else:
        return value


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
            # response.data has format from translation: {text: "...", reasoning: "...", content: [...]}
            # Role is required - must be present (ChatBot ensures this)
            # If there's an error from the chatbot, skip appending and exit loop
            if "error" in response.data:
                _logger.warning("Chatbot returned error, skipping response: %s", response.data["error"])
                break
            assert "role" in response.data, f"ChatBot response missing 'role' field: {response.data.keys()}"

            # Build content parts from accumulated response data
            content_parts = []

            # Add reasoning if present
            if response.data.get("reasoning"):
                content_parts.append(ContentPart(part_type="reasoning", reasoning=response.data["reasoning"]))

            # Add tool calls from content array if present
            if "content" in response.data and isinstance(response.data["content"], list):
                # Filter for tool_use items
                tool_use_items = [
                    item for item in response.data["content"]
                    if isinstance(item, dict) and item.get("type") == "tool_use"
                ]
                if tool_use_items:
                    content_parts.append(ContentPart(part_type="tool_calls", tool_calls=tool_use_items))

            self.chat_history.append_message(Message(
                role=response.data["role"],
                content=content_parts
            ))

            # Check if response contains tool calls
            # Tool calls are stored in content array with type="tool_use"
            content_array = response.data.get("content", [])
            tool_calls_list = [
                item for item in content_array
                if isinstance(item, dict) and item.get("type") == "tool_use"
            ]

            # Add text content if present
            if response.data.get("text"):
                content_parts.append(ContentPart(part_type="text", text=response.data["text"]))

            _logger.debug("Tool calls detected: %s", tool_calls_list)
            _logger.debug("response.data keys: %s", list(response.data.keys()))

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
                            # Cast arguments to correct types based on function signature
                            args = _cast_args_to_types(tool.func, args)
                            _logger.debug("Casted args for %s: %s", tool_name, args)

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
