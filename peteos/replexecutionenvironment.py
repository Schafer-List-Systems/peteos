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

    def _append_tool_result(
        self,
        tool_name: str,
        content: str,
        success: bool,
    ) -> None:
        """Append a tool result message to the chat history.

        Args:
            tool_name: Name of the tool that was executed.
            content: Result content as a string.
            success: Whether the execution succeeded.
        """
        msg = Message(
            role="tool_result",
            content=[
                ContentPart(part_type="tool_result", name=tool_name, content=content),
                ContentPart(part_type="bool", success=success),
            ],
        )
        self.chat_history.append_message(msg)

    async def _run_impl(self) -> None:
        """
        Run the REPL loop.

        Processes chat history through the chatbot, executes tool calls, and
        appends output to ChatHistory. Loops until final answer is received
        or interrupt flag is set.
        """
        while not self._interrupt:
            # --- Phase 1: Call chatbot and collect response ---
            response = await self.chatbot.send_message(self.chat_history)
            async for _ in response:
                if self._interrupt:
                    break

            if self._interrupt:
                await self._call_hooks("before_loop_exit", "interrupt")
                break

            # --- Phase 2: Error handling ---
            if "error" in response.data:
                _logger.warning("Chatbot returned error, skipping response: %s", response.data["error"])
                break
            assert "role" in response.data, f"ChatBot response missing 'role' field: {response.data.keys()}"

            # --- Phase 3: Build complete content parts ---
            content_parts: list[ContentPart] = []

            if response.data.get("reasoning"):
                content_parts.append(
                    ContentPart(part_type="reasoning", reasoning=response.data["reasoning"])
                )

            # Extract tool calls once, reuse for both building and executing
            tool_calls_list: list[dict] = []
            content_array = response.data.get("content", [])
            if isinstance(content_array, list):
                tool_calls_list = [
                    item for item in content_array
                    if isinstance(item, dict) and item.get("type") == "tool_use"
                ]

            if tool_calls_list:
                content_parts.append(
                    ContentPart(part_type="tool_calls", tool_calls=tool_calls_list)
                )

            # Add text before appending so the Message is complete
            if response.data.get("text"):
                content_parts.append(ContentPart(part_type="text", text=response.data["text"]))

            # --- Phase 4: Append message to history ---
            self.chat_history.append_message(
                Message(role=response.data["role"], content=content_parts)
            )

            # --- Phase 5: Execute tool calls (flat logic, no deep nesting) ---
            if tool_calls_list:
                history_length_before = len(self.chat_history.messages)

                for tool_call in tool_calls_list:
                    tool_name = tool_call.get("name")
                    args = json.loads(tool_call.get("arguments", "{}"))
                    tool = self.tool_manager.get_tool(tool_name)

                    if not tool:
                        self._append_tool_result(
                            tool_name=tool_name,
                            content=f"Error: Tool '{tool_name}' not found",
                            success=False,
                        )
                        await self._call_hooks("after_tool_execution", tool_call, f"Error: Tool '{tool_name}' not found", False)
                        continue

                    args = _cast_args_to_types(tool.func, args)

                    hook_result = await self._call_hooks("before_tool_execution", tool_call)
                    if hook_result is not None:
                        allow, message = hook_result
                        if not allow:
                            self._append_tool_result(tool_name=tool_name, content=message, success=False)
                            await self._call_hooks("after_tool_execution", tool_call, message, False)
                            continue

                    try:
                        # Call the tool
                        result = tool.execute(**args)
                        self._append_tool_result(tool_name=tool_name, content=str(result), success=True)
                        await self._call_hooks("after_tool_execution", tool_call, str(result), True)
                    except Exception as e:
                        self._append_tool_result(
                            tool_name=tool_name,
                            content=f"Error: {type(e).__name__}: {str(e)}",
                            success=False,
                        )
                        await self._call_hooks("after_tool_execution", tool_call, str(e), False)

                # Collect delta messages for the continue hook
                new_count = len(self.chat_history.messages) - history_length_before
                delta_messages = (
                    self.chat_history.messages[-new_count:] if new_count > 0 else []
                )
                hook_result = await self._call_hooks("before_loop_continue", delta_messages)
                if hook_result is not None:
                    should_exit, reason = hook_result
                    if should_exit:
                        await self._call_hooks("before_loop_exit", reason)
                        break
            else:
                # No tool calls
                if response.data.get("text"):
                    await self._call_hooks("before_loop_exit", "final_answer")
                    break
                # Only reasoning text — check hook for continue/exit decision
                hook_result = await self._call_hooks("before_loop_continue", [])
                if hook_result is not None:
                    should_exit, reason = hook_result
                    if should_exit:
                        await self._call_hooks("before_loop_exit", reason)
                        break
                # Otherwise continue the loop
