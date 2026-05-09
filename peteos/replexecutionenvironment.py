import json

import inspect
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

import inspect
from typing import Any, Callable, Dict

from peteos.chatbot import ChatBotManager, ChatHistory, Message, ContentPart
from peteos.executionenvironment import ExecutionEnvironment
from peteos.logger import get_logger
from peteos.role import Role
from peteos.toolmanager import ToolManager, Tool

if TYPE_CHECKING:
    from peteos.session import Session  # circular import guard

_logger = get_logger(__name__)


def _cast_args_to_types(func: Callable, args: Dict[str, Any]) -> Dict[str, Any]:
    signature = inspect.signature(func)
    casted_args: Dict[str, Any] = {}

    for param_name, param in signature.parameters.items():
        if param_name not in args:
            if param.default != inspect.Parameter.empty:
                casted_args[param_name] = param.default
            continue

        value = args[param_name]
        annotation = param.annotation

        if annotation == inspect.Parameter.empty or annotation == Any:
            if param.default != inspect.Parameter.empty:
                target_type = type(param.default)
                if type(value) != target_type:
                    value = _try_cast_value(value, target_type)
            casted_args[param_name] = value
            continue

        target_type = annotation
        if target_type == int and isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                pass
        elif target_type == float and isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        elif target_type == bool and isinstance(value, str):
            value = value.lower() in ("true", "1", "yes")
        elif target_type == str:
            if not isinstance(value, str):
                value = str(value)
        else:
            value = _try_cast_value(value, target_type)

        casted_args[param_name] = value

    return casted_args


def _try_cast_value(value: Any, target_type: type) -> Any:
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
        role: Role,
        session: Optional["Session"] = None,
    ):
        super().__init__(
            chatbot_manager=chatbot_manager,
            chat_history=chat_history,
            tool_manager=tool_manager,
            role=role,
        )
        self._session: "Session" = session

    async def step(self) -> tuple[str, dict | None]:
        """Execute one loop iteration.

        1. Call chatbot, collect response
        2. Append assistant message to chat_history
        3. For each tool call: fire before_tool hook → execute or return tool_pending
        4. Fire before_loop_continue or before_loop_exit hook

        Returns:
            ("done", None) | ("continue", None) | ("tool_pending", {"tool_call": dict})
        """
        # --- Phase 1: Call chatbot ---
        response = await self.chatbot.send_message(self.chat_history)
        async for _ in response:
            if self._interrupt:
                await self._call_hooks("before_loop_exit", "interrupt")
                return ("done", None)

        if self._interrupt:
            await self._call_hooks("before_loop_exit", "interrupt")
            return ("done", None)

        # --- Phase 2: Error handling ---
        if "error" in response.data:
            _logger.warning("Chatbot returned error, skipping response: %s", response.data["error"])
            return ("done", None)

        assert "role" in response.data, f"ChatBot response missing 'role' field: {response.data.keys()}"

        # --- Phase 3: Build content parts ---
        content_parts: list[ContentPart] = []

        if response.data.get("reasoning"):
            content_parts.append(
                ContentPart(part_type="reasoning", reasoning=response.data["reasoning"])
            )

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

        if response.data.get("text"):
            content_parts.append(ContentPart(part_type="text", text=response.data["text"]))

        # --- Phase 4: Append assistant message ---
        self.chat_history.append_message(
            Message(role=response.data["role"], content=content_parts)
        )

        # --- Phase 5: Execute tool calls ---
        if tool_calls_list:
            history_length_before = len(self.chat_history.messages)

            for tool_call in tool_calls_list:
                # Check for pending approval from previous iteration
                pending = getattr(self, "_pending_tool_call", None)
                if pending is not None:
                    # Execute the pending tool (approved by session)
                    await self.execute_pending_tool(pending)
                    self._pending_tool_call = None
                    continue

                tool_name = tool_call.get("name")
                args = json.loads(tool_call.get("arguments", "{}"))
                tool = self.tool_manager.get_tool(tool_name)

                if not tool:
                    msg = Message(
                        role="tool_result",
                        content=[ContentPart(
                            part_type="tool_result",
                            name=tool_name,
                            content=f"Error: Tool '{tool_name}' not found",
                        )],
                    )
                    self.chat_history.append_message(msg)
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
                    # If hook returns "pending approval" (new convention for milestone 2)
                    if allow is None or allow == "pending":
                        return ("tool_pending", {"tool_call": tool_call})

                try:
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

            new_count = len(self.chat_history.messages) - history_length_before
            delta_messages = (
                self.chat_history.messages[-new_count:] if new_count > 0 else []
            )
            hook_result = await self._call_hooks("before_loop_continue", delta_messages)
            if hook_result is not None:
                should_exit, reason = hook_result
                if should_exit:
                    await self._call_hooks("before_loop_exit", reason)
                    return ("done", None)
            # Loop continues to chatbot
            return ("continue", None)
        else:
            # No tool calls
            if response.data.get("text"):
                await self._call_hooks("before_loop_exit", "final_answer")
                return ("done", None)
            # Only reasoning text
            hook_result = await self._call_hooks("before_loop_continue", [])
            if hook_result is not None:
                should_exit, reason = hook_result
                if should_exit:
                    await self._call_hooks("before_loop_exit", reason)
                    return ("done", None)
            return ("continue", None)

    async def execute_pending_tool(self, tool_call: dict) -> None:
        """Execute a tool call that was previously pending approval.

        Called by Session._handle_approval after an ApprovalEvent is received.

        Args:
            tool_call: Dict with 'name' and 'arguments' of the tool.
        """
        tool_name = tool_call.get("name")
        args = json.loads(tool_call.get("arguments", "{}"))
        tool = self.tool_manager.get_tool(tool_name)

        if not tool:
            msg = Message(
                role="tool_result",
                content=[ContentPart(
                    part_type="tool_result",
                    name=tool_name,
                    content=f"Error: Tool '{tool_name}' not found",
                )],
            )
            self.chat_history.append_message(msg)
            await self._call_hooks("after_tool_execution", tool_call, f"Error: Tool '{tool_name}' not found", False)
            return

        args = _cast_args_to_types(tool.func, args)

        try:
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

    def _append_tool_result(
        self,
        tool_name: str,
        content: str,
        success: bool,
    ) -> None:
        msg = Message(
            role="tool_result",
            content=[
                ContentPart(part_type="tool_result", name=tool_name, content=content),
                ContentPart(part_type="bool", success=success),
            ],
        )
        self.chat_history.append_message(msg)
