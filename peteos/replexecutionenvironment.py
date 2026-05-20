import json

import inspect
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from peteos.chatbot import ChatBotManager, ChatHistory, Message, ContentPart
from peteos.executionenvironment import ExecutionEnvironment, ExecStatus
from peteos.logger import get_logger
from peteos.role import Role
from peteos.toolmanager import ToolManager

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
        session: "Session"
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

        1. Call chatbot, collect response (skip on re-entry after tool_pending)
        2. Append assistant message to chat_history
        3. For each tool call: fire before_tool hook → execute or return tool_pending
        4. Fire before_loop_continue or before_loop_exit hook

        Returns:
            ("done", None) | ("continue", None) | ("tool_pending", {"tool_call": dict})
        """
        # --- Phase 1: Call chatbot (skip if re-entering after tool_pending) ---
        has_text_part = False

        # We only send the extended chat history to the chatbot to get a further response when there are no (more)
        # pending tool calls.
        if not self._session.has_unfinished_tool_call():
            response = await self.chatbot.send_message(self.chat_history)
            async for _ in response:
                if self._interrupt:
                    await self._call_hooks("before_loop_exit", "interrupt")
                    return (ExecStatus.INTERRUPTED, None)

            if self._interrupt:
                await self._call_hooks("before_loop_exit", "interrupt")
                return (ExecStatus.INTERRUPTED, None)

            # --- Phase 2: Error handling ---
            if "error" in response.data:
                _logger.warning("Chatbot returned error, skipping response: %s", response.data["error"])
                return (ExecStatus.ERROR, None)

            if "role" not in response.data:
                _logger.error("ChatBot response missing 'role' field. Response data: %s", response.data)
                return (ExecStatus.ERROR, None)

            # --- Phase 3: Build content parts from content array in original order ---
            content_array: list = response.data.get("content", [])
            content_parts: list[ContentPart] = []

            for item in content_array:
                if not isinstance(item, dict):
                    _logger.error("Expected content item to be a dict, got %s", type(item).__name__)
                    continue
                item_type = item["type"]
                if item_type == "tool_use":
                    content_parts.append(ContentPart(part_type="tool_use", **item))
                    self._session.add_tool_call(item)
                elif item_type == "text":
                    content_value = item["content"]
                    content_parts.append(ContentPart(part_type="text", text=content_value))
                    has_text_part = True
                elif item_type == "thinking":
                    content_value = item["content"]
                    content_parts.append(
                        ContentPart(part_type="reasoning", reasoning=content_value)
                    )

            # --- Phase 4: Append assistant message ---
            _logger.debug("ChatBot response: role=%s, content_types=%s", response.data.get("role"), [item.get("type") for item in content_array] if isinstance(content_array, list) else "N/A")
            response_msg = Message(role=response.data["role"], content=content_parts)
            self.chat_history.append_message(response_msg)
            self._session.publish_notification(response_msg)

            # Fire hook so channels/app can snapshot per-message metadata
            await self._call_hooks("on_message_published", response_msg, self._session.uuid)

        # --- Phase 5: Execute tool calls ---
        from peteos.session import ToolApprovalStatus

        did_tool_calls = self._session.has_reviewed_tool_call()
        while self._session.has_reviewed_tool_call():
            record = self._session.pop_pending_tool_call()
            tool_call = record.tool_call
            tool_name = tool_call["name"]

            # Handle denied tool calls (status set by _handle_approval)
            if record.approval_status == ToolApprovalStatus.DENIED:
                denial_msg = record.tool_call.get("denied_reason", "Tool call was denied by user.")
                msg = Message(
                    role="tool_result",
                    content=[ContentPart(
                        part_type="tool_result",
                        name=tool_name,
                        content=denial_msg,
                    )],
                )
                self.chat_history.append_message(msg)
                self._session.publish_notification(msg)
                await self._call_hooks("after_tool_execution", record.tool_call, denial_msg, False)
                _logger.debug("[repl] step(): Tool call %s was denied by user", tool_name)
                return (ExecStatus.TOOL_DENIED, None)

            args = json.loads(tool_call.get("arguments", "{}"))
            _logger.debug("Attempting to execute tool: %s(%s)", tool_name, args)
            tool = self.tool_manager.get_tool(tool_name)

            if not tool:
                tool_not_found_msg = Message(
                    role="tool_result",
                    content=[ContentPart(
                        part_type="tool_result",
                        name=tool_name,
                        content=f"Error: Tool '{tool_name}' not found",
                    )],
                )
                self.chat_history.append_message(tool_not_found_msg)
                self._session.publish_notification(tool_not_found_msg)
                return (ExecStatus.TOOL_NOT_FOUND, None)

            args = _cast_args_to_types(tool.func, args)
            hook_result = await self._call_hooks("before_tool_execution", tool_call)
            if hook_result is not None:
                allow, message = hook_result
                if not allow:
                    self._append_tool_result(tool_name=tool_name, content=message, success=False)
                    return (ExecStatus.TOOL_DENIED, None)

            try:
                result = tool.execute(**args)
                self._append_tool_result(tool_name=tool_name, content=str(result), success=True)
                await self._call_hooks("after_tool_execution", tool_call, str(result), True)
                _logger.debug("Tool %s returned: %s", tool_name, str(result))
            except Exception as e:
                self._append_tool_result(
                    tool_name=tool_name,
                    content=f"Error: {type(e).__name__}: {str(e)}",
                    success=False,
                )
                _logger.debug("Tool %s failed: %s", tool_name, str(e))
                return (ExecStatus.TOOL_FAILED, None)

            _logger.debug("[repl] step(): tool call executed; continue")

        if did_tool_calls and not self._session.has_pending_tool_call():
            _logger.debug("[repl] step(): Did tool calls. Need to continue, such that the ChatBot can see the result.")
            return (ExecStatus.CONTINUE, None)

        if has_text_part and not self._session.has_pending_tool_call():
            # No tool calls — check for final answer or reasoning-only response
            _logger.debug("[repl] step(): Had final answer.")
            return (ExecStatus.FINISHED, None)

        if self._session.has_pending_tool_call():
            _logger.debug("[repl] step(): Waiting for user review of pending tool calls.")
            return (ExecStatus.PENDING, None)

        # no tool calls, no pending tools, no text part: only reasoning...
        _logger.debug("[repl] step(): Response contained only reasoning part(s).")
        return (ExecStatus.CONTINUE, None)

    async def execute_pending_tool(self, tool_call: dict) -> None:
        """Execute a tool call that was previously pending approval.

        Called by Session._handle_approval after an ApprovalEvent is received.

        Args:
            tool_call: Dict with 'name' and 'arguments' of the tool.
        """
        tool_call_id = tool_call.get("id", "")
        tool_name = tool_call.get("name")
        args = json.loads(tool_call.get("arguments", "{}"))
        _logger.debug("Executing pending tool: %s(%s)", tool_name, args)
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
            self._executed_tool_ids.add(tool_call_id)
            return

        args = _cast_args_to_types(tool.func, args)

        try:
            result = tool.execute(**args)
            self._append_tool_result(tool_name=tool_name, content=str(result), success=True)
            await self._call_hooks("after_tool_execution", tool_call, str(result), True)
            _logger.debug("Tool %s returned: %s", tool_name, str(result))
        except Exception as e:
            self._append_tool_result(
                tool_name=tool_name,
                content=f"Error: {type(e).__name__}: {str(e)}",
                success=False,
            )
            await self._call_hooks("after_tool_execution", tool_call, str(e), False)
            _logger.debug("Tool %s failed: %s", tool_name, str(e))

        self._executed_tool_ids.add(tool_call_id)

    def _append_tool_result(
        self,
        tool_name: str,
        content: str,
        success: bool,
    ) -> None:
        msg = Message(
            role="tool_result",
            content=[
                ContentPart(part_type="tool_result", name=tool_name, content=content)
            ],
        )
        self.chat_history.append_message(msg)
        self._session.publish_notification(msg)
