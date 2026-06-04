import json

import asyncio
import inspect
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from peteos.chatbot import ChatHistory, Message, ContentPart
from peteos.executionenvironment import ExecutionEnvironment, ExecStatus
from peteos.logger import get_logger
from peteos.role import Role
from peteos.toolmanager import ToolManager

if TYPE_CHECKING:
    from peteos.session import Session  # circular import guard

_logger = get_logger(__name__)


def _cast_args_to_types(func: Callable, args: Dict[str, Any]) -> Dict[str, Any]:
    """Type-cast the arguments the LLM actually provided.

    Only casts values for parameters the LLM included in its call.
    Does not inject defaults or hallucinate missing arguments.
    """
    signature = inspect.signature(func)
    casted_args: Dict[str, Any] = {}

    for param_name, value in args.items():
        if param_name not in signature.parameters:
            casted_args[param_name] = value
            continue

        param = signature.parameters[param_name]
        annotation = param.annotation

        if annotation == inspect.Parameter.empty or annotation == Any:
            casted_args[param_name] = value
            continue

        if annotation == int and isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                pass
        elif annotation == float and isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        elif annotation == bool and isinstance(value, str):
            value = value.lower() in ("true", "1", "yes")
        elif annotation == str and not isinstance(value, str):
            value = str(value)
        else:
            value = _try_cast_value(value, annotation)

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
        chat_history: ChatHistory,
        tool_manager: ToolManager,
        role: Role,
    ):
        super().__init__(
            chat_history=chat_history,
            tool_manager=tool_manager,
            role=role,
        )
        self._role = role

    async def step(self, session: "Session") -> tuple[ExecStatus, dict | None]:
        """Execute one loop iteration.

        1. Call chatbot, collect response (skip on re-entry after tool_pending)
        2. Append assistant message to chat_history
        3. For each tool call: fire before_tool hook → execute or return tool_pending
        4. Continue/exit based on response type

        Args:
            session: The Session owning the state consumed by this step.

        Returns:
            ("done", None) | ("continue", None) | ("tool_pending", {"tool_call": dict})
        """
        # --- Phase 1: Call chatbot (skip if re-entering after tool_pending) ---
        has_text_part = False

        # We only send the extended chat history to the chatbot to get a further response when there are no (more)
        # pending tool calls.
        if not session.has_unfinished_tool_call():
            await self._call_hooks("before_send_to_chatbot", session, session.chat_history)
            response = await self.chatbot.send_message(session.chat_history)
            async for _ in response:
                pass

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
                    session.add_tool_call(item)
                elif item_type == "texttool_use":
                    content_parts.append(ContentPart(part_type="text", text=item["content"]))
                    content_parts.append(ContentPart(part_type="tool_use", **item))
                    session.add_tool_call(item)
                elif item_type == "text":
                    content_value = item["content"]
                    content_parts.append(ContentPart(part_type="text", text=content_value))
                    has_text_part = True
                elif item_type == "thinking":
                    content_value = item["content"]
                    content_parts.append(
                        ContentPart(part_type="reasoning", reasoning=content_value)
                    )
                else:
                    _logger.debug("Unknown item type: %s", item_type)
                    return (ExecStatus.ERROR, None)

            # --- Phase 4: Append assistant message ---
            _logger.debug("ChatBot response: role=%s, content_types=%s", response.data.get("role"), [item.get("type") for item in content_array] if isinstance(content_array, list) else "N/A")
            _logger.debug("ChatBot full response data: %s", json.dumps(response.data, indent=2, default=str))
            response_msg = Message(role=response.data["role"], content=content_parts)
            await session.append_and_notify(response_msg)

        # --- Phase 5: Execute tool calls ---
        from peteos.session import ToolApprovalStatus

        did_tool_calls = session.has_reviewed_tool_call()
        yielded = False
        while session.has_reviewed_tool_call():
            record = session.pop_pending_tool_call()
            tool_call = record.tool_call
            tool_name = tool_call["name"]

            # Handle denied tool calls (status set by _handle_approval)
            if record.approval_status == ToolApprovalStatus.DENIED:
                denial_msg = record.tool_call.get("denied_reason", "Tool call was denied by user.")
                tool_call_id = tool_call.get("id", "")
                msg = Message(
                    role="tool_result",
                    content=[ContentPart(
                        part_type="tool_result",
                        name=tool_name,
                        content=denial_msg,
                        tool_use_id=tool_call_id,
                    )],
                )
                await session.append_and_notify(msg)
                await self._call_hooks("after_tool_execution", session, record.tool_call, denial_msg, False)
                _logger.debug("[repl] step(): Tool call %s was denied by user", tool_name)
                return (ExecStatus.TOOL_DENIED, None)

            args = json.loads(tool_call.get("arguments", "{}"))
            _logger.debug("Attempting to execute tool: %s(%s)", tool_name, args)
            tool = session.tool_manager.get_tool(tool_name)

            if not tool:
                tool_call_id = tool_call.get("id", "")
                tool_not_found_msg = Message(
                    role="tool_result",
                    content=[ContentPart(
                        part_type="tool_result",
                        name=tool_name,
                        content=f"Error: Tool '{tool_name}' not found",
                        tool_use_id=tool_call_id,
                    )],
                )
                session.append_and_notify(tool_not_found_msg)
                return (ExecStatus.TOOL_NOT_FOUND, None)

            args = _cast_args_to_types(tool.func, args)
            hook_result = await self._call_hooks("before_tool_execution", session, tool_call)
            if hook_result is not None:
                allow, message = hook_result
                if not allow:
                    tool_call_id = tool_call.get("id", "")
                    await self._append_tool_result(session, tool_name=tool_name, content=message, tool_use_id=tool_call_id)
                    return (ExecStatus.TOOL_DENIED, None)

            tool_call_id = tool_call.get("id", "")
            try:
                result = tool.execute(**args, session=session)
                if asyncio.iscoroutine(result):
                    result = await result
                await self._append_tool_result(session, tool_name=tool_name, content=str(result), tool_use_id=tool_call_id)
                await self._call_hooks("after_tool_execution", session, tool_call, str(result), True)
                _logger.debug("Tool %s returned: %s", tool_name, str(result))
            except Exception as e:
                await self._append_tool_result(
                    session,
                    tool_name=tool_name,
                    content=f"Error: {type(e).__name__}: {str(e)}",
                    tool_use_id=tool_call_id
                )
                _logger.debug("Tool %s failed: %s", tool_name, str(e))
                return (ExecStatus.TOOL_FAILED, None)

            _logger.debug("[repl] step(): tool call executed; continue")
            if tool_name == "yield_back":
                yielded = True
                break

        if yielded:
            _logger.debug("[repl] step(): Agent called yield_back, finishing.")
            return (ExecStatus.FINISHED, None)

        if did_tool_calls and not session.has_pending_tool_call():
            _logger.debug("[repl] step(): Did tool calls. Need to continue, such that the ChatBot can see the result.")
            return (ExecStatus.CONTINUE, None)

        if has_text_part and not session.has_pending_tool_call():
            if self._role.behavior_policy == "continuous":
                _logger.debug("[repl] step(): Continuous agent produced text, keeping loop active.")
                return (ExecStatus.CONTINUE, None)
            _logger.debug("[repl] step(): Had final answer.")
            return (ExecStatus.FINISHED, None)

        if session.has_pending_tool_call():
            _logger.debug("[repl] step(): Waiting for user review of pending tool calls.")
            return (ExecStatus.PENDING, None)

        # no tool calls, no pending tools, no text part: only reasoning...
        _logger.debug("[repl] step(): Response contained only reasoning part(s).")
        return (ExecStatus.CONTINUE, None)

    async def _append_tool_result(
        self,
        session: "Session",
        tool_name: str,
        content: str,
        tool_use_id: str,
    ) -> None:
        msg = Message(
            role="tool_result",
            content=[
                ContentPart(part_type="tool_result", name=tool_name, content=content, tool_use_id=tool_use_id)
            ],
        )
        await session.append_and_notify(msg)
