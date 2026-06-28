"""ExecutionEnvironment — tool call management and execution."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from peteos.utils import get_logger

from peteos.conversation.context import Context as ConversationContext
from peteos.conversation.message import ContentPart, Message
from peteos.persona.role import Role
from peteos.persona.toolmanager import ToolManager

_logger = get_logger(__name__)


class ToolApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"


class ToolExecutionStatus(str, Enum):
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    DENIED = "denied"
    EXECUTING = "executing"
    EXECUTED = "executed"
    ABORTED = "aborted"


@dataclass
class ToolCallRecord:
    """Tracks the lifecycle of a tool call through approval and execution."""
    tool_call_id: str
    tool_call: ContentPart
    approval_status: ToolApprovalStatus = ToolApprovalStatus.PENDING
    execution_status: ToolExecutionStatus = ToolExecutionStatus.WAITING_FOR_APPROVAL
    nextcloud_message_id: Optional[str] = None
    execution_result: Optional[str] = None
    execution_success: Optional[bool] = None
    denied_reason: Optional[str] = None


@dataclass
class ToolCallGroup:
    """Encapsulates the tool call lifecycle for one assistant message.

    Owns the record list, result message reference, and the anchor name.
    Provides group-level queries and result injection.
    """

    id: str
    anchor_name: str
    records: list[ToolCallRecord] = field(default_factory=list)
    result_message: Optional[Message] = None

    def add_tool_call(self, record: ToolCallRecord) -> None:
        self.records.append(record)

    def has_pending(self) -> bool:
        return any(r.approval_status == ToolApprovalStatus.PENDING for r in self.records)

    def has_reviewed(self) -> bool:
        for r in self.records:
            if r.approval_status == ToolApprovalStatus.PENDING:
                return False
            return True
        return False

    def has_unfinished(self) -> bool:
        return any(r.execution_status != ToolExecutionStatus.EXECUTED for r in self.records)

    def is_done(self) -> bool:
        """Return True when all tool calls are finalized (no more pending or approved)."""
        for r in self.records:
            if r.approval_status not in (ToolApprovalStatus.DENIED, ToolApprovalStatus.APPROVED):
                return False
            if r.approval_status == ToolApprovalStatus.APPROVED:
                if r.execution_status != ToolExecutionStatus.EXECUTED:
                    return False
        return True

    def pop_first_reviewed(self) -> Optional[ToolCallRecord]:
        for i, r in enumerate(self.records):
            if r.approval_status != ToolApprovalStatus.PENDING:
                return self.records.pop(i)
        return None

    def set_result_message(self, message: Message) -> None:
        self.result_message = message

    def get_result_message(self) -> Optional[Message]:
        return self.result_message


@dataclass
class ApprovalEvent:
    """Event pushed to Runner.event_queue to signal approval of a tool call."""
    tool_call_id: str = ""
    tool_call: ContentPart = field(default_factory=lambda: ContentPart({"type": "tool_use"}))
    approved: bool = True


class ExecutionEnvironment:
    """Manages tool calls, approval, and execution.

    Provides the hook system and the full tool call lifecycle:

    - Queue management (``add_tool_call``)
    - Approval workflow (``_handle_approval``)
    - Single-tool execution (``execute_tool``)
    - Lifecycle hooks
    - Foreground group management

    The Runner owns the event loop and chatbot interaction.
    """

    def __init__(
        self,
        tool_manager: "ToolManager",
        role: "Role",
        auto_approve_tools: list[str],
        tool_failure_policy: str,
    ) -> None:
        self._tool_manager = tool_manager
        self._role = role
        self.auto_approve_tools = auto_approve_tools
        self.tool_failure_policy = tool_failure_policy
        self._groups: dict[str, ToolCallGroup] = {}  # group_id -> ToolCallGroup
        self._foreground_group: ToolCallGroup | None = None
        self._hooks: dict[str, list[Callable]] = {
            "before_send_to_chatbot": [],
            "after_step": [],
            "after_message_append": [],
            "before_notification_publish": [],
            "before_tool_execution": [],
            "after_tool_execution": [],
        }

    # ------------------------------------------------------------------ #
    # Hook management
    # ------------------------------------------------------------------ #

    def register_hook(self, hook_point: str, callback: Callable, *args: Any) -> None:
        """Register a hook callback for a specific hook point."""
        if hook_point not in self._hooks:
            raise ValueError(f"Unknown hook point: {hook_point}")
        self._hooks[hook_point].append(_partial(callback, *args))

    def deregister_hook(self, hook_point: str, callback: Callable) -> None:
        """Deregister a specific hook callback."""
        if hook_point not in self._hooks:
            raise ValueError(f"Unknown hook point: {hook_point}")
        for hook in list(self._hooks[hook_point]):
            if hasattr(hook, 'func') and hook.func == callback:
                self._hooks[hook_point].remove(hook)
                return
            elif hook == callback:
                self._hooks[hook_point].remove(hook)
                return
        raise ValueError(f"Callback not found for hook point '{hook_point}'")

    def deregister_all_hooks(self, hook_point: str) -> None:
        """Deregister all hooks for a specific hook point."""
        if hook_point not in self._hooks:
            raise ValueError(f"Unknown hook point: {hook_point}")
        self._hooks[hook_point].clear()

    async def call_hooks(self, hook_point: str, *args: Any) -> Any | None:
        """Call all hooks for *hook_point*, merging results by severity."""
        if hook_point not in self._hooks:
            raise ValueError(f"Unknown hook point: {hook_point}")
        merged: Any | None = None
        from peteos.engine.exec_status import ExecStatus, _merge_exec_status
        for callback in self._hooks[hook_point]:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, ExecStatus):
                merged = _merge_exec_status(merged, result)
        return merged

    async def call_hooks_deny(self, hook_point: str, *args: Any) -> Any | None:
        """Call all hooks for *hook_point*, collecting results and denying on the first deny."""
        if hook_point not in self._hooks:
            raise ValueError(f"Unknown hook point: {hook_point}")
        first_deny: Any | None = None
        for callback in self._hooks[hook_point]:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                result = await result
            if first_deny is None and isinstance(result, tuple) and len(result) == 2 and result[0] is False:
                first_deny = result
        return first_deny

    # ------------------------------------------------------------------ #
    # Foreground group management
    # ------------------------------------------------------------------ #

    def get_foreground_group(self) -> ToolCallGroup | None:
        """Return the current foreground tool call group, or None."""
        return self._foreground_group

    def create_tool_group(self, group_id: str, anchor_name: str) -> None:
        """Create a new tool call group and set it as the foreground group.

        Raises an error if a foreground group already exists.
        """
        if self._foreground_group is not None:
            raise ValueError("A foreground tool call group already exists")
        group = ToolCallGroup(id=group_id, anchor_name=anchor_name)
        self._groups[group_id] = group
        self._foreground_group = group

    def close_foreground_group(self) -> None:
        """Close and remove the foreground group."""
        if self._foreground_group is None:
            return
        self._groups.pop(self._foreground_group.id, None)
        self._foreground_group = None

    def add_tool_call(self, tool_call: ContentPart) -> ToolCallRecord:
        """Add a tool call to the foreground group. Returns the created record."""
        group = self._foreground_group
        if group is None:
            raise RuntimeError("No foreground tool call group")

        tc_id = tool_call.call_id
        if tc_id is None:
            raise ValueError("Tool call missing required 'call_id' field")

        tool_name = tool_call.name

        if not self._tool_manager or not self._tool_manager.get_tool(tool_name):
            record = ToolCallRecord(
                tool_call_id=tc_id,
                tool_call=tool_call,
                approval_status=ToolApprovalStatus.DENIED,
                execution_status=ToolExecutionStatus.DENIED,
                denied_reason=f"Tool '{tool_name}' is not available for this agent",
            )
        elif tool_name in self.auto_approve_tools:
            record = ToolCallRecord(
                tool_call_id=tc_id,
                tool_call=tool_call,
                approval_status=ToolApprovalStatus.APPROVED,
                execution_status=ToolExecutionStatus.EXECUTING,
            )
        else:
            record = ToolCallRecord(
                tool_call_id=tc_id,
                tool_call=tool_call,
                approval_status=ToolApprovalStatus.PENDING,
                execution_status=ToolExecutionStatus.WAITING_FOR_APPROVAL,
            )
        group.add_tool_call(record)
        return record

    def get_pending_tool_calls(self) -> list[ToolCallRecord]:
        """Return all pending tool call records from the foreground group."""
        group = self._foreground_group
        if group is None:
            return []
        return [r for r in group.records if r.approval_status == ToolApprovalStatus.PENDING]

    def find_pending_record(self, tool_call_id: str) -> Optional[ToolCallRecord]:
        """Find a tool call record by its tool_call_id in the foreground group."""
        group = self._foreground_group
        if group is None:
            return None
        for record in group.records:
            if record.tool_call_id == tool_call_id:
                return record
        return None

    def _handle_approval(self, event: Any) -> tuple[bool, Optional[str]]:
        """Handle an approval/denial event for a pending tool call.

        If the tool call is denied, all remaining PENDING records in the
        group are also denied.
        """
        group = self._foreground_group
        if group is None:
            return False, None

        for record in group.records:
            if record.tool_call_id == event.tool_call_id:
                if event.approved:
                    record.approval_status = ToolApprovalStatus.APPROVED
                    return True, group.id
                else:
                    record.approval_status = ToolApprovalStatus.DENIED
                    for other in group.records:
                        if other.tool_call_id != event.tool_call_id and other.approval_status == ToolApprovalStatus.PENDING:
                            other.approval_status = ToolApprovalStatus.DENIED
                            other.execution_status = ToolExecutionStatus.DENIED
                            other.denied_reason = "Tool group denied by user."
                    return False, group.id
        return False, None

    # ------------------------------------------------------------------ #
    # Result injection — EE owns result message and ContentPart injection.
    # ------------------------------------------------------------------ #

    async def execute_and_inject(self, tool_call: ContentPart, runner: "Runner | None" = None) -> tuple[str | None, bool]:
        """Execute a tool and inject the result ContentPart into the foreground group's result message.

        Creates the result message lazily — only when the first non-None
        result arrives. Returns (None, True) for fire-and-forget tools
        without adding a ContentPart to the result message.

        Args:
            tool_call: ContentPart with type "tool_use".
            runner: Optional runner to inject as ``runner`` kwarg into tool calls.

        Returns:
            Tuple of (result_string_or_None, success_bool).
        """
        group = self._foreground_group
        if group is None:
            raise RuntimeError("No foreground tool call group")

        result_str, success = await self.execute_tool(tool_call, runner=runner)
        if result_str is None:
            return None, success

        result_msg = self._foreground_group.result_message
        if result_msg is None:
            result_msg = Message.create("tool_result", [])
            self._foreground_group.result_message = result_msg
        cp = ContentPart.create_tool_result(tool_call.call_id, result_str)
        cp.raw_dict["name"] = tool_call.name
        result_msg.raw_dict["content"].append(cp.raw_dict)

        return result_str, success

    # ------------------------------------------------------------------ #
    # Tool execution — standalone
    # ------------------------------------------------------------------ #

    async def execute_tool(self, tool_call: ContentPart, runner: "Runner | None" = None) -> tuple[str | None, bool]:
        """Execute a single tool call.

        Looks up the tool by name, casts arguments, fires the
        ``before_tool_execution`` deny hook, executes the tool, fires
        the ``after_tool_execution`` hook, and returns the result.

        Returns (None, True) when the tool returns None (fire-and-forget),
        (result_string, True) for successful calls, and (error_msg, False)
        for failures.

        Args:
            tool_call: ContentPart with type "tool_use".

        Returns:
            Tuple of (result_string_or_None, success_bool).
        """
        tool_name = tool_call.name
        args = json.loads(tool_call.arguments or "{}")

        _logger.debug("Attempting to execute tool: %s(%s)", tool_name, args)

        tool = self._tool_manager.get_tool(tool_name)
        if not tool:
            _logger.warning("Tool '%s' not found", tool_name)
            return f"Error: Tool '{tool_name}' not found", False

        casted_args = _cast_args_to_types(tool.func, args)
        hook_result = await self.call_hooks_deny("before_tool_execution", tool_call)
        if hook_result is not None:
            allow, message = hook_result
            if not allow:
                _logger.debug("Tool %s denied by hook: %s", tool_name, message)
                return message, False

        try:
            result = tool.execute(**casted_args, runner=runner)
            if asyncio.iscoroutine(result):
                result = await result
            if result is None:
                await self.call_hooks("after_tool_execution", tool_call, None, True)
                _logger.debug("Tool %s returned None (fire-and-forget)", tool_name)
                return None, True
            result_str = str(result)
            await self.call_hooks("after_tool_execution", tool_call, result_str, True)
            _logger.debug("Tool %s returned: %s", tool_name, result_str)
            return result_str, True
        except Exception as e:
            _logger.debug("Tool %s failed: %s", tool_name, str(e))
            return f"Error: {type(e).__name__}: {str(e)}", False


# ------------------------------------------------------------------ #
# Utility: _partial — lightweight replacement for functools.partial
# ------------------------------------------------------------------ #

def _partial(func: Callable, *args: Any) -> Callable:
    """Return a callable that prepends *args* when invoked."""
    def wrapper(*extra: Any, **kwargs: Any) -> Any:
        return func(*args, *extra, **kwargs)
    return wrapper


# ------------------------------------------------------------------ #
# Type casting for tool arguments
# ------------------------------------------------------------------ #

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
