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
from peteos.toolmanager import ToolManager

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
    tool_call: dict
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
    tool_call: dict = field(default_factory=dict)
    approved: bool = True


class ExecutionEnvironment:
    """Manages tool calls, approval, and execution.

    Provides the hook system and the full tool call lifecycle:

    - Queue management (``add_tool_call``, ``pop_pending_tool_call``)
    - Approval workflow (``_handle_approval``)
    - Single-tool execution (``execute_tool``)
    - Lifecycle hooks

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
        self._tool_to_group: dict[str, str] = {}  # tool_call_id -> group_id
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
    # Tool call queue management — group-based
    # ------------------------------------------------------------------ #

    def create_tool_group(self, group_id: str, anchor_name: str) -> None:
        """Create a new tool call group. Anchor already exists in Context."""
        if group_id in self._groups:
            raise ValueError(f"Tool call group already exists: {group_id}")
        self._groups[group_id] = ToolCallGroup(id=group_id, anchor_name=anchor_name)

    def add_tool_call(self, tool_call: dict, group_id: str) -> ToolCallRecord:
        """Add a tool call to the specified group. Returns the created record."""
        tc_id = tool_call.get("id")
        if tc_id is None:
            raise ValueError("Tool call missing required 'id' field")
        if tc_id in self._tool_to_group:
            raise ValueError(f"Duplicate tool call id: {tc_id}")
        if group_id not in self._groups:
            raise ValueError(f"Unknown tool call group: {group_id}")

        group = self._groups[group_id]
        tool_name = tool_call.get("name", "")

        if not self._tool_manager or not self._tool_manager.get_tool(tool_name):
            tool_call["denied_reason"] = f"Tool '{tool_name}' is not available for this agent"
            record = ToolCallRecord(
                tool_call_id=tc_id,
                tool_call=tool_call,
                approval_status=ToolApprovalStatus.DENIED,
                execution_status=ToolExecutionStatus.DENIED,
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
        self._tool_to_group[tc_id] = group_id
        return record

    # ------------------------------------------------------------------ #
    # Group-aware queries (group_id=None aggregates across all groups)
    # ------------------------------------------------------------------ #

    def get_group(self, group_id: str) -> Optional[ToolCallGroup]:
        """Return a tool call group by ID."""
        return self._groups.get(group_id)

    def get_tool_calls_in_group(self, group_id: str) -> list[ToolCallRecord]:
        """Return all tool call records for a specific group."""
        group = self._groups.get(group_id)
        return list(group.records) if group else []

    def get_group_id(self, tool_call_id: str) -> Optional[str]:
        """Return the group ID for a given tool call ID."""
        return self._tool_to_group.get(tool_call_id)

    def has_pending_tool_call(self, group_id: Optional[str] = None) -> bool:
        """Check if any tool call needs user approval (approval status is PENDING)."""
        if group_id:
            group = self._groups.get(group_id)
            return group.has_pending() if group else False
        return any(g.has_pending() for g in self._groups.values())

    def has_unfinished_tool_call(self, group_id: Optional[str] = None) -> bool:
        """Check if any tool call is not yet fully executed."""
        groups = [self._groups[group_id]] if group_id else list(self._groups.values())
        return any(g.has_unfinished() for g in groups)

    def has_reviewed_tool_call(self, group_id: Optional[str] = None) -> bool:
        """Check if the first tool call in a group has been approved (no longer PENDING)."""
        groups = [self._groups[group_id]] if group_id else list(self._groups.values())
        return any(g.has_reviewed() for g in groups)

    def get_pending_tool_calls(self) -> list[ToolCallRecord]:
        """Return all currently pending tool calls across all groups."""
        result = []
        for group in self._groups.values():
            result.extend(group.records)
        return result

    def pop_pending_tool_call(self, group_id: str) -> ToolCallRecord:
        """Remove and return the first approved tool call from a group."""
        group = self._groups.get(group_id)
        if group is None:
            raise RuntimeError(f"Unknown tool call group: {group_id}")
        record = group.pop_first_reviewed()
        if record is None:
            raise RuntimeError(f"No reviewed tool calls in group: {group_id}")
        return record

    def is_tool_call_pending(self, tool_call_id: str) -> bool:
        """Check if a tool call needs user approval."""
        group_id = self._tool_to_group.get(tool_call_id)
        group = self._groups.get(group_id)
        if group is None:
            return False
        for record in group.records:
            if record.tool_call_id == tool_call_id:
                return record.approval_status == ToolApprovalStatus.PENDING
        return False

    def _find_pending_record(self, tool_call_id: str) -> Optional[ToolCallRecord]:
        """Find a pending tool call record by its tool_call_id."""
        group_id = self._tool_to_group.get(tool_call_id)
        group = self._groups.get(group_id)
        if group is None:
            return None
        for record in group.records:
            if record.tool_call_id == tool_call_id:
                return record
        return None

    def _handle_approval(self, event: Any) -> tuple[bool, Optional[str]]:
        """Handle an approval/denial event for a pending tool call.

        Returns (True, group_id) if approved, (False, group_id) if denied,
        (False, None) if tool call not found.
        """
        record = self._find_pending_record(event.tool_call_id)
        group_id = self._tool_to_group.get(event.tool_call_id)
        if record is None:
            return False, None

        if event.approved:
            record.approval_status = ToolApprovalStatus.APPROVED
            return True, group_id
        else:
            record.approval_status = ToolApprovalStatus.DENIED
            return False, group_id

    # ------------------------------------------------------------------ #
    # Result injection — EE owns result message and ContentPart injection.
    # ------------------------------------------------------------------ #

    def _ensure_result_message(self, group_id: str) -> Message:
        """Ensure a result message exists for this group, creating it if needed."""
        group = self._groups.get(group_id)
        if group is None:
            raise ValueError(f"Unknown tool call group: {group_id}")
        if group.result_message is None:
            group.set_result_message(Message({"role": "tool_result", "content": []}))
        return group.result_message

    async def execute_and_inject(self, tool_call: dict, group_id: str) -> tuple[str, bool]:
        """Execute a tool and inject the result ContentPart into the group's result message.

        Creates the result message if needed. The runner is responsible
        for appending and notifying the result message.
        Returns (result_string, success) — same as ``execute_tool``.
        """
        group = self._groups.get(group_id)
        if group is None:
            raise ValueError(f"Unknown tool call group: {group_id}")

        result_msg = self._ensure_result_message(group_id)
        result_str, success = await self.execute_tool(tool_call)

        cp = ContentPart.create_tool_result(tool_call.get("id", ""), result_str)
        cp.raw_dict["name"] = tool_call.get("name", "")
        result_msg.raw_dict["content"].append(cp.raw_dict)

        return result_str, success

    # ------------------------------------------------------------------ #
    # Tool execution — standalone
    # ------------------------------------------------------------------ #

    async def execute_tool(self, tool_call: dict) -> tuple[str, bool]:
        """Execute a single tool call.

        Looks up the tool by name, casts arguments, fires the
        ``before_tool_execution`` deny hook, executes the tool, fires
        the ``after_tool_execution`` hook, and returns the result.

        Args:
            tool_call: Dict with keys ``name``, ``arguments``, ``id``.

        Returns:
            Tuple of (result_string, success_bool).
        """
        tool_name = tool_call.get("name", "")
        args = json.loads(tool_call.get("arguments", "{}"))

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
            result = tool.execute(**casted_args, session=None)
            if asyncio.iscoroutine(result):
                result = await result
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
