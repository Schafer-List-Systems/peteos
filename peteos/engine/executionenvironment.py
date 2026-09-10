"""ExecutionEnvironment — tool call management and execution."""

from __future__ import annotations

import asyncio
import inspect
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from peteos.utils import get_logger
from peteos.utils import json

from peteos.conversation.message import ContentPart, Message
from peteos.persona.role import Role
from peteos.persona.toolmanager import ToolManager

if TYPE_CHECKING:
    from peteos.engine.runner import Runner

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
    result_message: Message
    records: list[ToolCallRecord] = field(default_factory=list)
    _any_real_result: bool = field(default=False)

    @property
    def any_real_result(self) -> bool:
        return self._any_real_result

    def mark_real_result(self) -> None:
        self._any_real_result = True

    def add_tool_call(self, record: ToolCallRecord) -> None:
        self.records.append(record)
        cp = ContentPart.create_tool_result(record.tool_call_id, "")
        cp.raw_dict["name"] = record.tool_call.name
        self.result_message.raw_dict["content"].append(cp.raw_dict)

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

    def deny_all_remaining(self, reason: str) -> None:
        """Deny all records that haven't executed yet."""
        for r in self.records:
            if r.execution_status != ToolExecutionStatus.EXECUTED:
                r.approval_status = ToolApprovalStatus.DENIED
                r.execution_status = ToolExecutionStatus.DENIED
                r.denied_reason = f"Tool group denied: {reason}"


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
        group = ToolCallGroup(
            id=group_id,
            anchor_name=anchor_name,
            result_message=Message.create("tool_result", []),
        )
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

    def _handle_approval(self, event: ApprovalEvent) -> tuple[bool, Optional[str]]:
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
                    group.deny_all_remaining("Tool group denied by user.")
                    return False, group.id
        return False, None

    # ------------------------------------------------------------------ #
    # Result injection — EE owns result message and ContentPart injection.
    # ------------------------------------------------------------------ #

    async def execute_and_inject(self, tool_call: ContentPart, runner: "Runner") -> tuple[str | None, bool]:
        """Execute a tool and update the matching placeholder in the group's result message.

        The result message with placeholders is pre-created when the tool group
        is created. This method finds the placeholder by call_id and updates
        its content. Returns (None, True) for fire-and-forget tools without
        modifying the placeholder.

        Args:
            tool_call: ContentPart with type "tool_use".
            runner: Runner to inject as ``runner`` kwarg into tool calls.

        Returns:
            Tuple of (result_string_or_None, success_bool).
        """
        group = self._foreground_group
        if group is None:
            raise RuntimeError("No foreground tool call group")

        result_str, success = await self.execute_tool(tool_call, runner)
        if result_str is None:
            return None, success

        result_msg = group.result_message
        for cp_raw in result_msg.raw_dict["content"]:
            if cp_raw.get("type") == "tool_result" and cp_raw.get("call_id") == tool_call.call_id:
                cp_raw["content"] = result_str
                break

        group.mark_real_result()
        return result_str, success

    # ------------------------------------------------------------------ #
    # Tool execution — standalone
    # ------------------------------------------------------------------ #

    async def execute_tool(self, tool_call: ContentPart, runner: "Runner") -> tuple[str | None, bool]:
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

        try:
            casted_args = _cast_args_to_types(tool.func, args)
        except ValueError as e:
            _logger.debug("Tool %s argument coercion failed: %s", tool_name, e)
            return f"Error: {e}", False
        hook_result = await runner.call_hooks_deny("before_tool_execution", tool_call)
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
                await runner.call_hooks("after_tool_execution", tool_call, None, True)
                _logger.debug("Tool %s returned None (fire-and-forget)", tool_name)
                return None, True
            result_str = str(result)
            await runner.call_hooks("after_tool_execution", tool_call, result_str, True)
            _logger.debug("Tool %s returned: %s", tool_name, result_str)
            return result_str, True
        except Exception as e:
            _logger.debug("Tool %s raised an exception: %s", tool_name, str(e))
            return f"Error: {type(e).__name__}: {str(e)}", False


# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
# Type casting for tool arguments
# ------------------------------------------------------------------ #

def _cast_args_to_types(func: Callable, args: Dict[str, Any]) -> Dict[str, Any]:
    """Type-cast the arguments the LLM actually provided.

    Only casts values for parameters the LLM included in its call.
    Does not inject defaults or hallucinate missing arguments.
    """
    from peteos.utils._schema import _recursive_cast, _resolve_type

    signature = inspect.signature(func)
    globalns = dict(sys.modules[func.__module__].__dict__)
    casted_args: Dict[str, Any] = {}

    for param_name, value in args.items():
        if param_name not in signature.parameters:
            casted_args[param_name] = value
            continue

        param = signature.parameters[param_name]
        raw_annotation = param.annotation

        if raw_annotation == inspect.Parameter.empty or raw_annotation == Any:
            casted_args[param_name] = value
            continue

        annotation = _resolve_type(raw_annotation, globalns)

        try:
            casted_args[param_name] = _recursive_cast(value, annotation)
        except ValueError as e:
            raise ValueError(
                f"Parameter '{param_name}' of tool '{func.__name__}': {e}"
            ) from e

    return casted_args
