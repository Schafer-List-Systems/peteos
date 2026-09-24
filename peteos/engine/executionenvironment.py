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


_APPROVAL_SEVERITY: dict[ToolApprovalStatus | None, int] = {
    None: -1,
    ToolApprovalStatus.PENDING: 0,
    ToolApprovalStatus.APPROVED: 1,
    ToolApprovalStatus.DENIED: 2,
}


def _merge_approval_status(
    current: ToolApprovalStatus,
    hook_result: Any,
) -> tuple[ToolApprovalStatus, str | None]:
    """Merge a hook result into the current approval status.

    Severity escalation:
    - None: no-op, keep current
    - True: upgrade PENDING→APPROVED only; no-op on APPROVED or DENIED
    - False or str: always DENIED; overwrites APPROVED or PENDING

    Returns (new_status, reason) — reason is current.denied_reason updated
    if the hook provided one.
    """
    if hook_result is None:
        return current, None

    if hook_result is True:
        if current == ToolApprovalStatus.PENDING:
            return ToolApprovalStatus.APPROVED, None
        return current, None

    reason = str(hook_result) if hook_result is not None else "Denied by on_tool_call hook"
    return ToolApprovalStatus.DENIED, reason


def _call_on_tool_call_hooks(
    runner: "Runner",
    role_name: str,
    tool_name: str,
    arguments: dict,
    respond: RespondHandle,
    initial_status: ToolApprovalStatus,
    initial_denied_reason: str | None = None,
) -> tuple[ToolApprovalStatus, str | None]:
    """Call all on_tool_call hooks and merge their approval verdicts.

    Every hook is called — this is a broadcast, not a short-circuit.
    Merging follows severity escalation: DENIED > APPROVED > PENDING.
    """
    hooks = runner._session.invocation_hooks.get("on_tool_call", [])

    denied_reasons: list[str] = []
    if initial_denied_reason:
        denied_reasons.append(initial_denied_reason)

    current_status = initial_status

    ctx = {
        "role": role_name,
        "session": runner._session,
        "tool_name": tool_name,
        "arguments": arguments,
        "approval_status": current_status,
        "denied_reason": initial_denied_reason,
        "respond": respond,
    }

    for hook in hooks:
        result = hook(ctx)
        current_status, hook_reason = _merge_approval_status(current_status, result)
        if hook_reason:
            denied_reasons.append(hook_reason)
        ctx["approval_status"] = current_status
        ctx["denied_reason"] = "\n".join(denied_reasons) if denied_reasons else None

    final_denied_reason = "\n".join(denied_reasons) if denied_reasons else None
    return current_status, final_denied_reason


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
    respond: Optional[RespondHandle] = None


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
class RespondHandle:
    """Injection handle for async tool call approval.

    Returned in the on_tool_call hook context. Captures the runner
    internally so the caller can approve or deny without a direct
    runner reference.
    """
    tool_call_id: str
    tool_call: ContentPart
    _runner: "Runner" = field(repr=False)

    def approve(self) -> None:
        """Approve the tool call."""
        from peteos.engine.executionenvironment import ApprovalEvent
        self._runner.push_event(ApprovalEvent(
            tool_call_id=self.tool_call_id,
            tool_call=self.tool_call,
            approved=True,
        ))

    def deny(self, reason: str | None = None) -> None:
        """Deny the tool call with an optional reason."""
        from peteos.engine.executionenvironment import ApprovalEvent
        self._runner.push_event(ApprovalEvent(
            tool_call_id=self.tool_call_id,
            tool_call=self.tool_call,
            approved=False,
        ))


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

    def add_tool_call(self, tool_call: ContentPart, runner: "Runner") -> ToolCallRecord:
        """Add a tool call to the foreground group. Returns the created record."""
        group = self._foreground_group
        if group is None:
            raise RuntimeError("No foreground tool call group")

        tc_id = tool_call.call_id
        if tc_id is None:
            raise ValueError("Tool call missing required 'call_id' field")

        tool_name = tool_call.name
        tool_args = json.loads(tool_call.arguments) if tool_call.arguments else {}

        if not self._tool_manager or not self._tool_manager.get_tool(tool_name):
            initial_status = ToolApprovalStatus.DENIED
            initial_denied_reason = f"Tool '{tool_name}' is not available for this agent"
            execution_status = ToolExecutionStatus.DENIED
        elif tool_name in self.auto_approve_tools:
            initial_status = ToolApprovalStatus.APPROVED
            initial_denied_reason = None
            execution_status = ToolExecutionStatus.EXECUTING
        else:
            initial_status = ToolApprovalStatus.PENDING
            initial_denied_reason = None
            execution_status = ToolExecutionStatus.WAITING_FOR_APPROVAL

        respond = RespondHandle(
            tool_call_id=tc_id,
            tool_call=tool_call,
            _runner=runner,
        )

        final_status, denied_reason = _call_on_tool_call_hooks(
            runner=runner,
            role_name=runner.role.name,
            tool_name=tool_name,
            arguments=tool_args,
            respond=respond,
            initial_status=initial_status,
            initial_denied_reason=initial_denied_reason,
        )

        record = ToolCallRecord(
            tool_call_id=tc_id,
            tool_call=tool_call,
            approval_status=final_status,
            execution_status=(
                ToolExecutionStatus.EXECUTING
                if final_status == ToolApprovalStatus.APPROVED
                else execution_status
            ),
            denied_reason=denied_reason,
            respond=respond,
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
        ``before_tool_execution`` deny hook, executes the tool, and returns
        the result. The ``after_tool_execution`` hook fires in
        ``runner._handle_tool_group`` after this returns.

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
                _logger.debug("Tool %s returned None (fire-and-forget)", tool_name)
                return None, True
            result_str = str(result)
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
