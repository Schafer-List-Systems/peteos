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
from peteos.utils.serialization import serialize

from peteos.conversation.message import ContentPart, Message, ToolApprovalStatus
from peteos.persona.role import Role
from peteos.persona.toolmanager import ToolManager

if TYPE_CHECKING:
    from peteos.engine.runner import Runner

_logger = get_logger(__name__)


class ToolExecutionStatus(str, Enum):
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    WAITING_FOR_EXECUTION = "waiting_for_execution"
    DENIED = "denied"
    EXECUTING = "executing"
    EXECUTED = "executed"
    ABORTED = "aborted"


class ApprovalDecision(str, Enum):
    """Possible decisions a hook can return from on_tool_call."""
    APPROVED = "approved"
    DENIED = "denied"
    PENDING = "pending"
    IGNORED = "ignored"


_APPROVAL_SEVERITY: dict[ToolApprovalStatus | None, int] = {
    None: -1,
    ToolApprovalStatus.PENDING: 0,
    ToolApprovalStatus.APPROVED: 1,
    ToolApprovalStatus.DENIED: 2,
}


def _coerce_to_approval_decision(
    result: Any,
) -> tuple[ApprovalDecision, str | None]:
    """Coerce any hook return value to an ApprovalDecision and optional denied_reason."""
    # If result is None, return IGNORED with no reason.
    if result is None:
        return ApprovalDecision.IGNORED, None

    # If result is True, return APPROVED with no reason.
    if result is True:
        return ApprovalDecision.APPROVED, None

    # If result is False, return DENIED with no reason.
    if result is False:
        return ApprovalDecision.DENIED, None
        
    # If result is already an ApprovalDecision, return it as-is.
    if isinstance(result, ApprovalDecision):
        return result, None

    # If result is a string, treat it as DENIED with the string as reason.
    if isinstance(result, str):
        return ApprovalDecision.DENIED, result

    # If result is a 2-tuple, unpack it and handle by type.
    if isinstance(result, tuple) and len(result) == 2:
        decision, denied_reason = result
        if isinstance(decision, ApprovalDecision):
            return decision, denied_reason
        if decision is True:
            return ApprovalDecision.APPROVED, denied_reason
        if decision is False or decision is None:
            return ApprovalDecision.DENIED, denied_reason
        return ApprovalDecision.DENIED, str(decision)

    # Fallback: coerce anything else to a string and return DENIED.
    return ApprovalDecision.DENIED, str(result)


def _merge_approval_status(
    current: ToolApprovalStatus,
    hook_result: Any,
) -> tuple[ToolApprovalStatus, str | None, bool]:
    """Merge a hook result into the current approval status.

    Returns (new_status, denied_reason, did_increment_count).
    - did_increment_count is True for APPROVED, DENIED, IGNORED
    - did_increment_count is False for PENDING (defer — hook will respond later)
    """
    # Coerce the hook result to an ApprovalDecision and optional denied_reason.
    decision, denied_reason = _coerce_to_approval_decision(hook_result)

    # If IGNORED, return current status with did_increment_count=True.
    if decision == ApprovalDecision.IGNORED:
        return current, None, True

    # If APPROVED, promote to APPROVED if still PENDING, else keep current; always increment.
    if decision == ApprovalDecision.APPROVED:
        if current == ToolApprovalStatus.PENDING:
            return ToolApprovalStatus.APPROVED, None, True
        return current, None, True

    # If DENIED, return DENIED with reason and increment.
    if decision == ApprovalDecision.DENIED:
        return ToolApprovalStatus.DENIED, denied_reason, True

    # If PENDING, keep current and do NOT increment (defer — hook will respond later).
    if decision == ApprovalDecision.PENDING:
        return current, None, False

    # Fallback: return current with increment (defensive, should be unreachable).
    return current, None, True


@dataclass
class ToolCallRecord:
    """Tracks the lifecycle of a tool call through approval and execution."""
    tool_call: ContentPart
    tool_result: ContentPart | None = None
    nextcloud_message_id: Optional[str] = None
    decisions: list[tuple[ToolApprovalStatus, str | None, bool]] = field(default_factory=list)
    responded_count: int = 0
    queued: bool = False


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

    def add_tool_call(self, tool_call: ContentPart) -> ToolCallRecord:
        """Create and link a record and its result placeholder, inject into the result message."""
        # Create result placeholder ...
        tool_result = ContentPart.create_tool_result(tool_call.call_id, "")

        # ... and inject the tool_result into the message, so the LLM sees a result slot.
        self.result_message.add_content(tool_result)

        # Create a record PENDING / WAITING_FOR_APPROVAL tool call record.
        record = ToolCallRecord(
            tool_call=tool_call,
            tool_result=tool_result
        )

        # Register and return the record.
        self.records.append(record)
        return record

    def has_pending(self) -> bool:
        return any(r.tool_call.approval_status == ToolApprovalStatus.PENDING for r in self.records)

    def has_reviewed(self) -> bool:
        for r in self.records:
            if r.tool_call.approval_status == ToolApprovalStatus.PENDING:
                return False
            return True
        return False

    def has_unfinished(self) -> bool:
        return any(r.tool_result.execution_status != ToolExecutionStatus.EXECUTED for r in self.records)

    def is_done(self) -> bool:
        """Return True when all tool calls are finalized (no more pending or approved)."""
        for r in self.records:
            if r.tool_call.approval_status not in (ToolApprovalStatus.DENIED, ToolApprovalStatus.APPROVED):
                return False
            if r.tool_call.approval_status == ToolApprovalStatus.APPROVED:
                if r.tool_result.execution_status != ToolExecutionStatus.EXECUTED:
                    return False
        return True

    def pop_first_reviewed(self) -> Optional[ToolCallRecord]:
        for i, r in enumerate(self.records):
            if r.tool_call.approval_status != ToolApprovalStatus.PENDING:
                return self.records.pop(i)
        return None

    def deny_all_remaining(self, reason: str) -> None:
        """Deny all records in this group that have not yet executed."""
        # Iterate over all tool call records in the group.
        for r in self.records:
            if r.tool_result.execution_status == ToolExecutionStatus.EXECUTED:
                continue

            # Mark the tool use as denied and the result as DENIED.
            r.tool_call.set_approval_status(ToolApprovalStatus.DENIED)
            r.tool_result.set_execution_status(ToolExecutionStatus.DENIED)
            r.tool_result.set_tool_result_content(f"[DENIED] {reason}")


def _merge_all_decisions(
    decisions: list[tuple[ToolApprovalStatus, str | None, bool]],
) -> tuple[ToolApprovalStatus, str | None]:
    """Merge all decisions into a final approval status and denied reason."""
    # Start with PENDING as default final status and an empty reasons list.
    current = ToolApprovalStatus.PENDING
    reasons: list[str] = []
    for status, reason, responded in decisions:
        if not responded:
            continue

        # Promote to DENIED and collect the reason.
        if status == ToolApprovalStatus.DENIED:
            current = ToolApprovalStatus.DENIED
            if reason:
                reasons.append(reason)

        # Promote to APPROVED only if still PENDING.
        elif status == ToolApprovalStatus.APPROVED:
            if current == ToolApprovalStatus.PENDING:
                current = ToolApprovalStatus.APPROVED

    # Join all reasons with newlines for the final denied reason.
    denied_reason = "\n".join(reasons) if reasons else None
    return current, denied_reason


async def _call_on_tool_call_hooks(
    hooks: list,
    record: "ToolCallRecord",
    runner: "Runner",
    role_name: str,
) -> None:
    """Call each hook and fill its decision slot in the record.

    Mutates record in place: decisions, responded_count.
    Stops early if all hooks have responded.
    """
    for i, hook in enumerate(hooks):
        # build respond handle so hooks can defer their decision to a later turn
        respond = RespondHandle(
            tool_call_id=record.tool_call.call_id,
            tool_call=record.tool_call,
            _runner=runner,
            _record=record,
            hook_index=i,
        )

        # give each hook the accumulated decision so far so it can see prior votes
        merged_status, merged_reason = _merge_all_decisions(record.decisions)
        ctx = {
            "role": role_name,
            "session": runner._session,
            "tool_name": record.tool_call.name,
            "arguments": json.loads(record.tool_call.arguments) if record.tool_call.arguments else {},
            "approval_status": merged_status,
            "denied_reason": merged_reason,
            "respond": respond,
        }

        # sync or async: await if hook returned a coroutine
        result = hook(ctx)
        if inspect.isawaitable(result):
            result = await result
            
        # record the vote; only increment count for votes that actually voted
        new_status, new_reason, did_increment = _merge_approval_status(merged_status, result)
        if did_increment:
            record.decisions[i] = (new_status, new_reason, True)
            record.responded_count += 1


@dataclass
class ApprovalEvent:
    """Event pushed to Runner.event_queue to signal approval of a tool call."""
    tool_call_id: str = ""
    tool_call: ContentPart = field(default_factory=lambda: ContentPart({"type": "tool_use"}))
    approved: bool = True
    denied_reason: str | None = None


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
    _record: "ToolCallRecord" = field(repr=False, default=None)
    hook_index: int = 0

    def _bind(self, record: "ToolCallRecord", hook_index: int) -> None:
        """Bind this handle to a specific hook slot in a record."""
        self._record = record
        self.hook_index = hook_index

    def approve(self) -> None:
        """Respond to this hook slot with approval."""
        # Return early if record is unbound (no-op).
        if self._record is None:
            return

        status, reason, responded = self._record.decisions[self.hook_index]
        if responded:
            raise RuntimeError(
                f"Hook at index {self.hook_index} already responded with {status.value}"
            )

        # Record the APPROVED vote in this hook's decision slot.
        self._record.decisions[self.hook_index] = (ToolApprovalStatus.APPROVED, None, True)
        self._record.responded_count += 1

        # If all hooks have responded and the record is queued, finalize.
        if (
            self._record.responded_count == len(self._record.decisions)
            and self._record.queued
        ):
            # Merge all decisions into a final status and reason; commit to the tool call.
            final_status, final_reason = _merge_all_decisions(self._record.decisions)
            self._record.tool_call.set_approval_status(final_status)

            # For approved: advance the result to WAITING_FOR_EXECUTION.
            if final_status == ToolApprovalStatus.APPROVED:
                self._record.tool_result.set_execution_status(ToolExecutionStatus.WAITING_FOR_EXECUTION)

            # Push an ApprovalEvent so the runner can process the final decision.
            self._runner.push_event(ApprovalEvent(
                tool_call_id=self.tool_call_id,
                tool_call=self.tool_call,
                approved=True,
                denied_reason=final_reason,
            ))

    def deny(self, reason: str | None = None) -> None:
        """Respond to this hook slot with denial."""
        # Return early if record is unbound (no-op).
        if self._record is None:
            return

        status, _, responded = self._record.decisions[self.hook_index]
        if responded:
            raise RuntimeError(
                f"Hook at index {self.hook_index} already responded with {status.value}"
            )

        # Record the DENIED vote in this hook's decision slot.
        msg = reason or "Denied by on_tool_call hook"
        self._record.decisions[self.hook_index] = (ToolApprovalStatus.DENIED, msg, True)
        self._record.responded_count += 1

        # If all hooks have responded and the record is queued, finalize.
        if (
            self._record.responded_count == len(self._record.decisions)
            and self._record.queued
        ):
            # Merge all decisions into a final status and reason; commit to the tool call.
            final_status, final_reason = _merge_all_decisions(self._record.decisions)
            self._record.tool_call.set_approval_status(final_status)

            # For denied: mark the result DENIED and embed the reason in the content.
            if final_status == ToolApprovalStatus.DENIED:
                self._record.tool_result.set_execution_status(ToolExecutionStatus.DENIED)
                self._record.tool_result.set_tool_result_content(f"[DENIED] {final_reason}")

            # Push an ApprovalEvent so the runner can process the final decision.
            self._runner.push_event(ApprovalEvent(
                tool_call_id=self.tool_call_id,
                tool_call=self.tool_call,
                approved=False,
                denied_reason=final_reason,
            ))


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

    async def add_tool_call(self, tool_call: ContentPart, runner: "Runner") -> ToolCallRecord:
        """Add a tool call to the foreground group. Returns the created record."""
        # Fail fast if no foreground group.
        if self._foreground_group is None:
            raise RuntimeError("No foreground tool call group")

        # Fail fast if the tool call has no call_id.
        if tool_call.call_id is None:
            raise ValueError("Tool call missing required 'call_id' field")

        # Create the record and its result placeholder via the group.
        record = self._foreground_group.add_tool_call(tool_call)

        def _existence_check(ctx):
            if not self._tool_manager or not self._tool_manager.get_tool(tool_call.name):
                return f"Tool '{tool_call.name}' is not available for this agent"
            return None

        def _auto_approve(ctx):
            if tool_call.name in self.auto_approve_tools:
                return True
            return None

        # Gather all hooks: existence check and auto_approve prepended, then user hooks.
        user_hooks = list(runner.session.invocation_hooks.get("on_tool_call", []))
        hooks = [_existence_check, _auto_approve] + user_hooks

        # Size the decisions array to the hook count; call hooks synchronously.
        record.decisions = [
            (ToolApprovalStatus.PENDING, None, False)
            for _ in hooks
        ]
        await _call_on_tool_call_hooks(hooks, record, runner, runner.role.name)

        # All hooks responded — merge decisions and finalize the record.
        if record.responded_count == len(record.decisions):
            final_status, final_reason = _merge_all_decisions(record.decisions)

            # Commit the final approval status to the tool call.
            record.tool_call.set_approval_status(final_status)

            # For approved: advance result to WAITING_FOR_EXECUTION.
            if final_status == ToolApprovalStatus.APPROVED:
                record.tool_result.set_execution_status(ToolExecutionStatus.WAITING_FOR_EXECUTION)

            # For denied with a reason: embed the reason in the result placeholder.
            if final_status == ToolApprovalStatus.DENIED and final_reason:
                record.tool_result.set_tool_result_content(f"[DENIED] {final_reason}")

            record.queued = False
            return record

        # Deferred path: at least one hook will respond asynchronously later.
        record.queued = True
        return record

    def get_pending_tool_calls(self) -> list[ToolCallRecord]:
        """Return all pending tool call records from the foreground group."""
        group = self._foreground_group
        if group is None:
            return []
        return [r for r in group.records if r.tool_call.approval_status == ToolApprovalStatus.PENDING]

    def find_pending_record(self, tool_call_id: str) -> Optional[ToolCallRecord]:
        """Find a tool call record by its tool_call_id in the foreground group."""
        group = self._foreground_group
        if group is None:
            return None
        for record in group.records:
            if record.tool_call.call_id == tool_call_id:
                return record
        return None

    def _handle_approval(self, event: ApprovalEvent) -> tuple[bool, Optional[str]]:
        """Handle an approval/denial event for a pending tool call.

        If the tool call is denied, all remaining PENDING records in the
        group are also denied.
        """
        # No foreground group → event is stale, ignore.
        group = self._foreground_group
        if group is None:
            return False, None

        # Find the record this event is targeting.
        for record in group.records:
            if record.tool_call.call_id == event.tool_call_id:
                if event.approved:
                    record.tool_call.set_approval_status(ToolApprovalStatus.APPROVED)
                    return True, group.id

                # Denied: update tool_use, set result content, cascade denial.
                else:
                    record.tool_call.set_approval_status(ToolApprovalStatus.DENIED)
                    record.tool_result.set_tool_result_content(f"[DENIED] {event.denied_reason}")
                    record.tool_result.raw_dict.pop("pending", None)
                    group.deny_all_remaining(f"Denied as consequence of '{record.tool_call.name}' being denied in the same response.")
                    return False, group.id
        return False, None

    # ------------------------------------------------------------------ #
    # Result injection — EE owns result message and ContentPart injection.
    # ------------------------------------------------------------------ #

    async def execute_and_inject(self, record: ToolCallRecord, runner: "Runner") -> tuple[str | None, bool]:
        """Execute a tool and update the matching placeholder in the group's result message.

        The result message with placeholders is pre-created when the tool group
        is created. This method finds the placeholder by call_id and updates
        its content. Returns (None, True) for fire-and-forget tools without
        modifying the placeholder.

        Args:
            record: ToolCallRecord with tool_call of type "tool_use".
            runner: Runner to inject as ``runner`` kwarg into tool calls.

        Returns:
            Tuple of (result_string_or_None, success_bool).
        """
        # no foreground group → programming error, not a tool error
        group = self._foreground_group
        if group is None:
            raise RuntimeError("No foreground tool call group")

        # Mark the record as actively running; EE owns this transition.
        record.tool_result.set_execution_status(ToolExecutionStatus.EXECUTING)
        result_str, success = await self.execute_tool(record.tool_call, runner)
        record.tool_result.set_execution_status(ToolExecutionStatus.EXECUTED)

        # Inject result into the tool_result placeholder so the LLM sees it.
        if result_str is not None:
            group.mark_real_result()
            record.tool_result.set_tool_result_content(result_str)

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
        # Extract tool name and parse JSON arguments from the tool call.
        tool_name = tool_call.name
        args = json.loads(tool_call.arguments or "{}")

        _logger.debug("Attempting to execute tool: %s(%s)", tool_name, args)

        # Look up the tool in the tool manager; return error if not found.
        tool = self._tool_manager.get_tool(tool_name)
        if not tool:
            _logger.warning("Tool '%s' not found", tool_name)
            return f"Error: Tool '{tool_name}' not found", False

        # Try to cast the arguments to the tool's parameter types; return error on failure.
        try:
            casted_args = _cast_args_to_types(tool.func, args)
        except ValueError as e:
            _logger.debug("Tool %s argument coercion failed: %s", tool_name, e)
            return f"Error: {e}", False

        # Call the before_tool_execution deny hook; return error if denied.
        hook_result = await runner.call_hooks_deny("before_tool_execution", tool_call)
        if hook_result is not None:
            allow, message = hook_result
            if not allow:
                _logger.debug("Tool %s denied by hook: %s", tool_name, message)
                return message, False

        # Execute the tool; if the result is a coroutine, await it.
        try:
            result = tool.execute(**casted_args, runner=runner)
            if asyncio.iscoroutine(result):
                result = await result

            # If result is None, return (None, True) for fire-and-forget.
            if result is None:
                _logger.debug("Tool %s returned None (fire-and-forget)", tool_name)
                return None, True

            # Serialize the result and return it with success.
            result_str = serialize(result)
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
    # Import the recursive cast and type resolver utilities.
    from peteos.utils._schema import _recursive_cast, _resolve_type

    # Inspect the function signature to get parameter metadata.
    signature = inspect.signature(func)
    # Build a namespace dict from the function's module for type resolution.
    globalns = dict(sys.modules[func.__module__].__dict__)
    casted_args: Dict[str, Any] = {}

    # Iterate over each (param_name, value) pair from the JSON args.
    for param_name, value in args.items():
        # If the param is not in the signature, pass it through unchanged.
        if param_name not in signature.parameters:
            casted_args[param_name] = value
            continue

        param = signature.parameters[param_name]
        raw_annotation = param.annotation

        # If no annotation or Any, pass it through unchanged.
        if raw_annotation == inspect.Parameter.empty or raw_annotation == Any:
            casted_args[param_name] = value
            continue

        # Resolve the raw annotation to an actual type using the namespace.
        annotation = _resolve_type(raw_annotation, globalns)

        # Try to recursively cast the value to the resolved annotation.
        try:
            casted_args[param_name] = _recursive_cast(value, annotation)
        except ValueError as e:
            raise ValueError(
                f"Parameter '{param_name}' of tool '{func.__name__}': {e}"
            ) from e

    return casted_args
