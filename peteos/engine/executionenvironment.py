"""ExecutionEnvironment — tool call management and execution."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from peteos.logger import get_logger

if TYPE_CHECKING:
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
        self._pending_tool_calls: list[ToolCallRecord] = []
        self._hooks: dict[str, list[Callable]] = {
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
    # Tool call queue management
    # ------------------------------------------------------------------ #

    def add_tool_call(self, tool_call: dict) -> None:
        """Add a tool call to the pending list, auto-approving if applicable."""
        tc_id = tool_call.get("id")
        if tc_id is None:
            raise ValueError("Tool call missing required 'id' field")
        for record in self._pending_tool_calls:
            if record.tool_call_id == tc_id:
                raise ValueError(f"Duplicate tool call id: {tc_id}")
        tool_name = tool_call.get("name", "")

        if not self._tool_manager:
            tool_call["denied_reason"] = f"Tool '{tool_name}' is not available for this agent"
            self._pending_tool_calls.append(ToolCallRecord(
                tool_call_id=tc_id,
                tool_call=tool_call,
                approval_status=ToolApprovalStatus.DENIED,
                execution_status=ToolExecutionStatus.DENIED,
            ))
            return

        if not self._tool_manager.get_tool(tool_name):
            tool_call["denied_reason"] = f"Tool '{tool_name}' is not available for this agent"
            self._pending_tool_calls.append(ToolCallRecord(
                tool_call_id=tc_id,
                tool_call=tool_call,
                approval_status=ToolApprovalStatus.DENIED,
                execution_status=ToolExecutionStatus.DENIED,
            ))
            return

        if tool_name in self.auto_approve_tools:
            self._pending_tool_calls.append(ToolCallRecord(
                tool_call_id=tc_id,
                tool_call=tool_call,
                approval_status=ToolApprovalStatus.APPROVED,
                execution_status=ToolExecutionStatus.EXECUTING,
            ))
        else:
            self._pending_tool_calls.append(ToolCallRecord(
                tool_call_id=tc_id,
                tool_call=tool_call,
                approval_status=ToolApprovalStatus.PENDING,
                execution_status=ToolExecutionStatus.WAITING_FOR_APPROVAL,
            ))

    def has_pending_tool_call(self) -> bool:
        """Check if any tool call needs user approval (approval status is PENDING)."""
        for record in self._pending_tool_calls:
            if record.approval_status == ToolApprovalStatus.PENDING:
                return True
        return False

    def has_unfinished_tool_call(self) -> bool:
        """Check if any tool call is not yet fully executed."""
        for record in self._pending_tool_calls:
            if record.execution_status != ToolExecutionStatus.EXECUTED:
                return True
        return False

    def has_reviewed_tool_call(self) -> bool:
        """Check if the first pending tool call has been approved (no longer PENDING)."""
        for record in self._pending_tool_calls:
            if record.approval_status != ToolApprovalStatus.PENDING:
                return True
            return False
        return False

    def get_pending_tool_calls(self) -> list[ToolCallRecord]:
        """Return all currently pending tool calls."""
        return list(self._pending_tool_calls)

    def pop_pending_tool_call(self) -> ToolCallRecord:
        """Remove and return the first tool call from the pending list."""
        if not self._pending_tool_calls:
            raise RuntimeError("No pending tool calls")
        return self._pending_tool_calls.pop(0)

    def is_tool_call_pending(self, tool_call_id: str) -> bool:
        """Check if a tool call needs user approval."""
        for record in self._pending_tool_calls:
            if record.tool_call_id == tool_call_id:
                return record.approval_status == ToolApprovalStatus.PENDING
        return False

    def _find_pending_record(self, tool_call_id: str) -> Optional[ToolCallRecord]:
        """Find a pending tool call record by its tool_call_id."""
        for record in self._pending_tool_calls:
            if record.tool_call_id == tool_call_id:
                return record
        return None

    def _handle_approval(self, event: Any) -> bool:
        """Handle an approval/denial event for a pending tool call.

        Returns True if the tool was approved (keep draining queue),
        False if the tool was denied (stop draining).
        """
        record = self._find_pending_record(event.tool_call_id)
        if record is None:
            return False

        if event.approved:
            record.approval_status = ToolApprovalStatus.APPROVED
            return True
        else:
            record.approval_status = ToolApprovalStatus.DENIED
            return False

    # ------------------------------------------------------------------ #
    # Tool execution
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
