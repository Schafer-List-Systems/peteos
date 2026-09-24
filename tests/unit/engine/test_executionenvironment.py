"""Unit tests for ExecutionEnvironment with tool groups."""

import asyncio
from unittest.mock import MagicMock, AsyncMock

import pytest

from peteos.utils import json

from peteos.engine.executionenvironment import (
    ExecutionEnvironment,
    ToolCallGroup,
    ToolCallRecord,
    ToolApprovalStatus,
    ToolExecutionStatus,
    ApprovalEvent,
)
from peteos.conversation.message import ContentPart


class TestExecutionEnvironmentGroupCreation:
    """Test create_tool_group."""

    def test_create_tool_group(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        fg = env.get_foreground_group()
        assert fg is not None
        assert fg.id == "g1"
        assert fg.anchor_name == "g1:tool_result"

    def test_create_tool_group_duplicate(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        with pytest.raises(ValueError, match="already exists"):
            env.create_tool_group("g1", "g1:tool_result")


class TestExecutionEnvironmentAddToolCall:
    """Test add_tool_call with foreground group."""

    def test_add_tool_call_pending(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "add", "{}")
        rec = env.add_tool_call(tc, runner=env._runner)
        assert rec.approval_status == ToolApprovalStatus.PENDING
        assert rec.execution_status == ToolExecutionStatus.WAITING_FOR_APPROVAL

    def test_add_tool_call_auto_approved(self, mock_runner):
        mock_tm = MagicMock()
        tool = MagicMock()
        tool.func = lambda a, b: a + b
        tool.func.__name__ = "add"
        tool.parameters = {"a": {"type": "int"}, "b": {"type": "int"}}
        tool.execute = MagicMock(return_value=5)
        mock_tm.get_tool.return_value = tool
        env = ExecutionEnvironment(
            tool_manager=mock_tm,
            role=mock_runner.role,
            auto_approve_tools=["add"],
            tool_failure_policy="abort",
        )
        env.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "add", "{}")
        rec = env.add_tool_call(tc, runner=mock_runner)
        assert rec.approval_status == ToolApprovalStatus.APPROVED
        assert rec.execution_status == ToolExecutionStatus.WAITING_FOR_EXECUTION

    def test_add_tool_call_unknown_tool(self, mock_runner):
        mock_tm = MagicMock()
        mock_tm.get_tool.return_value = None
        env = ExecutionEnvironment(
            tool_manager=mock_tm, role=mock_runner.role, auto_approve_tools=[], tool_failure_policy="abort"
        )
        env.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "unknown", "{}")
        rec = env.add_tool_call(tc, runner=mock_runner)
        assert rec.approval_status == ToolApprovalStatus.DENIED


class TestExecutionEnvironmentGroupQueries:
    """Test foreground-group query methods."""

    def test_has_pending_tool_call(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        tc = ContentPart.create_tool_use("tc1", "add", "{}")
        env.add_tool_call(tc, runner=env._runner)
        fg = env.get_foreground_group()
        assert fg is not None
        assert fg.has_pending() is True

    def test_has_reviewed_tool_call_all_pending(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        env.add_tool_call(ContentPart.create_tool_use("tc1", "add", "{}"), runner=env._runner)
        assert env.get_foreground_group().has_reviewed() is False

    def test_has_reviewed_tool_call_approved(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        rec = env.add_tool_call(ContentPart.create_tool_use("tc1", "add", "{}"), runner=env._runner)
        assert rec.approval_status == ToolApprovalStatus.PENDING
        rec.approval_status = ToolApprovalStatus.APPROVED
        assert env.get_foreground_group().has_reviewed() is True

    def test_has_unfinished_tool_call_executing(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        rec = env.add_tool_call(ContentPart.create_tool_use("tc1", "add", "{}"), runner=env._runner)
        rec.execution_status = ToolExecutionStatus.EXECUTING
        assert env.get_foreground_group().has_unfinished() is True

    def test_has_unfinished_tool_call_executed(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        rec = env.add_tool_call(ContentPart.create_tool_use("tc1", "add", "{}"), runner=env._runner)
        rec.execution_status = ToolExecutionStatus.EXECUTED
        assert env.get_foreground_group().has_unfinished() is False

    def test_get_tool_calls_in_group(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        env.add_tool_call(ContentPart.create_tool_use("tc1", "add", "{}"), runner=env._runner)
        env.add_tool_call(ContentPart.create_tool_use("tc2", "add", "{}"), runner=env._runner)
        assert len(env.get_foreground_group().records) == 2


class TestExecutionEnvironmentApproval:
    """Test _handle_approval with foreground group."""

    def test_handle_approval_approved(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        env.add_tool_call(ContentPart.create_tool_use("tc1", "add", "{}"), runner=env._runner)
        event = ApprovalEvent(tool_call_id="tc1", approved=True)
        approved, group_id = env._handle_approval(event)
        assert approved is True
        assert group_id == "g1"
        fg = env.get_foreground_group()
        assert fg.records[0].approval_status == ToolApprovalStatus.APPROVED

    def test_handle_approval_denied(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        env.add_tool_call(ContentPart.create_tool_use("tc1", "add", "{}"), runner=env._runner)
        event = ApprovalEvent(tool_call_id="tc1", approved=False)
        approved, group_id = env._handle_approval(event)
        assert approved is False
        assert group_id == "g1"
        fg = env.get_foreground_group()
        assert fg.records[0].approval_status == ToolApprovalStatus.DENIED

    def test_handle_approval_unknown_tool_call(self, env):
        event = ApprovalEvent(tool_call_id="nonexistent", approved=True)
        approved, group_id = env._handle_approval(event)
        assert approved is False
        assert group_id is None


class TestExecutionEnvironmentPopPending:
    """Test pop_first_reviewed."""

    def test_pop_approved_from_mixed(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        env.add_tool_call(ContentPart.create_tool_use("tc1", "add", "{}"), runner=env._runner)
        env.add_tool_call(ContentPart.create_tool_use("tc2", "add", "{}"), runner=env._runner)
        fg = env.get_foreground_group()
        fg.records[0].approval_status = ToolApprovalStatus.APPROVED
        rec = fg.pop_first_reviewed()
        assert rec.tool_call_id == "tc1"
        assert len(fg.records) == 1

    def test_pop_raises_on_empty(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        rec = env.get_foreground_group().pop_first_reviewed()
        assert rec is None
