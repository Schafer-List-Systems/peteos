"""Unit tests for ExecutionEnvironment with tool groups."""

import asyncio
import json
from unittest.mock import MagicMock, AsyncMock

import pytest

from peteos.engine.executionenvironment import (
    ExecutionEnvironment,
    ToolCallGroup,
    ToolCallRecord,
    ToolApprovalStatus,
    ToolExecutionStatus,
    ApprovalEvent,
)
from peteos.conversation.message import Message
from peteos.conversation.message_registry import MessageRegistry


@pytest.fixture
def mock_role():
    return MagicMock()


@pytest.fixture
def mock_tool_manager():
    tm = MagicMock()
    tool = MagicMock()
    tool.func = lambda a, b: a + b
    tool.func.__name__ = "add"
    tool.parameters = {"a": {"type": "int"}, "b": {"type": "int"}}
    tool.execute = MagicMock(return_value=5)
    tm.get_tool.return_value = tool
    return tm


@pytest.fixture
def env(mock_role, mock_tool_manager):
    return ExecutionEnvironment(
        tool_manager=mock_tool_manager,
        role=mock_role,
        auto_approve_tools=[],
        tool_failure_policy="abort",
    )


class TestExecutionEnvironmentGroupCreation:
    """Test create_tool_group."""

    def test_create_tool_group(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        assert env.get_group("g1") is not None
        assert env.get_group("g1").id == "g1"
        assert env.get_group("g1").anchor_name == "g1:tool_result"

    def test_create_tool_group_duplicate(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        with pytest.raises(ValueError, match="already exists"):
            env.create_tool_group("g1", "g1:tool_result")


class TestExecutionEnvironmentAddToolCall:
    """Test add_tool_call with groups."""

    def test_add_tool_call_pending(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        tool_call = {"id": "tc1", "name": "add", "arguments": "{}"}
        rec = env.add_tool_call(tool_call, "g1")
        assert rec.approval_status == ToolApprovalStatus.PENDING
        assert rec.execution_status == ToolExecutionStatus.WAITING_FOR_APPROVAL

    def test_add_tool_call_auto_approved(self, env):
        env2 = ExecutionEnvironment(
            tool_manager=MagicMock(),
            role=MagicMock(),
            auto_approve_tools=["add"],
            tool_failure_policy="abort",
        )
        env2.create_tool_group("g1", "g1:tool_result")
        tool_call = {"id": "tc1", "name": "add", "arguments": "{}"}
        rec = env2.add_tool_call(tool_call, "g1")
        assert rec.approval_status == ToolApprovalStatus.APPROVED
        assert rec.execution_status == ToolExecutionStatus.EXECUTING

    def test_add_tool_call_unknown_group(self, env):
        with pytest.raises(ValueError, match="Unknown tool call group"):
            env.add_tool_call({"id": "tc1", "name": "add", "arguments": "{}"}, "nonexistent")

    def test_add_tool_call_duplicate_id(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        env.add_tool_call({"id": "tc1", "name": "add", "arguments": "{}"}, "g1")
        with pytest.raises(ValueError, match="Duplicate tool call id"):
            env.add_tool_call({"id": "tc1", "name": "add", "arguments": "{}"}, "g1")

    def test_add_tool_call_unknown_tool(self):
        mock_tm = MagicMock()
        mock_tm.get_tool.return_value = None
        env = ExecutionEnvironment(
            tool_manager=mock_tm, role=MagicMock(), auto_approve_tools=[], tool_failure_policy="abort"
        )
        env.create_tool_group("g1", "g1:tool_result")
        rec = env.add_tool_call({"id": "tc1", "name": "unknown", "arguments": "{}"}, "g1")
        assert rec.approval_status == ToolApprovalStatus.DENIED


class TestExecutionEnvironmentGroupQueries:
    """Test group-aware query methods."""

    def test_has_pending_tool_call_no_group(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        tool_call = {"id": "tc1", "name": "add", "arguments": "{}"}
        env.add_tool_call(tool_call, "g1")
        assert env.has_pending_tool_call() is True

    def test_has_pending_tool_call_with_group(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        tool_call = {"id": "tc1", "name": "add", "arguments": "{}"}
        env.add_tool_call(tool_call, "g1")
        assert env.has_pending_tool_call("g1") is True
        assert env.has_pending_tool_call("nonexistent") is False

    def test_has_reviewed_tool_call_all_pending(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        env.add_tool_call({"id": "tc1", "name": "add", "arguments": "{}"}, "g1")
        assert env.has_reviewed_tool_call("g1") is False

    def test_has_reviewed_tool_call_approved(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        env.add_tool_call({"id": "tc1", "name": "add", "arguments": "{}"}, "g1")
        group = env.get_group("g1")
        assert group.records[0].approval_status == ToolApprovalStatus.PENDING
        group.records[0].approval_status = ToolApprovalStatus.APPROVED
        assert env.has_reviewed_tool_call("g1") is True

    def test_has_unfinished_tool_call_executing(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        rec = env.add_tool_call({"id": "tc1", "name": "add", "arguments": "{}"}, "g1")
        rec.execution_status = ToolExecutionStatus.EXECUTING
        assert env.has_unfinished_tool_call("g1") is True

    def test_has_unfinished_tool_call_executed(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        rec = env.add_tool_call({"id": "tc1", "name": "add", "arguments": "{}"}, "g1")
        rec.execution_status = ToolExecutionStatus.EXECUTED
        assert env.has_unfinished_tool_call("g1") is False

    def test_get_group_id(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        env.add_tool_call({"id": "tc1", "name": "add", "arguments": "{}"}, "g1")
        assert env.get_group_id("tc1") == "g1"

    def test_get_group_id_unknown(self, env):
        assert env.get_group_id("nonexistent") is None

    def test_get_tool_calls_in_group(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        env.add_tool_call({"id": "tc1", "name": "add", "arguments": "{}"}, "g1")
        env.add_tool_call({"id": "tc2", "name": "add", "arguments": "{}"}, "g1")
        assert len(env.get_tool_calls_in_group("g1")) == 2


class TestExecutionEnvironmentApproval:
    """Test _handle_approval with groups."""

    def test_handle_approval_approved(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        env.add_tool_call({"id": "tc1", "name": "add", "arguments": "{}"}, "g1")
        event = ApprovalEvent(tool_call_id="tc1", approved=True)
        approved, group_id = env._handle_approval(event)
        assert approved is True
        assert group_id == "g1"
        group = env.get_group("g1")
        assert group.records[0].approval_status == ToolApprovalStatus.APPROVED

    def test_handle_approval_denied(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        env.add_tool_call({"id": "tc1", "name": "add", "arguments": "{}"}, "g1")
        event = ApprovalEvent(tool_call_id="tc1", approved=False)
        approved, group_id = env._handle_approval(event)
        assert approved is False
        assert group_id == "g1"
        group = env.get_group("g1")
        assert group.records[0].approval_status == ToolApprovalStatus.DENIED

    def test_handle_approval_unknown_tool_call(self, env):
        event = ApprovalEvent(tool_call_id="nonexistent", approved=True)
        approved, group_id = env._handle_approval(event)
        assert approved is False
        assert group_id is None


class TestExecutionEnvironmentPopPending:
    """Test pop_pending_tool_call."""

    def test_pop_approved_from_mixed(self, env):
        env.create_tool_group("g1", "g1:tool_result")
        env.add_tool_call({"id": "tc1", "name": "add", "arguments": "{}"}, "g1")
        env.add_tool_call({"id": "tc2", "name": "add", "arguments": "{}"}, "g1")
        group = env.get_group("g1")
        group.records[0].approval_status = ToolApprovalStatus.APPROVED
        # tc2 still PENDING
        rec = env.pop_pending_tool_call("g1")
        assert rec.tool_call_id == "tc1"
        assert len(env.get_tool_calls_in_group("g1")) == 1

    def test_pop_raises_on_empty(self, env):
        with pytest.raises(RuntimeError):
            env.pop_pending_tool_call("g1")

    def test_pop_raises_on_unknown_group(self, env):
        with pytest.raises(RuntimeError):
            env.pop_pending_tool_call("nonexistent")


class TestExecutionEnvironmentExecuteAndInject:
    """Test execute_and_inject."""

    @pytest.mark.asyncio
    async def test_execute_and_inject_creates_result_message(self, env, mock_tool_manager):
        tool = mock_tool_manager.get_tool("add")
        env.create_tool_group("g1", "g1:tool_result")
        env.add_tool_call({"id": "tc1", "name": "add", "arguments": "{}"}, "g1")
        result_str, success = await env.execute_and_inject({"id": "tc1", "name": "add", "arguments": "{}"}, "g1")
        assert success is True
        assert result_str == "5"
        group = env.get_group("g1")
        assert group.result_message is not None
        assert len(group.result_message.raw_dict["content"]) == 1
        cp = group.result_message.raw_dict["content"][0]
        assert cp["type"] == "tool_result"
        assert cp["call_id"] == "tc1"
        assert cp["name"] == "add"
        assert cp["content"] == "5"

    @pytest.mark.asyncio
    async def test_execute_and_inject_appends_multiple_results(self, env, mock_tool_manager):
        tool = mock_tool_manager.get_tool("add")
        tool.execute = MagicMock(side_effect=[5, 10, 15])
        env.create_tool_group("g1", "g1:tool_result")
        env.add_tool_call({"id": "tc1", "name": "add", "arguments": "{}"}, "g1")
        env.add_tool_call({"id": "tc2", "name": "add", "arguments": "{}"}, "g1")

        for i, tc_id in enumerate(["tc1", "tc2"]):
            await env.execute_and_inject({"id": tc_id, "name": "add", "arguments": "{}"}, "g1")

        group = env.get_group("g1")
        assert group.result_message is not None
        assert len(group.result_message.raw_dict["content"]) == 2
        assert group.result_message.raw_dict["content"][0]["call_id"] == "tc1"
        assert group.result_message.raw_dict["content"][1]["call_id"] == "tc2"

    @pytest.mark.asyncio
    async def test_execute_and_inject_fails_gracefully(self, env, mock_tool_manager):
        tool = mock_tool_manager.get_tool("add")
        tool.execute = MagicMock(side_effect=RuntimeError("kaboom"))
        env.create_tool_group("g1", "g1:tool_result")
        env.add_tool_call({"id": "tc1", "name": "add", "arguments": "{}"}, "g1")
        result_str, success = await env.execute_and_inject({"id": "tc1", "name": "add", "arguments": "{}"}, "g1")
        assert success is False
        assert "RuntimeError" in result_str
        group = env.get_group("g1")
        assert group.result_message is not None
        assert len(group.result_message.raw_dict["content"]) == 1

    @pytest.mark.asyncio
    async def test_execute_and_inject_tool_not_found(self, env):
        mock = MagicMock()
        mock.get_tool.return_value = None
        env2 = ExecutionEnvironment(
            tool_manager=mock, role=MagicMock(), auto_approve_tools=[], tool_failure_policy="abort"
        )
        env2.create_tool_group("g1", "g1:tool_result")
        env2.add_tool_call({"id": "tc1", "name": "nonexistent", "arguments": "{}"}, "g1")
        result_str, success = await env2.execute_and_inject(
            {"id": "tc1", "name": "nonexistent", "arguments": "{}"}, "g1"
        )
        assert success is False
        assert "not found" in result_str

    @pytest.mark.asyncio
    async def test_execute_and_inject_unknown_group(self):
        env = ExecutionEnvironment(
            tool_manager=MagicMock(), role=MagicMock(), auto_approve_tools=[], tool_failure_policy="abort"
        )
        with pytest.raises(ValueError, match="Unknown tool call group"):
            await env.execute_and_inject({"id": "tc1", "name": "add", "arguments": "{}"}, "nonexistent")
