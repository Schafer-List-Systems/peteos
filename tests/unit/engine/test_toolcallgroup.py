"""Unit tests for ToolCallGroup."""

from peteos.conversation.message import ContentPart, Message

from peteos.engine.executionenvironment import (
    ToolCallGroup,
    ToolCallRecord,
    ToolApprovalStatus,
    ToolExecutionStatus,
)


def _make_group(id: str, anchor_name: str) -> ToolCallGroup:
    """Helper to create a ToolCallGroup with a valid result_message."""
    return ToolCallGroup(
        id=id,
        anchor_name=anchor_name,
        result_message=Message.create("tool_result", []),
    )


def _make_tc(tool_call_id: str, name: str) -> ContentPart:
    """Helper to create a ContentPart tool_use."""
    return ContentPart.create_tool_use(tool_call_id, name, "{}")


def test_toolcallgroup_initialization():
    """Test ToolCallGroup creation with required fields."""
    group = _make_group("test-1", "test-1:tool_result")
    assert group.id == "test-1"
    assert group.anchor_name == "test-1:tool_result"
    assert group.records == []
    assert group.result_message is not None
    assert group.result_message.role == "tool_result"


def test_toolcallgroup_has_pending_no_records():
    """Test has_pending returns False when no records."""
    group = _make_group("g1", "g1:tool_result")
    assert group.has_pending() is False


def test_toolcallgroup_has_pending_with_pending_record():
    """Test has_pending returns True when any record is PENDING."""
    group = _make_group("g1", "g1:tool_result")
    rec = ToolCallRecord(
        tool_call_id="tc1",
        tool_call=_make_tc("tc1", "test"),
        approval_status=ToolApprovalStatus.PENDING,
    )
    group.add_tool_call(rec)
    assert group.has_pending() is True


def test_toolcallgroup_has_pending_with_approved_record():
    """Test has_pending returns False when all records are approved."""
    group = _make_group("g1", "g1:tool_result")
    rec = ToolCallRecord(
        tool_call_id="tc1",
        tool_call=_make_tc("tc1", "test"),
        approval_status=ToolApprovalStatus.APPROVED,
    )
    group.add_tool_call(rec)
    assert group.has_pending() is False


def test_toolcallgroup_has_reviewed_empty():
    """Test has_reviewed returns False when no records."""
    group = _make_group("g1", "g1:tool_result")
    assert group.has_reviewed() is False


def test_toolcallgroup_has_reviewed_with_pending_record():
    """Test has_reviewed returns False when first record is PENDING."""
    group = _make_group("g1", "g1:tool_result")
    rec = ToolCallRecord(
        tool_call_id="tc1",
        tool_call=_make_tc("tc1", "test"),
        approval_status=ToolApprovalStatus.PENDING,
    )
    group.add_tool_call(rec)
    assert group.has_reviewed() is False


def test_toolcallgroup_has_reviewed_with_approved_record():
    """Test has_reviewed returns True when first record is APPROVED."""
    group = _make_group("g1", "g1:tool_result")
    rec = ToolCallRecord(
        tool_call_id="tc1",
        tool_call=_make_tc("tc1", "test"),
        approval_status=ToolApprovalStatus.APPROVED,
    )
    group.add_tool_call(rec)
    assert group.has_reviewed() is True


def test_toolcallgroup_has_reviewed_mixed():
    """Test has_reviewed returns True when first record is approved despite pending later records."""
    group = _make_group("g1", "g1:tool_result")
    rec1 = ToolCallRecord(
        tool_call_id="tc1",
        tool_call=_make_tc("tc1", "test"),
        approval_status=ToolApprovalStatus.APPROVED,
    )
    rec2 = ToolCallRecord(
        tool_call_id="tc2",
        tool_call=_make_tc("tc2", "test2"),
        approval_status=ToolApprovalStatus.PENDING,
    )
    group.add_tool_call(rec1)
    group.add_tool_call(rec2)
    assert group.has_reviewed() is True


def test_toolcallgroup_has_unfinished_executing():
    """Test has_unfinished returns True when execution status is EXECUTING."""
    group = _make_group("g1", "g1:tool_result")
    rec = ToolCallRecord(
        tool_call_id="tc1",
        tool_call=_make_tc("tc1", "test"),
        approval_status=ToolApprovalStatus.APPROVED,
        execution_status=ToolExecutionStatus.EXECUTING,
    )
    group.add_tool_call(rec)
    assert group.has_unfinished() is True


def test_toolcallgroup_has_unfinished_executed():
    """Test has_unfinished returns False when all records are EXECUTED."""
    group = _make_group("g1", "g1:tool_result")
    rec = ToolCallRecord(
        tool_call_id="tc1",
        tool_call=_make_tc("tc1", "test"),
        approval_status=ToolApprovalStatus.APPROVED,
        execution_status=ToolExecutionStatus.EXECUTED,
    )
    group.add_tool_call(rec)
    assert group.has_unfinished() is False


def test_toolcallgroup_pop_first_reviewed_approved():
    """Test pop_first_reviewed returns first approved record and removes it."""
    group = _make_group("g1", "g1:tool_result")
    rec1 = ToolCallRecord(
        tool_call_id="tc1",
        tool_call=_make_tc("tc1", "test"),
        approval_status=ToolApprovalStatus.APPROVED,
    )
    rec2 = ToolCallRecord(
        tool_call_id="tc2",
        tool_call=_make_tc("tc2", "test2"),
        approval_status=ToolApprovalStatus.PENDING,
    )
    group.add_tool_call(rec1)
    group.add_tool_call(rec2)

    popped = group.pop_first_reviewed()
    assert popped.tool_call_id == "tc1"
    assert len(group.records) == 1
    assert group.has_pending() is True


def test_toolcallgroup_pop_first_reviewed_skip_pending():
    """Test pop_first_reviewed skips PENDING records to find first APPROVED."""
    group = _make_group("g1", "g1:tool_result")
    rec1 = ToolCallRecord(
        tool_call_id="tc1",
        tool_call=_make_tc("tc1", "test"),
        approval_status=ToolApprovalStatus.PENDING,
    )
    rec2 = ToolCallRecord(
        tool_call_id="tc2",
        tool_call=_make_tc("tc2", "test2"),
        approval_status=ToolApprovalStatus.APPROVED,
    )
    group.add_tool_call(rec1)
    group.add_tool_call(rec2)

    popped = group.pop_first_reviewed()
    assert popped.tool_call_id == "tc2"
    assert len(group.records) == 1
    assert group.records[0].tool_call_id == "tc1"


def test_toolcallgroup_pop_first_reviewed_none():
    """Test pop_first_reviewed returns None when all records are PENDING."""
    group = _make_group("g1", "g1:tool_result")
    rec = ToolCallRecord(
        tool_call_id="tc1",
        tool_call=_make_tc("tc1", "test"),
        approval_status=ToolApprovalStatus.PENDING,
    )
    group.add_tool_call(rec)

    assert group.pop_first_reviewed() is None


def test_toolcallgroup_result_message_initialized():
    """Test result_message is always initialized."""
    group = _make_group("g1", "g1:tool_result")
    assert group.result_message is not None
    assert group.result_message.role == "tool_result"
    assert len(group.result_message.raw_dict["content"]) == 0


def test_toolcallgroup_add_multiple_records():
    """Test adding multiple records to a group."""
    group = _make_group("g1", "g1:tool_result")

    for i in range(5):
        rec = ToolCallRecord(
            tool_call_id=f"tc{i}",
            tool_call=_make_tc(f"tc{i}", "test"),
            approval_status=ToolApprovalStatus.APPROVED,
        )
        group.add_tool_call(rec)

    assert len(group.records) == 5
    assert len(group.result_message.raw_dict["content"]) == 5
