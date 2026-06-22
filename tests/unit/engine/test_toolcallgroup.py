"""Unit tests for ToolCallGroup."""

from peteos.engine.executionenvironment import (
    ToolCallGroup,
    ToolCallRecord,
    ToolApprovalStatus,
    ToolExecutionStatus,
)


def test_toolcallgroup_initialization():
    """Test ToolCallGroup creation with required fields."""
    group = ToolCallGroup(id="test-1", anchor_name="test-1:tool_result")
    assert group.id == "test-1"
    assert group.anchor_name == "test-1:tool_result"
    assert group.records == []
    assert group.get_result_message() is None


def test_toolcallgroup_has_pending_no_records():
    """Test has_pending returns False when no records."""
    group = ToolCallGroup(id="g1", anchor_name="g1:tool_result")
    assert group.has_pending() is False


def test_toolcallgroup_has_pending_with_pending_record():
    """Test has_pending returns True when any record is PENDING."""
    group = ToolCallGroup(id="g1", anchor_name="g1:tool_result")
    rec = ToolCallRecord(
        tool_call_id="tc1",
        tool_call={"id": "tc1", "name": "test", "arguments": "{}"},
        approval_status=ToolApprovalStatus.PENDING,
    )
    group.add_tool_call(rec)
    assert group.has_pending() is True


def test_toolcallgroup_has_pending_with_approved_record():
    """Test has_pending returns False when all records are approved."""
    group = ToolCallGroup(id="g1", anchor_name="g1:tool_result")
    rec = ToolCallRecord(
        tool_call_id="tc1",
        tool_call={"id": "tc1", "name": "test", "arguments": "{}"},
        approval_status=ToolApprovalStatus.APPROVED,
    )
    group.add_tool_call(rec)
    assert group.has_pending() is False


def test_toolcallgroup_has_reviewed_empty():
    """Test has_reviewed returns False when no records."""
    group = ToolCallGroup(id="g1", anchor_name="g1:tool_result")
    assert group.has_reviewed() is False


def test_toolcallgroup_has_reviewed_with_pending_record():
    """Test has_reviewed returns False when first record is PENDING."""
    group = ToolCallGroup(id="g1", anchor_name="g1:tool_result")
    rec = ToolCallRecord(
        tool_call_id="tc1",
        tool_call={"id": "tc1", "name": "test", "arguments": "{}"},
        approval_status=ToolApprovalStatus.PENDING,
    )
    group.add_tool_call(rec)
    assert group.has_reviewed() is False


def test_toolcallgroup_has_reviewed_with_approved_record():
    """Test has_reviewed returns True when first record is APPROVED."""
    group = ToolCallGroup(id="g1", anchor_name="g1:tool_result")
    rec = ToolCallRecord(
        tool_call_id="tc1",
        tool_call={"id": "tc1", "name": "test", "arguments": "{}"},
        approval_status=ToolApprovalStatus.APPROVED,
    )
    group.add_tool_call(rec)
    assert group.has_reviewed() is True


def test_toolcallgroup_has_reviewed_mixed():
    """Test has_reviewed returns True when first record is approved despite pending later records."""
    group = ToolCallGroup(id="g1", anchor_name="g1:tool_result")
    rec1 = ToolCallRecord(
        tool_call_id="tc1",
        tool_call={"id": "tc1", "name": "test", "arguments": "{}"},
        approval_status=ToolApprovalStatus.APPROVED,
    )
    rec2 = ToolCallRecord(
        tool_call_id="tc2",
        tool_call={"id": "tc2", "name": "test2", "arguments": "{}"},
        approval_status=ToolApprovalStatus.PENDING,
    )
    group.add_tool_call(rec1)
    group.add_tool_call(rec2)
    assert group.has_reviewed() is True


def test_toolcallgroup_has_unfinished_executing():
    """Test has_unfinished returns True when execution status is EXECUTING."""
    group = ToolCallGroup(id="g1", anchor_name="g1:tool_result")
    rec = ToolCallRecord(
        tool_call_id="tc1",
        tool_call={"id": "tc1", "name": "test", "arguments": "{}"},
        approval_status=ToolApprovalStatus.APPROVED,
        execution_status=ToolExecutionStatus.EXECUTING,
    )
    group.add_tool_call(rec)
    assert group.has_unfinished() is True


def test_toolcallgroup_has_unfinished_executed():
    """Test has_unfinished returns False when all records are EXECUTED."""
    group = ToolCallGroup(id="g1", anchor_name="g1:tool_result")
    rec = ToolCallRecord(
        tool_call_id="tc1",
        tool_call={"id": "tc1", "name": "test", "arguments": "{}"},
        approval_status=ToolApprovalStatus.APPROVED,
        execution_status=ToolExecutionStatus.EXECUTED,
    )
    group.add_tool_call(rec)
    assert group.has_unfinished() is False


def test_toolcallgroup_pop_first_reviewed_approved():
    """Test pop_first_reviewed returns first approved record and removes it."""
    group = ToolCallGroup(id="g1", anchor_name="g1:tool_result")
    rec1 = ToolCallRecord(
        tool_call_id="tc1",
        tool_call={"id": "tc1", "name": "test", "arguments": "{}"},
        approval_status=ToolApprovalStatus.APPROVED,
    )
    rec2 = ToolCallRecord(
        tool_call_id="tc2",
        tool_call={"id": "tc2", "name": "test2", "arguments": "{}"},
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
    group = ToolCallGroup(id="g1", anchor_name="g1:tool_result")
    rec1 = ToolCallRecord(
        tool_call_id="tc1",
        tool_call={"id": "tc1", "name": "test", "arguments": "{}"},
        approval_status=ToolApprovalStatus.PENDING,
    )
    rec2 = ToolCallRecord(
        tool_call_id="tc2",
        tool_call={"id": "tc2", "name": "test2", "arguments": "{}"},
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
    group = ToolCallGroup(id="g1", anchor_name="g1:tool_result")
    rec = ToolCallRecord(
        tool_call_id="tc1",
        tool_call={"id": "tc1", "name": "test", "arguments": "{}"},
        approval_status=ToolApprovalStatus.PENDING,
    )
    group.add_tool_call(rec)

    assert group.pop_first_reviewed() is None


def test_toolcallgroup_result_message_lifecycle():
    """Test setting and getting result message."""
    from peteos.conversation.message import Message

    group = ToolCallGroup(id="g1", anchor_name="g1:tool_result")
    assert group.get_result_message() is None

    msg = Message.create(role="tool_result", content_parts=[])
    group.set_result_message(msg)
    assert group.get_result_message() is msg

    # Replacing is allowed
    msg2 = Message.create(role="tool_result", content_parts=[])
    group.set_result_message(msg2)
    assert group.get_result_message() is msg2


def test_toolcallgroup_add_multiple_records():
    """Test adding multiple records to a group."""
    group = ToolCallGroup(id="g1", anchor_name="g1:tool_result")

    for i in range(5):
        rec = ToolCallRecord(
            tool_call_id=f"tc{i}",
            tool_call={"id": f"tc{i}", "name": "test", "arguments": "{}"},
            approval_status=ToolApprovalStatus.APPROVED,
        )
        group.add_tool_call(rec)

    assert len(group.records) == 5
