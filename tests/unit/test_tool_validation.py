"""Tests for tool availability validation in Session.add_tool_call()."""

from unittest.mock import MagicMock

import pytest

from peteos.chatbot import ChatHistory, ChatBotManager
from peteos.role import Role
from peteos.session import Session, ToolApprovalStatus, ToolExecutionStatus
from peteos.toolmanager import ToolManager


def _make_mock_env():
    """Create a mock execution environment."""
    mock = MagicMock()
    mock._hooks = {"before_tool_execution": [], "after_tool_execution": [],
                   "before_notification_publish": [], "before_send_to_chatbot": [],
                   "before_loop_continue": [], "after_step": []}
    return mock


def _make_session(tool_manager: ToolManager, auto_approve: list[str] | None = None) -> Session:
    role = Role(name="test", description="A test role", auto_approve_tools=auto_approve or [])
    return Session(role=role, tool_manager=tool_manager, execution_environment=_make_mock_env())


class TestAddToolCallValidation:
    """Test that add_tool_call validates tool existence before queuing."""

    def test_hallucinated_tool_gets_denied(self):
        """Non-existent tool is queued as DENIED, not PENDING."""
        tm = ToolManager()
        session = _make_session(tm)
        session.add_tool_call({"id": "tc-1", "name": "nonexistent_tool", "input": {}})
        record = session._pending_tool_calls[0]
        assert record.approval_status == ToolApprovalStatus.DENIED
        assert record.execution_status == ToolExecutionStatus.DENIED

    def test_denied_tool_stores_reason(self):
        """Denied tool call stores the reason in the tool_call dict."""
        tm = ToolManager()
        session = _make_session(tm)
        session.add_tool_call({"id": "tc-1", "name": "fake_tool", "input": {}})
        record = session._pending_tool_calls[0]
        assert record.tool_call["denied_reason"] == "Tool 'fake_tool' is not available for this agent"

    def test_valid_tool_auto_approved(self):
        """Existing tool in auto_approve_tools gets APPROVED."""
        tm = ToolManager()
        tm.register_tool(func=lambda: None, name="real_tool", description="A real tool")
        session = _make_session(tm, auto_approve=["real_tool"])
        session.add_tool_call({"id": "tc-1", "name": "real_tool", "input": {}})
        record = session._pending_tool_calls[0]
        assert record.approval_status == ToolApprovalStatus.APPROVED
        assert record.execution_status == ToolExecutionStatus.EXECUTING

    def test_valid_tool_pending(self):
        """Existing tool not in auto_approve_tools gets PENDING."""
        tm = ToolManager()
        tm.register_tool(func=lambda: None, name="real_tool", description="A real tool")
        session = _make_session(tm, auto_approve=[])
        session.add_tool_call({"id": "tc-1", "name": "real_tool", "input": {}})
        record = session._pending_tool_calls[0]
        assert record.approval_status == ToolApprovalStatus.PENDING
        assert record.execution_status == ToolExecutionStatus.WAITING_FOR_APPROVAL

    def test_valid_tool_not_in_auto_approve_still_works(self):
        """Tool exists but not auto-approved → PENDING (existing behavior unchanged)."""
        tm = ToolManager()
        tm.register_tool(func=lambda: None, name="my_tool", description="A tool")
        session = _make_session(tm)
        session.add_tool_call({"id": "tc-1", "name": "my_tool", "input": {"x": 1}})
        record = session._pending_tool_calls[0]
        assert record.approval_status == ToolApprovalStatus.PENDING
        assert record.tool_call["input"] == {"x": 1}
