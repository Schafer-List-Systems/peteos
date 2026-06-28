"""Unit tests for ExecStatus."""

import pytest

from peteos.engine.exec_status import ExecStatus, _merge_exec_status, _EXEC_SEVERITY


class TestExecStatusMembers:
    """Test ExecStatus enum values exist and are strings."""

    @pytest.mark.parametrize(
        "member,expected",
        [
            (ExecStatus.FINISHED, "finished"),
            (ExecStatus.CONTINUE, "continue"),
            (ExecStatus.PENDING, "pending"),
            (ExecStatus.ERROR, "error"),
            (ExecStatus.TOOL_NOT_FOUND, "tool_not_found"),
            (ExecStatus.TOOL_FAILED, "tool_failed"),
            (ExecStatus.TOOL_DENIED, "tool_denied"),
        ],
    )
    def test_values(self, member, expected):
        assert member.value == expected

    def test_comparison_with_string(self):
        assert ExecStatus.FINISHED == "finished"
        assert ExecStatus.CONTINUE == "continue"
        assert "finished" == ExecStatus.FINISHED

    def test_iteration(self):
        members = list(ExecStatus)
        assert len(members) == 7


class TestExecSeverityMap:
    """Test _EXEC_SEVERITY ordering."""

    def test_none_is_lowest(self):
        assert _EXEC_SEVERITY[None] == -1

    def test_ordering_is_monotonic(self):
        values = [
            _EXEC_SEVERITY[ExecStatus.CONTINUE],
            _EXEC_SEVERITY[ExecStatus.FINISHED],
            _EXEC_SEVERITY[ExecStatus.PENDING],
            _EXEC_SEVERITY[ExecStatus.TOOL_NOT_FOUND],
            _EXEC_SEVERITY[ExecStatus.TOOL_DENIED],
            _EXEC_SEVERITY[ExecStatus.TOOL_FAILED],
            _EXEC_SEVERITY[ExecStatus.ERROR],
        ]
        assert values == sorted(values)

    def test_all_members_present(self):
        for member in ExecStatus:
            assert member in _EXEC_SEVERITY
        assert None in _EXEC_SEVERITY


class TestMergeExecStatus:
    """Test _merge_exec_status severity merging."""

    def test_none_returns_other(self):
        for status in ExecStatus:
            assert _merge_exec_status(None, status) is status
            assert _merge_exec_status(status, None) is status

    def test_none_none_is_none(self):
        assert _merge_exec_status(None, None) is None

    def test_same_status_returns_same(self):
        for status in ExecStatus:
            assert _merge_exec_status(status, status) is status

    def test_worst_wins_continue_vs_finished(self):
        assert _merge_exec_status(ExecStatus.CONTINUE, ExecStatus.FINISHED) is ExecStatus.FINISHED
        assert _merge_exec_status(ExecStatus.FINISHED, ExecStatus.CONTINUE) is ExecStatus.FINISHED

    def test_worst_wins_error_vs_continue(self):
        result = _merge_exec_status(ExecStatus.CONTINUE, ExecStatus.ERROR)
        assert result is ExecStatus.ERROR
        result = _merge_exec_status(ExecStatus.ERROR, ExecStatus.CONTINUE)
        assert result is ExecStatus.ERROR

    def test_tool_denied_vs_tool_failed(self):
        assert _merge_exec_status(ExecStatus.TOOL_DENIED, ExecStatus.TOOL_FAILED) is ExecStatus.TOOL_FAILED
        assert _merge_exec_status(ExecStatus.TOOL_FAILED, ExecStatus.TOOL_DENIED) is ExecStatus.TOOL_FAILED

    def test_commutive(self):
        pairs = [
            (ExecStatus.CONTINUE, ExecStatus.FINISHED),
            (ExecStatus.TOOL_DENIED, ExecStatus.TOOL_NOT_FOUND),
            (ExecStatus.ERROR, ExecStatus.PENDING),
        ]
        for a, b in pairs:
            assert _merge_exec_status(a, b) is _merge_exec_status(b, a)

    def test_associative(self):
        a, b, c = ExecStatus.CONTINUE, ExecStatus.TOOL_DENIED, ExecStatus.ERROR
        first = _merge_exec_status(_merge_exec_status(a, b), c)
        second = _merge_exec_status(a, _merge_exec_status(b, c))
        assert first is second
