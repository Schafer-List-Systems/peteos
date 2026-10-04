"""Unit tests for approval_voting — channel-driven tool call approval."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from peteos.chatbot import ContentPart
from peteos.engine.approval_voting import (
    _coerce_to_approval_decision,
    _merge_approval_status,
    _resolve_decision,
    ApprovalDecision,
    ApprovalElection,
)


# ------------------------------------------------------------------ #
# _coerce_to_approval_decision
# ------------------------------------------------------------------ #

class TestCoerceToApprovalDecision:
    """Coercion from hook return values to ApprovalDecision."""

    @pytest.mark.parametrize(
        "input_,expected_decision,expected_reason",
        [
            (None, ApprovalDecision.IGNORED, None),
            (True, ApprovalDecision.APPROVED, None),
            (False, ApprovalDecision.DENIED, None),
            (ApprovalDecision.APPROVED, ApprovalDecision.APPROVED, None),
            (ApprovalDecision.DENIED, ApprovalDecision.DENIED, None),
            (ApprovalDecision.IGNORED, ApprovalDecision.IGNORED, None),
            ("denied because", ApprovalDecision.DENIED, "denied because"),
            ((True, None), ApprovalDecision.APPROVED, None),
            ((False, "reason"), ApprovalDecision.DENIED, "reason"),
            ((ApprovalDecision.APPROVED, None), ApprovalDecision.APPROVED, None),
            ((False, None), ApprovalDecision.DENIED, None),
            ((None, None), ApprovalDecision.DENIED, None),
            ((123, None), ApprovalDecision.DENIED, "123"),
            ("nope", ApprovalDecision.DENIED, "nope"),
        ],
    )
    def test_coerce(self, input_, expected_decision, expected_reason):
        decision, reason = _coerce_to_approval_decision(input_)
        assert decision == expected_decision
        assert reason == expected_reason


# ------------------------------------------------------------------ #
# _merge_approval_status
# ------------------------------------------------------------------ #

class TestMergeApprovalStatus:
    """Merging hook results into a running approval status."""

    def test_ignored_does_not_change_pending(self):
        decision, reason, did_increment = _merge_approval_status(
            ApprovalDecision.PENDING, None
        )
        assert decision == ApprovalDecision.PENDING
        assert reason is None
        assert did_increment is True

    def test_approved_promotes_pending(self):
        decision, reason, did_increment = _merge_approval_status(
            ApprovalDecision.PENDING, True
        )
        assert decision == ApprovalDecision.APPROVED
        assert did_increment is True

    def test_denied_overrides_pending(self):
        decision, reason, did_increment = _merge_approval_status(
            ApprovalDecision.PENDING, "no"
        )
        assert decision == ApprovalDecision.DENIED
        assert reason == "no"
        assert did_increment is True

    def test_denied_overrides_approved(self):
        decision, reason, did_increment = _merge_approval_status(
            ApprovalDecision.APPROVED, False
        )
        assert decision == ApprovalDecision.DENIED
        assert did_increment is True

    def test_approved_does_not_override_denied(self):
        decision, reason, did_increment = _merge_approval_status(
            ApprovalDecision.DENIED, True
        )
        assert decision == ApprovalDecision.DENIED
        assert did_increment is True

    def test_pending_is_deferred(self):
        decision, reason, did_increment = _merge_approval_status(
            ApprovalDecision.PENDING, ApprovalDecision.PENDING
        )
        assert decision == ApprovalDecision.PENDING
        assert did_increment is False

    def test_pending_does_not_clear_existing_decision(self):
        decision, reason, did_increment = _merge_approval_status(
            ApprovalDecision.APPROVED, ApprovalDecision.PENDING
        )
        assert decision == ApprovalDecision.APPROVED
        assert did_increment is False


# ------------------------------------------------------------------ #
# _resolve_decision
# ------------------------------------------------------------------ #

class TestResolveDecision:
    """Resolving a list of decisions into a final decision."""

    def _resp(self, decision, reason=None, responded=True):
        return (decision, reason, responded)

    def test_all_pending_returns_pending(self):
        decisions = [self._resp(ApprovalDecision.PENDING) for _ in range(3)]
        decision, reason = _resolve_decision(decisions)
        assert decision == ApprovalDecision.PENDING
        assert reason is None

    def test_denied_wins_over_approved(self):
        decisions = [
            self._resp(ApprovalDecision.APPROVED),
            self._resp(ApprovalDecision.DENIED, "conflict"),
            self._resp(ApprovalDecision.APPROVED),
        ]
        decision, reason = _resolve_decision(decisions)
        assert decision == ApprovalDecision.DENIED
        assert reason == "conflict"

    def test_approved_wins_when_no_denied(self):
        decisions = [
            self._resp(ApprovalDecision.APPROVED),
            self._resp(ApprovalDecision.PENDING),
            self._resp(ApprovalDecision.APPROVED),
        ]
        decision, reason = _resolve_decision(decisions)
        assert decision == ApprovalDecision.APPROVED
        assert reason is None

    def test_multiple_denied_reasons_joined(self):
        decisions = [
            self._resp(ApprovalDecision.DENIED, "reason one"),
            self._resp(ApprovalDecision.DENIED, "reason two"),
        ]
        decision, reason = _resolve_decision(decisions)
        assert decision == ApprovalDecision.DENIED
        assert reason == "reason one\nreason two"

    def test_not_responded_ignored_in_resolution(self):
        decisions = [
            self._resp(ApprovalDecision.DENIED, "x", responded=False),
            self._resp(ApprovalDecision.APPROVED, responded=True),
        ]
        decision, reason = _resolve_decision(decisions)
        assert decision == ApprovalDecision.APPROVED

    def test_denied_without_reason(self):
        decisions = [self._resp(ApprovalDecision.DENIED)]
        decision, reason = _resolve_decision(decisions)
        assert decision == ApprovalDecision.DENIED
        assert reason is None


# ------------------------------------------------------------------ #
# ApprovalElection
# ------------------------------------------------------------------ #

class TestApprovalElection:
    """ApprovalElection coordinates policy + channel voting for a tool call."""

    @pytest.fixture
    def tool_call(self):
        return ContentPart.create_tool_use(
            call_id="call-42",
            name="test_tool",
            arguments='{"x": 1}',
        )

    @pytest.fixture
    def mock_runner(self):
        runner = MagicMock()
        runner._notify_channels = AsyncMock()
        runner.push_event = MagicMock()
        runner._execution_environment = MagicMock()
        runner._execution_environment._elections = {}
        return runner

    # --- Policy layer ---

    @pytest.mark.asyncio
    async def test_policy_approved_resolves_immediately(self, tool_call, mock_runner):
        """When a policy returns APPROVED, the election resolves without channel voting."""
        election = ApprovalElection(_content_part=tool_call)

        async def approve_policy(ctx):
            return ApprovalDecision.APPROVED

        election.register_policies([approve_policy])
        await election.start(mock_runner)

        assert election._decision == ApprovalDecision.APPROVED
        assert election._countdown_task is None
        mock_runner.push_event.assert_called_once()
        mock_runner._notify_channels.assert_called_once()

    @pytest.mark.asyncio
    async def test_policy_denied_resolves_immediately(self, tool_call, mock_runner):
        """When a policy returns DENIED, the election resolves without channel voting."""
        election = ApprovalElection(_content_part=tool_call)

        def deny_policy(ctx):
            return ApprovalDecision.DENIED

        election.register_policies([deny_policy])
        await election.start(mock_runner)

        assert election._decision == ApprovalDecision.DENIED

    @pytest.mark.asyncio
    async def test_all_ignored_continues_to_voter_phase(self, tool_call, mock_runner):
        """When all policies return IGNORED, the election escalates to voter phase."""
        election = ApprovalElection(_content_part=tool_call)

        def ignore_policy(ctx):
            return ApprovalDecision.IGNORED

        election.register_policies([ignore_policy])
        await election.start(mock_runner)

        assert election._decision is None
        assert election._countdown_task is None
        assert election._published is False

    @pytest.mark.asyncio
    async def test_mixed_policies_denied_wins(self, tool_call, mock_runner):
        """When one policy denies, resolution is DENIED regardless of other approvals."""
        election = ApprovalElection(_content_part=tool_call)

        def approve_policy(ctx):
            return ApprovalDecision.APPROVED

        def deny_policy(ctx):
            return ApprovalDecision.DENIED

        election.register_policies([approve_policy, deny_policy])
        await election.start(mock_runner)

        assert election._decision == ApprovalDecision.DENIED

    # --- Voter voting ---

    @pytest.mark.asyncio
    async def test_first_vote_starts_countdown(self, tool_call, mock_runner):
        """The first channel vote opens the voting window."""
        election = ApprovalElection(_content_part=tool_call)

        def ignore_policy(ctx):
            return ApprovalDecision.IGNORED

        election.register_policies([ignore_policy])
        await election.start(mock_runner)

        election.vote("channel-a", ApprovalDecision.APPROVED)

        assert "channel-a" in election._votes
        assert election._countdown_task is not None

    @pytest.mark.asyncio
    async def test_latest_vote_wins(self, tool_call, mock_runner):
        """When the same channel votes twice, the later vote wins."""
        election = ApprovalElection(_content_part=tool_call)

        election.vote("channel-a", ApprovalDecision.APPROVED)
        election.vote("channel-a", ApprovalDecision.DENIED, "changed my mind")

        decision, reason = election._votes["channel-a"]
        assert decision == ApprovalDecision.DENIED
        assert reason == "changed my mind"

    @pytest.mark.asyncio
    async def test_vote_after_decision_ignored(self, tool_call, mock_runner):
        """A vote arriving after a decision is already made is ignored."""
        election = ApprovalElection(_content_part=tool_call)
        election._decision = ApprovalDecision.APPROVED

        election.vote("channel-a", ApprovalDecision.DENIED)

        assert "channel-a" not in election._votes

    def test_ignored_vote_raises(self, tool_call):
        """A voter cannot cast IGNORED — it must actively decide."""
        election = ApprovalElection(_content_part=tool_call)

        with pytest.raises(ValueError, match="must decide"):
            election.vote("channel-a", ApprovalDecision.IGNORED)

    # --- Countdown resolution ---

    @pytest.mark.asyncio
    async def test_countdown_resolves_denied_over_approved(self, tool_call, mock_runner):
        """DENIED wins over APPROVED when countdown closes."""
        election = ApprovalElection(_content_part=tool_call)
        election._runner = mock_runner

        election.vote("chan-a", ApprovalDecision.APPROVED)
        election.vote("chan-b", ApprovalDecision.DENIED, "too risky")

        await election._resolve_from_voters()

        assert election._decision == ApprovalDecision.DENIED
        assert election._reason == "too risky"

    @pytest.mark.asyncio
    async def test_countdown_resolves_approved_when_no_denied(self, tool_call, mock_runner):
        """APPROVED wins if no DENIED when countdown closes."""
        election = ApprovalElection(_content_part=tool_call)
        election._runner = mock_runner

        # chan-a does not vote — that IS the abstention (IGNORED is not a valid vote).
        election.vote("chan-b", ApprovalDecision.APPROVED)

        await election._resolve_from_voters()

        assert election._decision == ApprovalDecision.APPROVED

    # --- Decision publishing ---

    @pytest.mark.asyncio
    async def test_publish_sends_approval_event_to_channels(self, tool_call, mock_runner):
        """publish_decision sends ApprovalEvent to all channels."""
        election = ApprovalElection(_content_part=tool_call)
        election._decision = ApprovalDecision.APPROVED
        election._reason = None
        election._runner = mock_runner

        await election.publish_decision()

        mock_runner._notify_channels.assert_called_once()
        call_args = mock_runner._notify_channels.call_args[0]
        event = call_args[0]
        assert event.tool_call_id == "call-42"
        assert event.approved is True
        assert event.denied_reason is None

    @pytest.mark.asyncio
    async def test_publish_sends_denied_event_to_channels(self, tool_call, mock_runner):
        """publish_decision sends a denied ApprovalEvent to all channels."""
        election = ApprovalElection(_content_part=tool_call)
        election._decision = ApprovalDecision.DENIED
        election._reason = "policy violation"
        election._runner = mock_runner

        await election.publish_decision()

        mock_runner._notify_channels.assert_called_once()
        call_args = mock_runner._notify_channels.call_args[0]
        event = call_args[0]
        assert event.tool_call_id == "call-42"
        assert event.approved is False
        assert event.denied_reason == "policy violation"

    @pytest.mark.asyncio
    async def test_publish_always_pushes_event(self, tool_call, mock_runner):
        """publish_decision always pushes to the runner event queue on resolution."""
        election = ApprovalElection(_content_part=tool_call)
        election._decision = ApprovalDecision.APPROVED
        election._runner = mock_runner

        await election.publish_decision()

        mock_runner.push_event.assert_called_once()
        mock_runner._notify_channels.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_does_not_push_event_twice(self, tool_call, mock_runner):
        """publish_decision guards against double-publish within the same resolution."""
        election = ApprovalElection(_content_part=tool_call)
        election._decision = ApprovalDecision.APPROVED
        election._runner = mock_runner

        await election.publish_decision()

        # The second call must raise — double-publish is a bug.
        with pytest.raises(RuntimeError, match="already published"):
            await election.publish_decision()
