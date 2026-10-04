"""Channel-driven content-part approval voting mechanism."""

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peteos.engine.runner import Runner

from peteos.chatbot import ContentPart
from peteos.engine.events import ApprovalEvent


_logger = logging.getLogger(__name__)

CHANNEL_APPROVAL_WINDOW = 2.0

class ApprovalDecision(str, Enum):
    """Possible decisions a hook can return from on_tool_call."""
    APPROVED = "approved"
    DENIED = "denied"
    PENDING = "pending"
    IGNORED = "ignored"


_APPROVAL_SEVERITY: dict[ApprovalDecision | None, int] = {
    None: -1,
    ApprovalDecision.PENDING: 0,
    ApprovalDecision.IGNORED: 1,
    ApprovalDecision.APPROVED: 2,
    ApprovalDecision.DENIED: 3,
}

_SEVERITY_TO_DECISION: dict[int, ApprovalDecision] = {v: k for k, v in _APPROVAL_SEVERITY.items() if k is not None}


@dataclass
class PolicyRespondHandle:
    """Injection handle for async tool call approval.

    Returned in the on_tool_call hook context. Captures the runner
    internally so the caller can approve or deny without a direct
    runner reference.
    """
    _election: "ApprovalElection"
    _policy_index: int

    async def approve(self) -> None:
        """Respond to this hook slot with approval."""
        # Hand in the policie's vote
        await self._election.policy_vote(self._policy_index, ApprovalDecision.APPROVED)

    async def deny(self, reason: str | None = None) -> None:
        """Respond to this hook slot with denial."""
        # Hand in the policie's vote
        await self._election.policy_vote(self._policy_index, ApprovalDecision.DENIED)


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
    current: ApprovalDecision,
    hook_result: Any,
) -> tuple[ApprovalDecision, str | None, bool]:
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
        if current == ApprovalDecision.PENDING:
            return ApprovalDecision.APPROVED, None, True
        return current, None, True

    # If DENIED, return DENIED with reason and increment.
    if decision == ApprovalDecision.DENIED:
        return ApprovalDecision.DENIED, denied_reason, True

    # If PENDING, keep current and do NOT increment (defer — hook will respond later).
    if decision == ApprovalDecision.PENDING:
        return current, None, False

    # Fallback: return current with increment (defensive, should be unreachable).
    return current, None, True


def _resolve_decision(
    decisions: list[tuple[ApprovalDecision, str | None, bool]],
) -> tuple[ApprovalDecision, str | None]:
    """Merge all decisions — max severity wins; IGNORED and PENDING are skipped."""
    # Collect severities and denial reasons from all deciding respondents.
    severities: list[int] = []
    reasons: list[str] = []
    for status, reason, responded in decisions:
        # Skip absent and PENDING respondents — they contribute no decision.
        if not responded or status is ApprovalDecision.PENDING:
            continue
        severities.append(_APPROVAL_SEVERITY[status])
        if status == ApprovalDecision.DENIED and reason:
            reasons.append(reason)

    # If no deciding respondent contributed, escalate with PENDING.
    if not severities:
        return ApprovalDecision.PENDING, None

    # The winner is the highest-severity decision among all deciding respondents.
    # IGNORED has severity 1, APPROVED 2, DENIED 3 — so IGNORED can still win over
    # PENDING while DENIED always outranks both.
    winner = _SEVERITY_TO_DECISION[max(severities)]
    denied_reason = "\n".join(reasons) if reasons else None
    return winner, denied_reason


@dataclass
class ApprovalElection:
    """Coordinates content-part approval across policies and voters.

    Mirrors the deferred-respond pattern from ExecutionEnvironment.
    Each policy gets a decision slot. If all return IGNORED and no voter
    has responded yet, the voting window opens for voter contributions.
    """
    _content_part: ContentPart
    _runner: "Runner" = field(default=None, repr=False)

    _policies: list[object] = field(default_factory=list)
    _policy_responses = 0
    _policy_decisions: list[tuple[ApprovalDecision, str | None, bool]] = field(default_factory=list)

    _decision: ApprovalDecision | None = field(default=None)
    _reason: str | None = None

    _votes: dict[str, tuple[ApprovalDecision, str | None]] = field(default_factory=dict)
    _countdown_task: asyncio.Task | None = field(default=None)
    _published: bool = field(default=False)

    def register_policies(self, policies: list[object]) -> None:
        """Register the policy layer to consult before escalating to voters."""
        self._policies = list(policies)

    async def start(self, runner: "Runner") -> None:
        """Run the approval algorithm and publish the decision as soon as one is resolved."""
        # Bind the runner for async respond handles.
        self._runner = runner

        # Let policies vote. Escalate only if the election was ignored by all policies.
        policy_result = await self._call_policies()
        if policy_result is not ApprovalDecision.IGNORED:
            return

        # If we already have a voter upon escalation, start the countdown.
        # Otherwise the first voter will start the countdown.
        if self._votes and not self._countdown_task:
            self._start_countdown()

    async def _call_policies(self) -> ApprovalDecision:
        """Call each hook and fill its decision slot in the record."""
        # Initialize the policie's answer slots.
        self._policy_decisions = [(ApprovalDecision.IGNORED, None, False)] * len(self._policies)

        for i, policy in enumerate(self._policies):
            # build respond handle so hooks can defer their decision to a later turn
            respond = PolicyRespondHandle(self, i)

            # sync or async: await if hook returned a coroutine
            ctx = {
                "content": self._content_part,
                "respond": respond,
            }
            result = policy(ctx)
            if inspect.isawaitable(result):
                result = await result

            # Coerce the policy vote and record it for resolution.
            decision, denied_reason = _coerce_to_approval_decision(result)
            await self.policy_vote(i, decision, denied_reason, resolve=False)

        return await self._resolve_from_policies()

    async def policy_vote(
        self,
        policy_index: int,
        decision: ApprovalDecision,
        reason: str | None = None,
        resolve: bool = True,
    ) -> ApprovalDecision:
        # Ignore this vote if it is (still) deferred.
        decided = decision is not ApprovalDecision.PENDING
        if not decided:
            return ApprovalDecision.PENDING

        # Ensure that each policy has only one vote.
        d, _, responded = self._policy_decisions[policy_index]
        if responded:
            raise RuntimeError(
                f"Hook at index {policy_index} already responded with {d.value}"
            )

        # record the vote; only increment count for votes that actually voted
        self._policy_decisions[policy_index] = (decision, reason, True)
        self._policy_responses += 1

        # Try to resolve the decision only when explicitly requested.
        if not resolve:
            return ApprovalDecision.PENDING
        return await self._resolve_from_policies()

    async def _resolve_from_policies(self) -> ApprovalDecision:
        # Return if some policies did not respond yet.
        if self._policy_responses < len(self._policy_decisions):
            return ApprovalDecision.PENDING

        # Otherwise, resolve the decision.
        decision, merged_reason = _resolve_decision(self._policy_decisions)
        
        # If we got here, there must be no pending policies left
        if decision is ApprovalDecision.PENDING:
            raise RuntimeError("Cannot publish a PENDING decision")

        # Publish the decision, if at least one policy did not ignore it.
        if decision is not ApprovalDecision.IGNORED:
            self._decision = decision
            self._reason = merged_reason
            await self.publish_decision()

        # Provide information about the resolution so far.
        return decision

    def vote(self, voter_id: str, decision: ApprovalDecision, reason: str | None = None) -> None:
        """Record a voter's explicit decision."""
        # Reject implicit abstention — voters must decide.
        if decision is ApprovalDecision.IGNORED:
            raise ValueError("A voter must decide: APPROVED or DENIED.")

        # If a decision has already been made, ignore this vote.
        if self._decision:
            return

        # Capture first-arrival before mutating the ledger.
        first_voter = not self._votes

        # Record or overwrite this voter's decision.
        self._votes[voter_id] = (decision, reason)

        # Open the voting window on the first arrival — timer starts only once.
        if first_voter and self._decision is None:
            self._start_countdown()

    async def _resolve_from_voters(self) -> tuple[ApprovalDecision, str | None]:
        """Evaluate collected voter votes. DENIED wins over APPROVED."""
        # The first voter starts the countdown. Therefore, at least one voter is present
        assert self._votes

        # Resolve decision from votes.
        voter_decisions = [
            (v[0], v[1], True)
            for v in self._votes.values()
        ]
        self._decision, self._reason = _resolve_decision(voter_decisions)

        return self._decision, self._reason

    def _start_countdown(self) -> None:
        """Start the voting window. Called once on the first voter vote."""
        self._countdown_task = asyncio.ensure_future(self._countdown_waiter())

    async def _countdown_waiter(self) -> None:
        """Await the window duration then resolve from voters if not already resolved."""
        # Give each voter a chance to vote within a time frame.
        await asyncio.sleep(CHANNEL_APPROVAL_WINDOW)
        
        # Resolve the decision. Since the timer started, at least one vote is available.
        await self._resolve_from_voters()

        # Publish decision
        await self.publish_decision()

    async def publish_decision(self) -> None:
        # Guard against double-publish — raises if this election was already published.
        if self._published:
            raise RuntimeError("Decision already published")
        self._published = True

        # Remove this election from the execution environment so it cannot accept
        # further votes.
        self._runner._execution_environment._elections.pop(self._content_part.call_id, None)

        # Construct the approval event.
        approval = ApprovalEvent(
            tool_call_id=self._content_part.call_id,
            tool_call=self._content_part,
            approved=self._decision is ApprovalDecision.APPROVED,
            denied_reason=self._reason
        )

        # Push the approval event into the runner's queue.
        self._runner.push_event(approval)

        # Notify all channels about the approval event.
        await self._runner._notify_channels(approval)
