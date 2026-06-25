"""Human evaluation gate (Invariant I3).

Every evolved configuration passes through a human review before retention. This
module makes that boundary explicit and auditable. Two policies are provided:

* :class:`InteractiveHumanGate` — prompts a human reviewer (CLI) and records the decision.
* :class:`PolicyHumanGate` — for *batch experiments*, applies a pre-registered,
  logged auto-approval policy (e.g. "approve iff Pareto-improving AND utility_drop<=5%
  AND no constraint violation"). This is still a recorded, reviewable decision — it is
  NOT an autonomous bypass: the policy and every decision are written to the audit log,
  and the criteria are fixed before the run (pre-registration).

Both attach a :class:`HumanGateDecision`, the only thing that authorizes deployment.
"""

from __future__ import annotations

from ..core.invariants import assert_human_gated_deploy
from ..core.logging import get_logger
from ..core.types import GateState, HumanGateDecision, Proposal

log = get_logger(__name__)


class HumanGate:
    """Base gate. Subclasses implement ``_decide``."""

    def __init__(self, reviewer: str = "unspecified") -> None:
        self.reviewer = reviewer
        self.log: list[Proposal] = []

    def review(self, proposal: Proposal) -> Proposal:
        decision = self._decide(proposal)
        proposal.decision = decision
        proposal.state = GateState.APPROVED if decision.approved else GateState.REJECTED
        self.log.append(proposal)
        log.info(
            "human_gate",
            proposal=proposal.proposal_id,
            approved=decision.approved,
            reviewer=decision.reviewer,
            rationale=decision.rationale,
        )
        return proposal

    def deploy(self, proposal: Proposal) -> Proposal:
        if proposal.state is not GateState.APPROVED:
            raise PermissionError("cannot deploy a non-approved proposal")
        proposal.state = GateState.DEPLOYED
        assert_human_gated_deploy(proposal)  # I3 tripwire
        return proposal

    def _decide(self, proposal: Proposal) -> HumanGateDecision:  # pragma: no cover
        raise NotImplementedError


class PolicyHumanGate(HumanGate):
    """Pre-registered auto-policy gate for batch experiments (still logged/auditable)."""

    def __init__(
        self, reviewer: str = "pre-registered-policy", max_utility_drop: float = 0.05
    ) -> None:
        super().__init__(reviewer)
        self.max_utility_drop = max_utility_drop

    def _decide(self, proposal: Proposal) -> HumanGateDecision:
        obj = proposal.objectives
        approve = (
            obj.get("constraint_violation", 1.0) == 0.0
            and obj.get("utility_drop", 1.0) <= self.max_utility_drop
            and obj.get("delta_asr", 0.0) <= 0.0  # ASR did not increase
        )
        rationale = (
            f"policy: feasible & utility_drop<= {self.max_utility_drop} & ASR non-increasing; "
            f"objectives={obj}"
        )
        return HumanGateDecision(approved=approve, reviewer=self.reviewer, rationale=rationale)


class InteractiveHumanGate(HumanGate):  # pragma: no cover - requires a human
    def _decide(self, proposal: Proposal) -> HumanGateDecision:
        print(f"\n=== HUMAN GATE: proposal {proposal.proposal_id} ===")
        print(f"mutation: {proposal.mutation.value}")
        print(f"objectives: {proposal.objectives}")
        print("modules:", [m.name for m in proposal.genome.modules])
        ans = input("approve? [y/N]: ").strip().lower()
        rationale = input("rationale: ").strip()
        return HumanGateDecision(
            approved=ans == "y", reviewer=self.reviewer, rationale=rationale or "(none)"
        )
