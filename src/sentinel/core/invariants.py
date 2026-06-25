"""Runtime enforcement of the five non-negotiable safety invariants.

These functions are called at the boundaries where a violation could occur. They
raise :class:`InvariantViolation` (a hard error) rather than logging a warning,
so any breach fails the run and the CI suite. See ``docs/01-architecture-and-design.md`` §0.
"""

from __future__ import annotations

from .types import GateState, Genome, ModuleKind, MutationOp, Proposal

# Module kinds the evolution engine is permitted to create/modify. This is the
# whitelist behind Invariant I2/I4 — anything outside it is rejected.
ALLOWED_EVOLUTION_KINDS: frozenset[ModuleKind] = frozenset(
    {
        ModuleKind.INPUT_VALIDATOR,
        ModuleKind.THREAT_DETECTOR,
        ModuleKind.OUTPUT_GUARD,
        ModuleKind.PRIVILEGE_CHECK,
        ModuleKind.POLICY_CONSTRAINT,
        ModuleKind.ROUTER,
    }
)

ALLOWED_MUTATIONS: frozenset[MutationOp] = frozenset(MutationOp)


class InvariantViolation(RuntimeError):
    """Raised when a safety invariant would be breached."""


def assert_defensive_genome(genome: Genome) -> None:
    """I2/I4: a genome may contain only whitelisted defensive module kinds."""
    for m in genome.modules:
        if m.kind not in ALLOWED_EVOLUTION_KINDS:
            raise InvariantViolation(
                f"genome {genome.genome_id} contains non-defensive module kind {m.kind!r}"
            )


def assert_allowed_mutation(op: MutationOp) -> None:
    """I2: only the closed set of defensive mutations is permitted."""
    if op not in ALLOWED_MUTATIONS:
        raise InvariantViolation(f"mutation {op!r} is not in the allowed defensive set")


def assert_human_gated_deploy(proposal: Proposal) -> None:
    """I3: a proposal may reach DEPLOYED only via an APPROVED human decision.

    Called immediately before any deployment. Guarantees there is no autonomous
    PENDING -> DEPLOYED transition.
    """
    if proposal.state is GateState.DEPLOYED:
        if proposal.decision is None or not proposal.decision.approved:
            raise InvariantViolation(
                f"proposal {proposal.proposal_id} deployed without an approved human gate decision"
            )
    if proposal.state is GateState.PENDING and proposal.decision is not None:
        raise InvariantViolation(
            f"proposal {proposal.proposal_id} has a decision but is still PENDING"
        )


def assert_no_attack_synthesis(component_name: str, emits_attacks: bool) -> None:
    """I1: no component may emit attacks. Components self-declare; this is a tripwire."""
    if emits_attacks:
        raise InvariantViolation(
            f"component {component_name!r} declared attack-emitting output — forbidden (I1)"
        )
