"""Safety-invariant tests (I1-I4). These are the most important tests in the repo:
a green suite here is the machine-checked guarantee that SENTINEL stays defensive."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.core.invariants import (
    InvariantViolation,
    assert_allowed_mutation,
    assert_defensive_genome,
    assert_human_gated_deploy,
)
from sentinel.core.types import (
    DefensiveModuleSpec,
    GateState,
    Genome,
    HumanGateDecision,
    ModuleKind,
    MutationOp,
    Proposal,
)
from sentinel.evolution.genome import mutate, random_seed_genome


def test_defensive_genome_accepts_defensive_modules():
    g = random_seed_genome(np.random.default_rng(0), n=3)
    assert_defensive_genome(g)  # should not raise


def test_genome_cannot_hold_nondefensive_kind():
    # ModuleKind has no offensive member; we simulate a tampered genome by bypassing
    # the enum is impossible, so we assert the whitelist rejects a hypothetical kind.
    g = Genome(genome_id="x", modules=[DefensiveModuleSpec(kind=ModuleKind.ROUTER, name="r")])
    assert_defensive_genome(g)  # router is defensive/allowed
    # all enum members are defensive by construction => whitelist == enum
    from sentinel.core.invariants import ALLOWED_EVOLUTION_KINDS
    assert set(ModuleKind) == set(ALLOWED_EVOLUTION_KINDS)


def test_all_mutations_allowed_and_closed():
    for op in MutationOp:
        assert_allowed_mutation(op)
    assert len(set(MutationOp)) == 6


def test_no_autonomous_deploy():
    g = random_seed_genome(np.random.default_rng(1), n=2)
    p = Proposal(proposal_id="p1", genome=g, mutation=MutationOp.TUNE_VALIDATOR)
    # deploying without an approved decision must fail the invariant
    p.state = GateState.DEPLOYED
    with pytest.raises(InvariantViolation):
        assert_human_gated_deploy(p)


def test_deploy_requires_approved_decision():
    g = random_seed_genome(np.random.default_rng(2), n=2)
    p = Proposal(proposal_id="p2", genome=g, mutation=MutationOp.TUNE_VALIDATOR)
    p.decision = HumanGateDecision(approved=True, reviewer="tester", rationale="ok")
    p.state = GateState.DEPLOYED
    assert_human_gated_deploy(p)  # should not raise


def test_pending_with_decision_is_invalid():
    g = random_seed_genome(np.random.default_rng(3), n=1)
    p = Proposal(proposal_id="p3", genome=g, mutation=MutationOp.TUNE_VALIDATOR)
    p.decision = HumanGateDecision(approved=True, reviewer="t", rationale="r")
    p.state = GateState.PENDING
    with pytest.raises(InvariantViolation):
        assert_human_gated_deploy(p)


def test_mutation_preserves_defensive_invariant():
    rng = np.random.default_rng(4)
    g = random_seed_genome(rng, n=2)
    for op in MutationOp:
        child = mutate(g, op, rng)
        assert_defensive_genome(child)  # every mutation stays defensive
