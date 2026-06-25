"""Genome construction and the closed set of defensive mutation operators.

Every mutation is validated against the safety invariants *before* it is returned,
so an illegal (non-defensive) genome can never be produced (Invariants I2/I4).
"""

from __future__ import annotations

import uuid

import numpy as np

from ..core.invariants import assert_allowed_mutation, assert_defensive_genome
from ..core.types import (
    DefenseStrategy,
    DefensiveModuleSpec,
    Genome,
    ModuleKind,
    MutationOp,
    ThreatClass,
)

# Map each defensive strategy to the module kind that implements it.
_STRATEGY_KIND: dict[DefenseStrategy, ModuleKind] = {
    DefenseStrategy.INPUT_SANITIZATION: ModuleKind.INPUT_VALIDATOR,
    DefenseStrategy.INSTRUCTION_DATA_SEPARATION: ModuleKind.INPUT_VALIDATOR,
    DefenseStrategy.PRIVILEGE_NARROWING: ModuleKind.PRIVILEGE_CHECK,
    DefenseStrategy.RETRIEVAL_GROUNDING: ModuleKind.INPUT_VALIDATOR,
    DefenseStrategy.MEMORY_QUARANTINE: ModuleKind.POLICY_CONSTRAINT,
    DefenseStrategy.OUTPUT_VALIDATION: ModuleKind.OUTPUT_GUARD,
}


def random_seed_genome(rng: np.random.Generator, n: int = 1) -> Genome:
    all_strats = list(DefenseStrategy)
    idx = rng.choice(len(all_strats), size=min(n, len(all_strats)), replace=False)
    strategies = [all_strats[int(i)] for i in idx]
    modules = [
        DefensiveModuleSpec(
            kind=_STRATEGY_KIND[s], name=s.value,
            params={"threshold": float(rng.uniform(0.3, 0.7))}, targets=[],
        )
        for s in strategies
    ]
    g = Genome(genome_id=uuid.uuid4().hex[:10], modules=modules, generation=0)
    assert_defensive_genome(g)
    return g


def empty_genome() -> Genome:
    return Genome(genome_id=uuid.uuid4().hex[:10], modules=[], generation=0)


def _new_module(strategy: DefenseStrategy, threshold: float, targets: list[ThreatClass]) -> DefensiveModuleSpec:
    return DefensiveModuleSpec(
        kind=_STRATEGY_KIND[strategy], name=strategy.value,
        params={"threshold": threshold}, targets=targets,
    )


def mutate(
    genome: Genome,
    op: MutationOp,
    rng: np.random.Generator,
    *,
    target_class: ThreatClass | None = None,
    priors: dict[str, float] | None = None,
) -> Genome:
    """Apply one allowed mutation, returning a new genome. Validated for safety."""
    assert_allowed_mutation(op)
    mods = [m.model_copy(deep=True) for m in genome.modules]
    present = {m.name for m in mods}
    priors = priors or {}

    def pick_strategy() -> DefenseStrategy:
        candidates = [s for s in DefenseStrategy if s.value not in present]
        if not candidates:
            candidates = list(DefenseStrategy)
        # bias by bandit-derived priors (higher posterior -> more likely)
        weights = np.array([priors.get(s.value, 1.0) + 1e-3 for s in candidates])
        return candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]

    tgts = [target_class] if target_class else []

    if op in (MutationOp.ADD_VALIDATOR, MutationOp.ADD_DETECTOR, MutationOp.ADD_POLICY_CONSTRAINT):
        s = pick_strategy()
        mods.append(_new_module(s, float(rng.uniform(0.3, 0.7)), tgts))
    elif op in (MutationOp.TUNE_VALIDATOR, MutationOp.TUNE_DETECTOR):
        if mods:
            m = mods[int(rng.integers(len(mods)))]
            cur = m.params.get("threshold", 0.5)
            m.params["threshold"] = float(np.clip(cur + rng.normal(0, 0.1), 0.05, 0.95))
    elif op is MutationOp.REROUTE:
        if len(mods) >= 2:
            i, j = rng.choice(len(mods), size=2, replace=False)
            mods[i], mods[j] = mods[j], mods[i]

    child = Genome(
        genome_id=uuid.uuid4().hex[:10], modules=mods,
        parent_id=genome.genome_id, generation=genome.generation + 1,
    )
    assert_defensive_genome(child)  # I2/I4 tripwire
    return child
