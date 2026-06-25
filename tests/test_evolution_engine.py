"""Evolution engine tests with a pure-function evaluator (dependency injection, not a
model mock). Verifies: human-gated deployment, convergence telemetry, archive growth."""

from __future__ import annotations

import numpy as np

from sentinel.core.types import GateState
from sentinel.evolution.engine import EvolutionEngine
from sentinel.evolution.fitness import FitnessVector
from sentinel.evolution.genome import random_seed_genome
from sentinel.evolution.human_gate import PolicyHumanGate


def _evaluator(genome) -> FitnessVector:
    # reward more modules with lower ASR (deterministic, bounded), small utility cost
    n = len(genome.modules)
    asr = max(0.05, 0.7 - 0.08 * n)
    return FitnessVector(
        asr=asr, recall=min(0.99, 0.6 + 0.05 * n), precision=0.9,
        fpr=0.02, utility_drop=min(0.04, 0.005 * n), latency_s=1.0, token_cost=200.0,
    )


def test_evolution_is_human_gated_and_improves():
    gate = PolicyHumanGate()
    engine = EvolutionEngine(_evaluator, gate, population_size=8, offspring_per_gen=6)
    rng = np.random.default_rng(0)
    engine.initialize([random_seed_genome(rng, n=k) for k in (1, 1, 2, 2)])
    first_best = min(p[1].asr for p in engine.population)
    for _ in range(10):
        proposals = engine.step()
        for p in proposals:
            # any deployed proposal MUST carry an approved decision (Invariant I3)
            if p.state is GateState.DEPLOYED:
                assert p.decision is not None and p.decision.approved
    last_best = min(p[1].asr for p in engine.population)
    assert last_best <= first_best  # evolution did not regress
    assert engine.archive.coverage() > 0.0
    assert len(gate.log) >= 10  # every generation produced a logged gate decision


def test_convergence_flag():
    gate = PolicyHumanGate()
    engine = EvolutionEngine(_evaluator, gate, population_size=6, offspring_per_gen=4)
    rng = np.random.default_rng(1)
    engine.initialize([random_seed_genome(rng, n=4) for _ in range(4)])
    for _ in range(12):
        engine.step()
    # with a deterministic evaluator the best ASR stabilizes -> converged
    assert engine.converged()
