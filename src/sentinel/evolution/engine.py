"""Evolution engine — NSGA-II spine + MAP-Elites archive, human-gated (design doc §3.5).

Loop (state machine §1.3c):
    STABLE --(graph shows module persistently bypassed/missing)--> PROPOSE
    PROPOSE -> EVALUATE (held-out probes + clean tasks) -> CANDIDATE | REJECT
    CANDIDATE -> HUMAN_GATE -> DEPLOY | ARCHIVE

The engine never deploys autonomously: it produces :class:`Proposal` objects that
must pass :class:`HumanGate`. Mutations are restricted to the closed defensive set.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

import numpy as np

from ..core.logging import get_logger
from ..core.types import Genome, MutationOp, Proposal, ThreatClass
from .fitness import FitnessVector
from .genome import mutate
from .human_gate import HumanGate
from .map_elites import MapElites
from .nsga2 import select_survivors

log = get_logger(__name__)

# evaluator: genome -> FitnessVector (runs the held-out adversarial + clean grids)
Evaluator = Callable[[Genome], FitnessVector]


class EvolutionEngine:
    def __init__(
        self,
        evaluator: Evaluator,
        human_gate: HumanGate,
        *,
        population_size: int = 12,
        offspring_per_gen: int = 8,
        bypass_trigger: int = 5,
        seed: int = 0,
    ) -> None:
        self.evaluate = evaluator
        self.gate = human_gate
        self.pop_size = population_size
        self.offspring = offspring_per_gen
        self.bypass_trigger = bypass_trigger
        self.rng = np.random.default_rng(seed)
        self.archive = MapElites()
        self.population: list[tuple[Genome, FitnessVector]] = []
        self.history: list[dict] = []  # per-generation telemetry for stability/convergence

    def should_evolve(self, bypass_counts: dict[str, int]) -> bool:
        """Trigger: any defense persistently bypassed (design doc §1.3c)."""
        return any(c >= self.bypass_trigger for c in bypass_counts.values())

    def initialize(self, seeds: list[Genome]) -> None:
        for g in seeds:
            f = self.evaluate(g)
            self.population.append((g, f))
            self.archive.add(g, f)

    def _propose(
        self, priors: dict[str, float], target_class: ThreatClass | None
    ) -> list[Genome]:
        children: list[Genome] = []
        ops = list(MutationOp)
        for _ in range(self.offspring):
            parent, _ = self.population[int(self.rng.integers(len(self.population)))]
            op = ops[int(self.rng.integers(len(ops)))]
            child = mutate(parent, op, self.rng, target_class=target_class, priors=priors)
            children.append(child)
        return children

    def step(
        self,
        priors: dict[str, float] | None = None,
        target_class: ThreatClass | None = None,
    ) -> list[Proposal]:
        """One generation: propose -> evaluate -> NSGA-II select -> human-gate the best.

        Returns the gated proposals (approved ones are deployed by the caller)."""
        priors = priors or {}
        children = self._propose(priors, target_class)
        child_fits = [self.evaluate(c) for c in children]
        for c, f in zip(children, child_fits, strict=True):
            self.archive.add(c, f)

        combined = self.population + list(zip(children, child_fits, strict=True))
        fits = [f for _, f in combined]
        survivors = select_survivors(fits, self.pop_size)
        self.population = [combined[i] for i in survivors]

        # telemetry
        best = min(self.population, key=lambda gf: gf[1].asr)
        churn = len(set(c.fingerprint() for c in children) - {g.fingerprint() for g, _ in self.population})
        self.history.append(
            {
                "best_asr": best[1].asr,
                "archive_coverage": self.archive.coverage(),
                "module_churn": churn,
                "pop_mean_asr": float(np.mean([f.asr for _, f in self.population])),
            }
        )

        # human gate the current best candidate(s)
        proposals: list[Proposal] = []
        bg, bf = best
        baseline_asr = self.history[0]["best_asr"] if self.history else bf.asr
        prop = Proposal(
            proposal_id=uuid.uuid4().hex[:10],
            genome=bg,
            mutation=MutationOp.TUNE_VALIDATOR,
            objectives={
                "asr": bf.asr,
                "recall": bf.recall,
                "precision": bf.precision,
                "utility_drop": bf.utility_drop,
                "delta_asr": bf.asr - baseline_asr,
                "constraint_violation": bf.constraint_violation(),
            },
        )
        gated = self.gate.review(prop)
        if gated.decision and gated.decision.approved:
            self.gate.deploy(gated)
        proposals.append(gated)
        return proposals

    def converged(self, window: int = 5, tol: float = 1e-3) -> bool:
        """Defense Convergence Rate / Stability Index: best-ASR stable over a window."""
        if len(self.history) < window:
            return False
        recent = [h["best_asr"] for h in self.history[-window:]]
        return float(np.std(recent)) < tol
