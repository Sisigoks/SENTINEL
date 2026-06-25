"""Security fitness — constrained multi-objective (design doc §3.6, G7).

Selection uses **Pareto dominance** over the objective vector
    f = (1 - ASR, recall, precision)            [maximize all]
subject to hard constraints
    FPR <= eps_fp, clean_utility_drop <= 0.05, latency <= L_max, cost <= C_max.

A candidate violating any constraint is dominated by any feasible candidate
(constraint-domination, Deb 2002). For *reporting* we also compute the paper's
scalar SecurityFitness = (1 - ASR) / (cost * latency) and DefenseEfficiency.

Scalarizing for *selection* is rejected because the weight choice would itself
become an attack on the result's validity; constraints encode the hard utility gate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FitnessVector:
    asr: float
    recall: float
    precision: float
    fpr: float
    utility_drop: float
    latency_s: float
    token_cost: float

    # objectives to MAXIMIZE
    def objectives(self) -> tuple[float, float, float]:
        return (1.0 - self.asr, self.recall, self.precision)

    def feasible(
        self, eps_fp: float = 0.10, max_util_drop: float = 0.05,
        max_latency: float = 30.0, max_cost: float = 20000.0,
    ) -> bool:
        return (
            self.fpr <= eps_fp
            and self.utility_drop <= max_util_drop
            and self.latency_s <= max_latency
            and self.token_cost <= max_cost
        )

    def constraint_violation(
        self, eps_fp: float = 0.10, max_util_drop: float = 0.05,
        max_latency: float = 30.0, max_cost: float = 20000.0,
    ) -> float:
        return (
            max(0.0, self.fpr - eps_fp)
            + max(0.0, self.utility_drop - max_util_drop)
            + max(0.0, self.latency_s - max_latency) / max_latency
            + max(0.0, self.token_cost - max_cost) / max_cost
        )


def dominates(a: FitnessVector, b: FitnessVector) -> bool:
    """Constraint-domination: feasible beats infeasible; among feasible, Pareto;
    among infeasible, lower total violation."""
    av, bv = a.constraint_violation(), b.constraint_violation()
    if av == 0.0 and bv > 0.0:
        return True
    if av > 0.0 and bv == 0.0:
        return False
    if av > 0.0 and bv > 0.0:
        return av < bv
    # both feasible -> Pareto dominance over maximization objectives
    ao, bo = a.objectives(), b.objectives()
    no_worse = all(x >= y for x, y in zip(ao, bo, strict=True))
    strictly_better = any(x > y for x, y in zip(ao, bo, strict=True))
    return no_worse and strictly_better


class SecurityFitness:
    """Scalar reporting metrics (not used for selection)."""

    @staticmethod
    def scalar(f: FitnessVector) -> float:
        # (1 - ASR) / (cost * latency); guard divide-by-zero, scale cost to ~k-tokens
        denom = max(f.token_cost / 1000.0, 1e-3) * max(f.latency_s, 1e-3)
        return (1.0 - f.asr) / denom

    @staticmethod
    def defense_efficiency(delta_security: float, delta_complexity: int) -> float:
        return delta_security / max(delta_complexity, 1)
