"""Meta-defense layer — contextual multi-armed bandit (design doc §3.4, G5).

Problem: given a threat's behavioral signature (context x), choose the defensive
strategy (arm) most likely to neutralize it without harming utility. This is an
online contextual bandit:

* **Arms** = the six DefenseStrategy values.
* **Context** = the behavioral signature vector x ∈ R^d.
* **Reward** = 1 if the attack is neutralized AND clean-task utility preserved, else 0
  (with a shaped penalty for utility loss).

We implement **LinUCB** (Li et al., WWW 2010) — disjoint linear models per arm with an
upper-confidence bound that provably balances exploration/exploitation with
sublinear regret. We also expose **Thompson sampling** as an alternative posterior
policy. No hardcoded threat->defense table (paper requirement).

Why a bandit, not full RL: the action has an immediate, observable reward and no
long-horizon credit assignment, so a contextual bandit is the right (sample-efficient,
stable) tool. Posteriors are exported as priors to the evolution engine.
"""

from __future__ import annotations

import numpy as np

from ..core.types import DefenseStrategy

_ARMS: list[DefenseStrategy] = list(DefenseStrategy)


class _LinUCBArm:
    def __init__(self, dim: int, ridge: float = 1.0) -> None:
        self.A = np.eye(dim) * ridge      # d x d
        self.b = np.zeros(dim)            # d
        self.n = 0

    def theta(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def ucb(self, x: np.ndarray, alpha: float) -> float:
        A_inv = np.linalg.inv(self.A)
        mean = float(self.theta() @ x)
        bonus = alpha * float(np.sqrt(x @ A_inv @ x))
        return mean + bonus

    def sample(self, x: np.ndarray, rng: np.random.Generator) -> float:
        """Thompson sample of expected reward (Bayesian linear regression posterior)."""
        A_inv = np.linalg.inv(self.A)
        theta_tilde = rng.multivariate_normal(self.theta(), A_inv)
        return float(theta_tilde @ x)

    def update(self, x: np.ndarray, reward: float) -> None:
        self.A += np.outer(x, x)
        self.b += reward * x
        self.n += 1


class MetaDefenseSelector:
    def __init__(
        self,
        context_dim: int,
        *,
        policy: str = "linucb",
        alpha: float = 1.0,
        max_strategies: int = 2,
        seed: int = 0,
    ) -> None:
        self.dim = context_dim
        self.policy = policy
        self.alpha = alpha
        self.max_strategies = max_strategies
        self._arms = {a: _LinUCBArm(context_dim) for a in _ARMS}
        self._rng = np.random.default_rng(seed)

    def _context(self, signature_vector: np.ndarray) -> np.ndarray:
        x = np.asarray(signature_vector, dtype=np.float64)
        if x.shape[0] < self.dim:
            x = np.concatenate([x, np.zeros(self.dim - x.shape[0])])
        return x[: self.dim]

    def select(self, signature_vector: np.ndarray) -> list[DefenseStrategy]:
        """Pick the top-k strategies by UCB / Thompson score for this context."""
        x = self._context(signature_vector)
        if self.policy == "thompson":
            scores = {a: arm.sample(x, self._rng) for a, arm in self._arms.items()}
        else:
            scores = {a: arm.ucb(x, self.alpha) for a, arm in self._arms.items()}
        ranked = sorted(scores, key=lambda a: scores[a], reverse=True)
        return ranked[: self.max_strategies]

    def update(
        self, signature_vector: np.ndarray, chosen: list[DefenseStrategy], reward: float
    ) -> None:
        x = self._context(signature_vector)
        for a in chosen:
            self._arms[a].update(x, reward)

    def posteriors(self) -> dict[str, float]:
        """Mean expected reward per arm at the origin-context — used as evolution priors."""
        return {a.value: float(np.linalg.norm(arm.theta())) for a, arm in self._arms.items()}
