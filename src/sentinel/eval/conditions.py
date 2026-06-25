"""The five defense conditions (paper Table 8).

Each adds exactly one mechanism so pairwise differences attribute protection to a
component:

    Vanilla            no defense                     (lower bound, max ASR)
    StaticFilter       fixed keyword/rule guard       (value of static defense)
    ReflectionDefense  self-critique for safety only  (value of reflective checking)
    MetaDefense        learns threat -> countermeasure(value of defense meta-learning)
    FullSENTINEL       meta-defense + evolution        (value of defensive evolution)

A condition maps a screened :class:`ThreatEvent` to a :class:`DefensePlan`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.types import DefensePlan, DefenseStrategy, ThreatEvent
from ..meta_defense.bandit import MetaDefenseSelector


class DefenseCondition(ABC):
    name: str

    @abstractmethod
    def plan(self, event: ThreatEvent) -> DefensePlan: ...

    def update(self, event: ThreatEvent, plan: DefensePlan, reward: float) -> None:
        """Online conditions (meta-defense) learn here; others no-op."""


class Vanilla(DefenseCondition):
    name = "vanilla"

    def plan(self, event: ThreatEvent) -> DefensePlan:
        return DefensePlan(strategies=[], rationale="no defense", selected_by="vanilla")


class StaticFilter(DefenseCondition):
    name = "static_filter"
    # fixed, non-learning guard: always sanitize + separate instructions from data
    _FIXED = [DefenseStrategy.INPUT_SANITIZATION, DefenseStrategy.INSTRUCTION_DATA_SEPARATION]

    def plan(self, event: ThreatEvent) -> DefensePlan:
        strategies = list(self._FIXED) if event.is_threat else []
        return DefensePlan(strategies=strategies, rationale="fixed guard", selected_by="static")


class ReflectionDefense(DefenseCondition):
    """Self-critique for safety only: applies output validation + sanitization when the
    cascade flags a threat, emulating a reflexion-style safety pass (no learning)."""

    name = "reflection_defense"

    def plan(self, event: ThreatEvent) -> DefensePlan:
        if not event.is_threat:
            return DefensePlan(strategies=[], selected_by="reflection")
        return DefensePlan(
            strategies=[DefenseStrategy.OUTPUT_VALIDATION, DefenseStrategy.INPUT_SANITIZATION],
            rationale="self-critique safety pass",
            selected_by="reflection",
        )


class MetaDefense(DefenseCondition):
    """Contextual-bandit countermeasure learning (no architecture evolution)."""

    name = "meta_defense"

    def __init__(self, context_dim: int, policy: str = "linucb", seed: int = 0) -> None:
        self.selector = MetaDefenseSelector(context_dim, policy=policy, seed=seed)

    def plan(self, event: ThreatEvent) -> DefensePlan:
        if not event.is_threat or event.signature is None:
            return DefensePlan(strategies=[], selected_by="meta_defense")
        strategies = self.selector.select(event.signature.vector())
        return DefensePlan(strategies=strategies, rationale="bandit selection", selected_by="meta_defense")

    def update(self, event: ThreatEvent, plan: DefensePlan, reward: float) -> None:
        if event.signature is not None and plan.strategies:
            self.selector.update(event.signature.vector(), plan.strategies, reward)


class FullSentinel(MetaDefense):
    """Meta-defense + defensive evolution. The evolved genome biases strategy selection;
    the evolution loop is driven externally by :class:`EvolutionEngine` (human-gated)."""

    name = "full_sentinel"

    def __init__(self, context_dim: int, policy: str = "linucb", seed: int = 0) -> None:
        super().__init__(context_dim, policy=policy, seed=seed)
        self.deployed_strategies: set[DefenseStrategy] = set()

    def plan(self, event: ThreatEvent) -> DefensePlan:
        base = super().plan(event)
        if not event.is_threat:
            return base
        # union with currently deployed (human-gated) evolved defenses
        strategies = list(dict.fromkeys(base.strategies + list(self.deployed_strategies)))
        return DefensePlan(strategies=strategies, rationale="meta+evolution", selected_by="full_sentinel")

    def deploy_genome_strategies(self, strategies: set[DefenseStrategy]) -> None:
        self.deployed_strategies |= strategies


CONDITIONS = ["vanilla", "static_filter", "reflection_defense", "meta_defense", "full_sentinel"]


def build_condition(name: str, context_dim: int, seed: int = 0) -> DefenseCondition:
    if name == "vanilla":
        return Vanilla()
    if name == "static_filter":
        return StaticFilter()
    if name == "reflection_defense":
        return ReflectionDefense()
    if name == "meta_defense":
        return MetaDefense(context_dim, seed=seed)
    if name == "full_sentinel":
        return FullSentinel(context_dim, seed=seed)
    raise ValueError(f"unknown condition {name!r}")
