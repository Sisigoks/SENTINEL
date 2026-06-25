"""Experiment grids (paper §5.2, §6).

Runs the adversarial grid (defenses x models x threats) and the clean grid (capability
retention). Produces per-condition ASR learning curves (windowed over experience) — the
raw material for the flagship ASR-AULC and the page-1 figure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..agent import SentinelAgent
from ..core.logging import get_logger
from ..corpora.loaders import ProbeCorpus
from ..eval.conditions import DefenseCondition
from ..metrics.catalog import asr, asr_aulc

log = get_logger(__name__)


@dataclass
class ConditionRun:
    condition: str
    model: str
    seed: int
    asr_curve: list[float] = field(default_factory=list)
    final_asr: float = 0.0
    asr_aulc: float = 0.0
    per_probe_success: list[bool] = field(default_factory=list)
    detection_pred: list[bool] = field(default_factory=list)
    detection_true: list[bool] = field(default_factory=list)
    latency_s: float = 0.0
    tokens: int = 0


def run_adversarial(
    agent: SentinelAgent,
    condition: DefenseCondition,
    corpus: ProbeCorpus,
    *,
    model_name: str,
    seed: int = 0,
    window: int = 12,
    shuffle: bool = True,
) -> ConditionRun:
    """Stream probes through the condition, recording windowed ASR (the learning curve)."""
    rng = np.random.default_rng(seed)
    probes = corpus.seen()
    order = rng.permutation(len(probes)) if shuffle else np.arange(len(probes))

    run = ConditionRun(condition=condition.name, model=model_name, seed=seed)
    t0 = time.time()
    window_succ: list[bool] = []
    for cycle, i in enumerate(order):
        probe = probes[i]
        event, outcome = agent.run_probe(probe, condition, cycle)
        run.per_probe_success.append(outcome.attack_succeeded)
        run.detection_pred.append(event.is_threat)
        run.detection_true.append(True)  # every probe is a real attack
        run.tokens += outcome.tokens
        window_succ.append(outcome.attack_succeeded)
        if len(window_succ) >= window:
            run.asr_curve.append(asr(window_succ))
            window_succ = []
    if window_succ:
        run.asr_curve.append(asr(window_succ))

    run.final_asr = asr(run.per_probe_success[-window:]) if run.per_probe_success else 0.0
    run.asr_aulc = asr_aulc(run.asr_curve)
    run.latency_s = time.time() - t0
    log.info("adversarial run", condition=condition.name, model=model_name,
             final_asr=run.final_asr, aulc=run.asr_aulc)
    return run


def run_clean(
    agent: SentinelAgent, condition: DefenseCondition, clean_tasks: list[tuple[str, str]],
    *, model_name: str,
) -> float:
    """Clean-task accuracy under the condition (utility retention).

    clean_tasks: list of (question, expected_substring). Returns accuracy in [0,1].
    """
    from ..core.types import InputChannel, ThreatClass
    from ..corpora.loaders import Probe

    correct = 0
    for q, expected in clean_tasks:
        probe = Probe(text=q, threat_class=ThreatClass.PROMPT_INJECTION,
                      channel=InputChannel.USER_TASK, source="builtin-illustrative")
        event = agent.screen(probe)
        # benign tasks should NOT be flagged; if flagged it's a false positive but still solved
        plan = condition.plan(event)
        from .. import agent as agent_mod
        from ..defenses.modules import DefenseContext, apply_defenses
        ctx = DefenseContext(agent_mod.SYSTEM_PROMPT, q, "", InputChannel.USER_TASK, set(), set())
        defended = apply_defenses(plan.strategies, ctx)
        res = agent.substrate.solve(defended.system_prompt, defended.user_task, agent.solve_cfg)
        if expected.lower() in res.answer.lower():
            correct += 1
    return correct / max(len(clean_tasks), 1)
