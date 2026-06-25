"""Meta-defense bandit, threat graph, and metric-catalog tests."""

from __future__ import annotations

import numpy as np

from sentinel.core.types import (
    BehavioralSignature,
    DefenseStrategy,
    InputChannel,
    Outcome,
    SignatureAxis,
    ThreatClass,
    ThreatEvent,
)
from sentinel.graph.threat_graph import ThreatGraph
from sentinel.meta_defense.bandit import MetaDefenseSelector
from sentinel.metrics.catalog import (
    asr,
    asr_aulc,
    kl_divergence,
    shannon_diversity,
    signature_drift,
    time_to_hardening,
)


def _sig(values: dict[SignatureAxis, float] | None = None) -> BehavioralSignature:
    base = {a: 0.1 for a in SignatureAxis}
    if values:
        base.update(values)
    return BehavioralSignature(axes=base, residual=[0.0] * 4)


def _event(cls: ThreatClass, sig: BehavioralSignature) -> ThreatEvent:
    return ThreatEvent(
        event_id=cls.value + str(np.random.randint(1_000_000)),
        channel=InputChannel.USER_TASK,
        raw_text="x",
        is_threat=True,
        threat_class=cls,
        confidence=0.9,
        signature=sig,
    )


def test_bandit_learns_rewarding_arm():
    dim = 6 + 4
    sel = MetaDefenseSelector(dim, policy="linucb", max_strategies=1, seed=0)
    ctx = _sig().vector()
    # reward only when INPUT_SANITIZATION chosen
    for _ in range(200):
        sel.select(ctx)
        # teach the bandit that INPUT_SANITIZATION is rewarding for this context
        sel.update(ctx, [DefenseStrategy.INPUT_SANITIZATION], 1.0)
        sel.update(ctx, [DefenseStrategy.OUTPUT_VALIDATION], 0.0)
    final = sel.select(ctx)
    assert DefenseStrategy.INPUT_SANITIZATION in final


def test_threat_graph_recurrence_and_reuse():
    g = ThreatGraph()
    sig = _sig()
    for cyc in range(6):
        cls = ThreatClass.PROMPT_INJECTION if cyc % 2 == 0 else ThreatClass.TOOL_MISUSE
        ev = _event(cls, sig)
        g.add_event(ev, cyc)
        out = Outcome(event_id=ev.event_id, attack_succeeded=False, blocked=False)
        g.record_outcome(ev, [DefenseStrategy.INPUT_SANITIZATION], out)
    rec = g.recurrence()
    assert rec[ThreatClass.PROMPT_INJECTION.value]  # has occurrences
    reuse = g.defense_reuse()
    # input_sanitization defeated 2 distinct classes
    assert reuse[DefenseStrategy.INPUT_SANITIZATION.value] == 2


def test_threat_graph_similarity():
    g = ThreatGraph()
    s1 = _sig({SignatureAxis.TOOL_ABUSE: 0.9})
    s2 = _sig({SignatureAxis.TOOL_ABUSE: 0.85})
    g.add_event(_event(ThreatClass.TOOL_MISUSE, s1), 0)
    g.add_event(_event(ThreatClass.TOOL_MISUSE, s2), 1)
    nbrs = g.similar(s1, k=2)
    assert nbrs and nbrs[0][1] > 0.5


def test_metrics_basic():
    assert asr([True, False, True, False]) == 0.5
    assert asr_aulc([1.0, 0.0]) == 0.5
    tth = time_to_hardening([0.9, 0.6, 0.4, 0.03])
    assert tth[0.5] == 2 and tth[0.05] == 3
    assert kl_divergence({"a": 10}, {"a": 10}) < 1e-6
    assert shannon_diversity({"a": 1, "b": 1}) > 0.6
    d = signature_drift(np.array([1, 0, 0]), np.array([0, 1, 0]))
    assert d["cosine_distance"] > 0.9
