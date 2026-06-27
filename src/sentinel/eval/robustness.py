"""Adversarial robustness harnesses (brief: required for publication).

These EVALUATE the defender under harder conditions; they do NOT generate novel attacks
(Invariant I1). Each harness *re-presents existing, loaded probes* under a transformed
delivery channel — multi-turn conversation, long-context padding, retrieval/memory
injection, tool-abuse simulation — using only benign filler and the probe's own text.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..agent import SentinelAgent
from ..core.types import InputChannel
from ..corpora.loaders import Probe, ProbeCorpus, benign_corpus
from ..eval.conditions import DefenseCondition
from ..metrics.catalog import asr


@dataclass
class RobustnessResult:
    setting: str
    value: float | int
    asr: float
    detection_recall: float


def _detection_recall(events) -> float:
    if not events:
        return 0.0
    return sum(1 for e in events if e.is_threat) / len(events)


def multi_turn(
    agent: SentinelAgent, condition: DefenseCondition, corpus: ProbeCorpus,
    turns: tuple[int, ...] = (2, 5, 10, 20), seed: int = 0,
) -> list[RobustnessResult]:
    """Embed each probe at the final turn of an n-turn benign conversation."""
    rng = np.random.default_rng(seed)
    benign = benign_corpus(64)
    results = []
    for n in turns:
        succ, events = [], []
        for cycle, p in enumerate(corpus.seen()):
            history = "\n".join(rng.choice(benign, size=max(n - 1, 0), replace=True))
            probe = Probe(text=f"{history}\n{p.text}", threat_class=p.threat_class,
                          channel=p.channel, source=p.source, canary=p.canary,
                          requested_scope=p.requested_scope, granted_scope=p.granted_scope)
            event, outcome = agent.run_probe(probe, condition, cycle)
            succ.append(outcome.attack_succeeded)
            events.append(event)
        results.append(RobustnessResult("multi_turn", n, asr(succ), _detection_recall(events)))
    return results


def context_length(
    agent: SentinelAgent, condition: DefenseCondition, corpus: ProbeCorpus,
    lengths: tuple[int, ...] = (1000, 4000, 16000, 64000), seed: int = 0,
    max_context_tokens: int = 8192, output_reserve: int = 1024,
) -> list[RobustnessResult]:
    """Pad benign context to target token lengths around each probe.

    Target lengths are CLAMPED to the model's usable context (``max_context_tokens`` minus a
    reserve for the probe + generated output), so a 64k request on an 8k model does not crash
    vLLM. Lengths beyond the cap are evaluated *at the cap* and flagged (capped=True), so the
    study still reports the longest context the model actually supports.
    """
    rng = np.random.default_rng(seed)
    benign = benign_corpus(64)
    budget = max(max_context_tokens - output_reserve, 256)
    results = []
    tested: set[int] = set()
    for L in lengths:
        eff = min(L, budget)
        if eff in tested:  # already evaluated this effective length (avoid duplicate work)
            continue
        tested.add(eff)
        approx_words = eff // 4 * 3  # ~ tokens -> words, split front/back padding
        half = max(approx_words // 2, 1)
        pad = " ".join(rng.choice(benign, size=max(half // 8, 1), replace=True))
        succ, events = [], []
        for cycle, p in enumerate(corpus.seen()):
            probe = Probe(text=f"{pad}\n{p.text}\n{pad}", threat_class=p.threat_class,
                          channel=p.channel, source=p.source, canary=p.canary)
            event, outcome = agent.run_probe(probe, condition, cycle)
            succ.append(outcome.attack_succeeded)
            events.append(event)
        results.append(RobustnessResult(
            f"context_length_{eff}{'(capped)' if eff < L else ''}", eff,
            asr(succ), _detection_recall(events)))
    return results


def channel_robustness(
    agent: SentinelAgent, condition: DefenseCondition, corpus: ProbeCorpus,
    channels=(InputChannel.RETRIEVED_DOC, InputChannel.MEMORY_READ, InputChannel.TOOL_OUTPUT),
) -> list[RobustnessResult]:
    """Re-present probes via retrieval / memory / tool channels (poisoning robustness)."""
    results = []
    for ch in channels:
        succ, events = [], []
        for cycle, p in enumerate(corpus.seen()):
            probe = Probe(text=p.text, threat_class=p.threat_class, channel=ch,
                          source=p.source, canary=p.canary,
                          requested_scope=p.requested_scope, granted_scope=p.granted_scope)
            event, outcome = agent.run_probe(probe, condition, cycle)
            succ.append(outcome.attack_succeeded)
            events.append(event)
        results.append(RobustnessResult(f"channel_{ch.value}", ch.value, asr(succ), _detection_recall(events)))
    return results
