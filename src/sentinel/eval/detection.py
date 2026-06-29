"""Detector-quality evaluation: recall, FPR, and per-class precision/recall/F1.

The adversarial grid measures ASR (did the attack get through) but only sees attacks, so it
cannot measure the **False-Positive Rate** (benign inputs wrongly flagged) or per-class
classification quality — both Tier-1 metrics in the paper. This module screens the probe set
*and* a benign reference set through the Sentinel cascade (no LLM generation, so it is cheap)
and reports the full detection picture, including the confusion across OWASP classes.
"""

from __future__ import annotations

from ..core.types import InputChannel
from ..corpora.loaders import ProbeCorpus, benign_corpus
from ..metrics.catalog import detection_recall, false_positive_rate
from ..sentinel_layer.cascade import SentinelCascade


def evaluate_detection(cascade: SentinelCascade, corpus: ProbeCorpus, n_benign: int = 120) -> dict:
    benign = benign_corpus(n_benign)
    y_true_threat: list[bool] = []
    y_pred_threat: list[bool] = []
    y_true_cls: list[str] = []
    y_pred_cls: list[str] = []

    for p in corpus.seen():
        ev = cascade.screen(p.text, p.channel,
                            granted_scope=p.granted_scope, requested_scope=p.requested_scope)
        y_true_threat.append(True)
        y_pred_threat.append(ev.is_threat)
        y_true_cls.append(p.threat_class.value)
        y_pred_cls.append(ev.threat_class.value if ev.threat_class else "BENIGN")

    for t in benign:
        ev = cascade.screen(t, InputChannel.USER_TASK)
        y_true_threat.append(False)
        y_pred_threat.append(ev.is_threat)
        y_true_cls.append("BENIGN")
        y_pred_cls.append(ev.threat_class.value if ev.threat_class else "BENIGN")

    recall = detection_recall(y_true_threat, y_pred_threat)
    fpr = false_positive_rate(y_true_threat, y_pred_threat)

    # per-class precision/recall/F1 over the OWASP classes + BENIGN
    from sklearn.metrics import precision_recall_fscore_support
    labels = sorted(set(y_true_cls) | set(y_pred_cls))
    p_c, r_c, f1_c, support = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, labels=labels, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, labels=labels, average="macro", zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, labels=labels, average="weighted", zero_division=0)

    per_class = {
        lbl: {"precision": float(pp), "recall": float(rr), "f1": float(ff), "support": int(ss)}
        for lbl, pp, rr, ff, ss in zip(labels, p_c, r_c, f1_c, support, strict=True)
    }
    return {
        "detection_recall": float(recall),         # Tier 1: a missed attack is costliest
        "false_positive_rate": float(fpr),         # Tier 1: over-defensiveness guard
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),               # phase gate target: >= 0.75
        "weighted_f1": float(weighted_f1),
        "n_benign": n_benign,
        "n_threat": len(corpus.seen()),
        "per_class": per_class,
    }
