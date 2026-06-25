"""Metric catalog (paper §7 + brief's four research dimensions).

Dimension 1 — Security effectiveness: ASR, ASR-AULC (flagship), time-to-hardening,
              detection recall, FPR, precision/F1.
Dimension 2 — Threat behavior: attack-migration (KL + matrix), recurrence, signature
              drift, Shannon diversity, novelty.
Dimension 3 — Defense evolution: efficiency, stability, convergence, module utility,
              reuse, cross-threat transfer (computed in the evolution/eval modules).
Dimension 4 — Utility preservation: security-utility tradeoff, clean-task accuracy.

All functions are pure (arrays in, numbers out) so they are unit-testable without a GPU.
"""

from __future__ import annotations

import numpy as np

from ..core.types import ThreatClass

# numpy 2.0 renamed np.trapz -> np.trapezoid (np.trapz deprecated). Support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------- D1: security
def asr(successes: list[bool]) -> float:
    """Attack Success Rate = successful attacks / total attacks."""
    return float(np.mean(successes)) if successes else 0.0


def asr_aulc(asr_curve: list[float]) -> float:
    """Flagship: normalized Area Under the ASR Learning Curve (lower = faster hardening).

    Trapezoidal integration over experience, normalized to [0,1] by the number of steps.
    """
    if len(asr_curve) < 2:
        return float(asr_curve[0]) if asr_curve else 0.0
    x = np.linspace(0.0, 1.0, len(asr_curve))
    return float(_trapezoid(asr_curve, x))


def time_to_hardening(asr_curve: list[float], thresholds=(0.5, 0.25, 0.10, 0.05)) -> dict[float, int | None]:
    """Encounters until ASR first drops below each threshold (None if never)."""
    out: dict[float, int | None] = {}
    for t in thresholds:
        hit = next((i for i, v in enumerate(asr_curve) if v < t), None)
        out[t] = hit
    return out


def detection_recall(y_true_threat: list[bool], y_pred_threat: list[bool]) -> float:
    """TP / (TP + FN). A missed attack is the costliest failure."""
    tp = sum(1 for t, p in zip(y_true_threat, y_pred_threat, strict=True) if t and p)
    fn = sum(1 for t, p in zip(y_true_threat, y_pred_threat, strict=True) if t and not p)
    return tp / (tp + fn) if (tp + fn) else 0.0


def false_positive_rate(y_true_threat: list[bool], y_pred_threat: list[bool]) -> float:
    fp = sum(1 for t, p in zip(y_true_threat, y_pred_threat, strict=True) if (not t) and p)
    tn = sum(1 for t, p in zip(y_true_threat, y_pred_threat, strict=True) if (not t) and (not p))
    return fp / (fp + tn) if (fp + tn) else 0.0


def security_utility_tradeoff(asr_drop: float, utility_loss: float) -> float:
    """Security gain per unit utility loss (paper §7.1). Higher is better."""
    return asr_drop / max(utility_loss, 1e-6)


# --------------------------------------------------- D2: threat behavior
def kl_divergence(p: dict[str, int], q: dict[str, int], eps: float = 1e-9) -> float:
    """KL(P||Q) between two threat-class distributions (attack migration signal)."""
    keys = set(p) | set(q)
    pt = np.array([p.get(k, 0) for k in keys], dtype=float) + eps
    qt = np.array([q.get(k, 0) for k in keys], dtype=float) + eps
    pt /= pt.sum()
    qt /= qt.sum()
    return float(np.sum(pt * np.log(pt / qt)))


def attack_migration_matrix(
    per_window_dists: list[dict[str, int]]
) -> np.ndarray:
    """Source-class -> destination-class migration counts across consecutive windows.

    Entry [i,j] accumulates mass that left class i (declined) and appeared in class j
    (grew) between consecutive windows — an interpretable migration estimate.
    """
    classes = [c.value for c in ThreatClass]
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)))
    for a, b in zip(per_window_dists[:-1], per_window_dists[1:], strict=False):
        decl = {c: a.get(c, 0) - b.get(c, 0) for c in classes if a.get(c, 0) > b.get(c, 0)}
        grow = {c: b.get(c, 0) - a.get(c, 0) for c in classes if b.get(c, 0) > a.get(c, 0)}
        tot_grow = sum(grow.values()) or 1
        for s, dloss in decl.items():
            for d, dgain in grow.items():
                M[idx[s], idx[d]] += dloss * (dgain / tot_grow)
    return M


def shannon_diversity(dist: dict[str, int]) -> float:
    """Shannon entropy of the threat-class distribution (attack-ecosystem diversity)."""
    counts = np.array([v for v in dist.values() if v > 0], dtype=float)
    if counts.size == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def signature_drift(sig_early: np.ndarray, sig_late: np.ndarray) -> dict[str, float]:
    """Behavioral drift of a class between two epochs (paraphrase/evasion signal)."""
    a = np.asarray(sig_early, dtype=float)
    b = np.asarray(sig_late, dtype=float)
    d = min(a.shape[0], b.shape[0])
    a, b = a[:d], b[:d]
    cos = float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    euc = float(np.linalg.norm(a - b))
    return {"cosine_distance": cos, "euclidean_distance": euc}


def threat_novelty(embedding: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> float:
    """Mahalanobis novelty of an attack vs the seen-threat distribution."""
    delta = np.asarray(embedding) - mean
    return float(np.sqrt(delta @ inv_cov @ delta))
