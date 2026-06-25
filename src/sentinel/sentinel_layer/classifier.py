"""Stage 3 — NeuralThreatClassifier: frozen-encoder features + calibrated heads.

Design (doc G1): the base LLM is never fine-tuned. We freeze a transformer encoder
and learn lightweight per-class heads on top. Two properties are essential:

1. **Behavioral, paraphrase-robust** — operates in semantic embedding space, so novel
   phrasings of a known attack class are still caught (paper §2).
2. **Calibrated confidence** — the cascade gates downstream action on confidence, so
   probabilities must be meaningful. We use multinomial logistic regression with
   isotonic/temperature calibration (``CalibratedClassifierCV``).

Output: an OWASP ``ThreatClass`` (or BENIGN) plus a calibrated probability.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from ..core.types import CascadeStage, StageResult, ThreatClass

_CLASSES: list[str] = ["BENIGN", *[c.value for c in ThreatClass]]


class NeuralThreatClassifier:
    """One-vs-rest calibrated classifier over OWASP classes + BENIGN."""

    def __init__(self, c: float = 1.0, random_state: int = 0) -> None:
        base = LogisticRegression(
            C=c, max_iter=2000, multi_class="multinomial", class_weight="balanced"
        )
        self._clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self._labels: list[str] = []
        self._fitted = False

    def fit(self, embeddings: np.ndarray, labels: list[str]) -> None:
        x = np.asarray(embeddings, dtype=np.float64)
        self._clf.fit(x, labels)
        self._labels = list(self._clf.classes_)
        self._fitted = True

    def predict(self, embedding: np.ndarray) -> StageResult:
        if not self._fitted:
            raise RuntimeError("classifier must be fit before predict")
        x = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        proba = self._clf.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        label = self._labels[idx]
        conf = float(proba[idx])
        is_threat = label != "BENIGN"
        return StageResult(
            stage=CascadeStage.NEURAL,
            flagged=is_threat,
            score=conf,
            detail={
                "label": label,
                "proba": {lbl: float(p) for lbl, p in zip(self._labels, proba, strict=True)},
            },
        )

    def macro_metrics(self, embeddings: np.ndarray, labels: list[str]) -> dict[str, float]:
        """Per-class P/R/F1 + macro/weighted — used at the Sentinel phase gate (F1>=0.75)."""
        from sklearn.metrics import precision_recall_fscore_support

        preds = [self._labels[int(np.argmax(p))] for p in self._clf.predict_proba(embeddings)]
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, labels=self._labels, average="macro", zero_division=0
        )
        pw, rw, f1w, _ = precision_recall_fscore_support(
            labels, preds, labels=self._labels, average="weighted", zero_division=0
        )
        return {
            "macro_precision": float(p),
            "macro_recall": float(r),
            "macro_f1": float(f1),
            "weighted_f1": float(f1w),
        }
