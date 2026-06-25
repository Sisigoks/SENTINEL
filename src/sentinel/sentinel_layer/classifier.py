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

from collections import Counter

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from ..core.logging import get_logger
from ..core.types import CascadeStage, StageResult, ThreatClass

log = get_logger(__name__)

_CLASSES: list[str] = ["BENIGN", *[c.value for c in ThreatClass]]


class NeuralThreatClassifier:
    """Multinomial logistic head over OWASP classes + BENIGN, calibrated when the data
    supports it.

    Calibration (``CalibratedClassifierCV``) needs >= cv samples per class. We adapt the
    number of folds to the smallest class count and fall back to an *uncalibrated* head
    when there are too few samples (e.g. tiny smoke-test corpora). This keeps both the
    smoke run (``corpus.repeat=2``) and the full grid (60/class) working without crashing.
    Note: sklearn >=1.7 removed ``multi_class`` from LogisticRegression (multinomial is the
    default for multiclass), so it is not passed.
    """

    def __init__(self, c: float = 1.0, random_state: int = 0) -> None:
        self._c = c
        self._random_state = random_state
        self._clf: CalibratedClassifierCV | LogisticRegression | None = None
        self._labels: list[str] = []
        self._fitted = False

    def _model(self):
        """The fitted estimator (calibrated or raw). Raises if not yet fit."""
        if self._clf is None:
            raise RuntimeError("classifier must be fit before use")
        return self._clf

    def _base(self) -> LogisticRegression:
        return LogisticRegression(
            C=self._c, max_iter=2000, class_weight="balanced", random_state=self._random_state
        )

    def fit(self, embeddings: np.ndarray, labels: list[str]) -> None:
        x = np.asarray(embeddings, dtype=np.float64)
        min_count = min(Counter(labels).values())
        base = self._base()

        if min_count >= 2:
            cv = min(3, min_count)
            method = "isotonic" if min_count >= 10 else "sigmoid"  # sigmoid is stabler on few pts
            clf = CalibratedClassifierCV(base, method=method, cv=cv)
            try:
                clf.fit(x, labels)
                self._clf = clf
            except Exception as exc:  # any calibration quirk -> uncalibrated fallback
                log.warning("calibration failed; using uncalibrated head",
                            error=str(exc), min_count=min_count)
                base = self._base()
                base.fit(x, labels)
                self._clf = base
        else:
            log.warning("too few samples/class to calibrate; using uncalibrated head",
                        min_count=min_count)
            base.fit(x, labels)
            self._clf = base

        self._labels = list(self._clf.classes_)
        self._fitted = True

    def predict(self, embedding: np.ndarray) -> StageResult:
        if not self._fitted:
            raise RuntimeError("classifier must be fit before predict")
        x = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        proba = self._model().predict_proba(x)[0]
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

        preds = [self._labels[int(np.argmax(p))] for p in self._model().predict_proba(embeddings)]
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
