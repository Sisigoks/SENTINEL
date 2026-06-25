"""Stage 2 — AnomalyScreen: embedding-space novelty / out-of-distribution detection.

Two complementary detectors over the frozen-encoder manifold (design doc G2):

* **Mahalanobis distance** to the benign distribution — parametric, strong when the
  benign manifold is roughly unimodal/Gaussian in encoder space.
* **Isolation Forest** — non-parametric, handles multimodal benign data and isolates
  rare points with few splits.

The novelty score is the max of the two normalized scores (fail-closed: either
detector firing routes the input deeper). This catches paraphrases and zero-days
that route around Stage 1's structural rules.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from ..core.types import CascadeStage, StageResult


class AnomalyScreen:
    def __init__(self, contamination: float = 0.05, random_state: int = 0) -> None:
        self._iforest = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=random_state
        )
        self._mean: np.ndarray | None = None
        self._inv_cov: np.ndarray | None = None
        self._maha_threshold: float = np.inf
        self._fitted = False

    def fit(self, benign_embeddings: np.ndarray) -> None:
        """Fit on benign reference embeddings (the in-distribution manifold)."""
        x = np.asarray(benign_embeddings, dtype=np.float64)
        self._iforest.fit(x)
        self._mean = x.mean(axis=0)
        cov = np.cov(x, rowvar=False)
        # ridge regularization for numerical stability in high dim
        cov += np.eye(cov.shape[0]) * 1e-3
        self._inv_cov = np.linalg.pinv(cov)
        d = self._mahalanobis(x)
        self._maha_threshold = float(np.quantile(d, 0.95))
        self._fitted = True

    def _mahalanobis(self, x: np.ndarray) -> np.ndarray:
        assert self._mean is not None and self._inv_cov is not None
        delta = x - self._mean
        return np.sqrt(np.einsum("ij,jk,ik->i", delta, self._inv_cov, delta))

    def screen(self, embedding: np.ndarray) -> StageResult:
        if not self._fitted:
            raise RuntimeError("AnomalyScreen.fit must be called before screen")
        x = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        maha = float(self._mahalanobis(x)[0])
        # IsolationForest: lower score_samples => more anomalous. Map to [0,1] novelty.
        iso_raw = float(self._iforest.score_samples(x)[0])
        iso_novelty = 1.0 / (1.0 + np.exp(iso_raw * 4.0))  # squashing
        maha_novelty = float(np.clip(maha / (self._maha_threshold + 1e-9), 0.0, 2.0) / 2.0)
        novelty = max(iso_novelty, maha_novelty)
        flagged = novelty >= 0.5
        return StageResult(
            stage=CascadeStage.ANOMALY,
            flagged=flagged,
            score=novelty,
            detail={"mahalanobis": maha, "iso_novelty": iso_novelty, "maha_novelty": maha_novelty},
        )
