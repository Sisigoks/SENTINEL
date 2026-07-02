"""Stage 4 — SignatureExtractor: the hybrid behavioral signature (design doc §3.2).

signature = [a1..a6]  (interpretable behavioral axes in [0,1])
          ⊕ z         (learned residual embedding, PCA-projected encoder features
                        orthogonalized against the axis subspace)

The interpretable axes make Attack-Migration / Signature-Drift / Recurrence analyses
*scientifically meaningful*; the residual preserves discriminative power for
similarity search in the threat graph.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from ..core.types import (
    BehavioralSignature,
    InputChannel,
    SignatureAxis,
    StageResult,
)


class SignatureExtractor:
    """Computes the six behavioral axes from structural+semantic evidence, plus a
    learned residual fit by PCA on encoder embeddings."""

    def __init__(self, residual_dim: int = 16) -> None:
        self._pca = PCA(n_components=residual_dim, random_state=0)
        self._benign_centroid: np.ndarray | None = None
        self._fitted = False

    def fit(self, threat_embeddings: np.ndarray, benign_embeddings: np.ndarray) -> None:
        self._pca.fit(threat_embeddings)
        self._benign_centroid = benign_embeddings.mean(axis=0)
        self._fitted = True

    def _residual(self, embedding: np.ndarray) -> list[float]:
        if not self._fitted:
            return []
        z = self._pca.transform(embedding.reshape(1, -1))[0]
        return [float(v) for v in z]

    def extract(
        self,
        text: str,
        channel: InputChannel,
        embedding: np.ndarray,
        rule_detail: dict,
        anomaly_detail: dict,
        granted_scope: set[str] | None = None,
        requested_scope: set[str] | None = None,
    ) -> tuple[StageResult, BehavioralSignature]:
        granted_scope = granted_scope or set()
        requested_scope = requested_scope or set()
        untrusted = channel in {
            InputChannel.RETRIEVED_DOC,
            InputChannel.TOOL_OUTPUT,
            InputChannel.MEMORY_READ,
        }

        # axis 1: semantic intent distance from benign centroid (cosine -> [0,1])
        if self._benign_centroid is not None:
            cos = float(
                np.dot(embedding, self._benign_centroid)
                / (np.linalg.norm(embedding) * np.linalg.norm(self._benign_centroid) + 1e-9)
            )
            semantic_intent = float(np.clip((1.0 - cos) / 2.0, 0.0, 1.0))
        else:
            semantic_intent = 0.0

        # axis 2: objective drift — role redefinition / new-instruction structure
        objective_drift = 1.0 if rule_detail.get("role_redefinition") else 0.0

        # axis 3: privilege escalation — requested scope beyond granted
        excess = requested_scope - granted_scope
        privilege_escalation = float(np.clip(len(excess) / max(len(requested_scope), 1), 0.0, 1.0))
        if rule_detail.get("tool_param_risk"):
            privilege_escalation = max(privilege_escalation, 0.6)

        # axis 4: instruction-hierarchy violation — imperative authority in untrusted span
        instruction_hierarchy = 1.0 if (untrusted and rule_detail.get("imperative_in_untrusted")) else 0.0

        # axis 5: trust-boundary crossing — novelty in untrusted channel
        trust_boundary = float(anomaly_detail.get("maha_novelty", 0.0)) if untrusted else 0.0

        # axis 6: tool abuse — risky tool params / delimiter injection
        tool_abuse = 0.0
        if rule_detail.get("tool_param_risk"):
            tool_abuse = 0.8
        if rule_detail.get("delimiter_break"):
            tool_abuse = max(tool_abuse, 0.4)

        axes = {
            SignatureAxis.SEMANTIC_INTENT: semantic_intent,
            SignatureAxis.OBJECTIVE_DRIFT: objective_drift,
            SignatureAxis.PRIVILEGE_ESCALATION: privilege_escalation,
            SignatureAxis.INSTRUCTION_HIERARCHY_VIOLATION: instruction_hierarchy,
            SignatureAxis.TRUST_BOUNDARY_CROSSING: trust_boundary,
            SignatureAxis.TOOL_ABUSE: tool_abuse,
        }
        sig = BehavioralSignature(axes=axes, residual=self._residual(embedding))
        from ..core.types import CascadeStage

        return StageResult(
            stage=CascadeStage.SIGNATURE,
            flagged=True,
            score=float(np.mean(list(axes.values()))),
            detail={"signature": sig.model_dump()},
        ), sig
