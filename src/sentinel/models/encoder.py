"""Frozen sentence encoder used by the anomaly screen, classifier, and signatures.

A *frozen* transformer encoder (no fine-tuning of the base LLM, per the paper's
budget and Invariant against modifying the base model). Embeddings are deterministic
and L2-normalized. Runs on the A100 (``device='cuda'``).
"""

from __future__ import annotations

import numpy as np

from ..core.logging import get_logger

log = get_logger(__name__)


class FrozenEncoder:
    """Wraps a SentenceTransformer; provides cached, normalized embeddings."""

    def __init__(
        self,
        model_name: str = "intfloat/e5-large-v2",
        device: str = "cuda",
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        log.info("loading encoder", model=model_name, device=device)
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = SentenceTransformer(model_name, device=device)
        self._model.eval()
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of normalized embeddings."""
        emb = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
