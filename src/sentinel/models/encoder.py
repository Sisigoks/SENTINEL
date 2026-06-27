"""Frozen sentence encoder used by the anomaly screen, classifier, and signatures.

Implemented directly on ``transformers`` (AutoModel + mean pooling) rather than
``sentence-transformers``. Rationale:

* No fine-tuning of any base model (paper budget; Invariant against modifying the base).
* Avoids the ``sentence-transformers`` -> ``torchcodec`` -> FFmpeg native-lib chain that
  breaks on minimal cloud images (e.g. Colab missing ``libavutil``). transformers is
  already a hard dependency and has no such requirement.

Embeddings are deterministic, mean-pooled over the last hidden state with the attention
mask, and L2-normalized — the standard recipe for E5/BGE-style encoders. Runs on the GPU
(``device='cuda'``) in fp16 by default to share the device with the served LLM.
"""

from __future__ import annotations

import numpy as np

from ..core.logging import get_logger

log = get_logger(__name__)


class FrozenEncoder:
    """Wraps a frozen HF encoder; provides normalized, mean-pooled embeddings."""

    def __init__(
        self,
        model_name: str = "intfloat/e5-large-v2",
        device: str = "cuda",
        batch_size: int = 64,
        normalize: bool = True,
        max_length: int = 512,
        half_on_cuda: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        log.info("loading encoder", model=model_name, device=device)
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_length = max_length
        self._torch = torch

        self._tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if (half_on_cuda and "cuda" in device) else torch.float32
        try:
            self._model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device)
        except (RuntimeError, OSError) as exc:
            # e.g. CUDA OOM when sharing the GPU with the served LLM — fall back to CPU.
            log.warning("encoder GPU load failed; falling back to CPU", error=str(exc))
            self.device = "cpu"
            self._model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to("cpu")
        self._model.eval()
        self._dim = int(self._model.config.hidden_size)

    @property
    def dim(self) -> int:
        return self._dim

    def _mean_pool(self, last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _to_cpu(self) -> None:
        if self.device != "cpu":
            log.warning("encoder CUDA OOM during encode; migrating to CPU")
            self.device = "cpu"
            self._model = self._model.float().to("cpu")
            self._torch.cuda.empty_cache()

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        torch = self._torch
        enc = self._tok(
            batch, padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            model_out = self._model(**enc)
        pooled = self._mean_pool(model_out.last_hidden_state, enc["attention_mask"])
        if self.normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.float().cpu().numpy().astype(np.float32)

    def encode(self, texts: list[str], progress: bool = False, desc: str = "encoding") -> np.ndarray:
        """Return an (n, dim) float32 array of normalized, mean-pooled embeddings.

        Set ``progress=True`` to show a tqdm bar over batches (useful when embedding a large
        corpus during detector fitting); per-probe (size-1) calls leave it off."""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        out: list[np.ndarray] = []
        rng = range(0, len(texts), self.batch_size)
        if progress and len(texts) > self.batch_size:
            from tqdm.auto import tqdm
            rng = tqdm(rng, desc=desc, unit="batch", leave=False)
        for i in rng:
            batch = texts[i : i + self.batch_size]
            try:
                out.append(self._embed_batch(batch))
            except (RuntimeError, MemoryError) as exc:
                if "out of memory" in str(exc).lower() and self.device != "cpu":
                    self._to_cpu()
                    out.append(self._embed_batch(batch))  # retry on CPU
                else:
                    raise
        return np.vstack(out)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
