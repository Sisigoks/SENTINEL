"""End-to-end pipeline smoke test on CPU.

Runs the FULL ``run_all`` driver — grid (batched), evolution, ablation, robustness,
cross-threat transfer, detection, statistics, figures, manifest — with a trivial in-test
backend + encoder. This catches runtime wiring bugs (lazy imports, missing symbols, bad
call signatures) that import-level tests miss, WITHOUT a GPU. The production path
(``build_backend`` -> vLLM) is not exercised here and is unaffected; ``run_all`` simply takes
the backend/encoder as arguments, so we pass lightweight test doubles.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from sentinel.corpora.loaders import load_builtin_seed_corpus
from sentinel.experiment import run_all
from sentinel.models.backend import Generation, GenerationConfig, ModelBackend


class _TestBackend(ModelBackend):
    """Deterministic backend that always refuses — exercises the full call graph, no GPU."""

    model_name = "test/refusing-model"

    def generate(self, prompt: str, cfg: GenerationConfig) -> Generation:
        return Generation(text="I cannot help with that; it is against policy.",
                          prompt_tokens=12, completion_tokens=8)

    def generate_batch(self, prompts: list[str], cfg: GenerationConfig) -> list[Generation]:
        return [self.generate(p, cfg) for p in prompts]
    # chat / chat_batch inherit the ABC defaults (which call generate/generate_batch)


class _TestEncoder:
    """Deterministic hash-based embeddings — no model download, no GPU."""

    model_name = "test/encoder"

    def __init__(self, dim: int = 24) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str], progress: bool = False, desc: str = "") -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        rows = []
        for t in texts:
            h = int(hashlib.sha256(t.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(h)
            v = rng.normal(size=self._dim)
            # nudge threat-ish text away from benign so the detector isn't degenerate
            if any(w in t.lower() for w in ("ignore", "reveal", "delete", "admin", "rm -rf", "system prompt")):
                v += 3.0
            rows.append(v.astype(np.float32))
        return np.vstack(rows)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def test_run_all_end_to_end(tmp_path: Path):
    corpus = load_builtin_seed_corpus(repeat=2)
    cfg = {
        "seed": 0,
        "solve": {"max_tokens": 16, "batch_size": 4},
        "model": {"max_model_len": 2048},
        "experiment": {
            "conditions": ["vanilla", "static_filter", "reflection_defense",
                           "meta_defense", "full_sentinel"],
            "window": 4,
            "seeds": [0, 1],
            "bootstrap": 200,
            "run_evolution": True,
            "run_ablation": True,
            "run_robustness": True,
        },
        "evolution": {
            "population_size": 4, "offspring_per_gen": 2, "generations": 2,
            "eval_probes": 6, "bypass_trigger": 5, "human_gate": "policy",
        },
    }
    out = tmp_path / "run"
    results = run_all(_TestBackend(), _TestEncoder(), corpus, cfg, str(out))

    # the whole pipeline produced its artifacts
    assert (out / "results.json").is_file()
    assert (out / "manifest.json").is_file()
    saved = json.loads((out / "results.json").read_text())
    for key in ("conditions", "detection", "stats", "evolution", "ablation",
                "robustness", "cross_threat_transfer", "hardware"):
        assert key in saved, f"missing results key: {key}"
    # all five conditions ran, each with raw per-seed samples for the ANOVA
    assert set(saved["conditions"]) == set(cfg["experiment"]["conditions"])
    assert len(saved["conditions"]["vanilla"]["final_asr_samples"]) == 2
    # detector quality computed (incl. FPR)
    assert "false_positive_rate" in saved["detection"]
    # figures written (PNG, color + B&W)
    figs = list((out / "figures").glob("*.png"))
    assert any("fig1_asr_curves" in f.name for f in figs)
    assert any(f.name.endswith("_bw.png") for f in figs)
