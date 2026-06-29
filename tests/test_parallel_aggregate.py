"""Tests for multi-GPU job planning and cross-model aggregation (no GPU needed)."""

from __future__ import annotations

import json
from pathlib import Path

from sentinel.aggregate import aggregate, load_runs
from sentinel.launcher import build_jobs, detect_gpus


def test_detect_gpus_is_nonnegative():
    assert detect_gpus() >= 0  # 0 on a CPU box, N on a GPU host


def test_build_jobs_fills_gpus():
    models = ["qwen3_14b", "deepseek_r1_distill_14b", "mistral_small", "qwen3_32b"]
    jobs = build_jobs(models, "conf/config_b200.yaml", "conf/config_b200_evo.yaml")
    assert len(jobs) == 8  # 4 flagship + 4 evo -> fills 8 B200s
    assert {j.model for j in jobs} == set(models)
    # without evo config -> one job per model
    assert len(build_jobs(models, "conf/config_b200.yaml", None)) == 4
    assert len(build_jobs(models, "conf/config_b200.yaml", "none")) == 4


def _fake_run(model: str, asr_by_cond: dict[str, list[float]]) -> dict:
    conds = {}
    for c, samples in asr_by_cond.items():
        conds[c] = {
            "final_asr_mean": sum(samples) / len(samples),
            "asr_aulc_mean": sum(samples) / len(samples),
            "final_asr_samples": samples,
            "asr_aulc_samples": samples,
            "detection_recall": 0.9,
        }
    return {"model": model, "conditions": conds}


def test_aggregate_two_way_anova(tmp_path: Path):
    runs = tmp_path / "runs"
    # two models, clear condition effect (vanilla high ASR, full_sentinel low)
    for model in ["Qwen", "DeepSeek"]:
        d = runs / model
        d.mkdir(parents=True)
        res = _fake_run(model, {
            "vanilla": [0.58, 0.60, 0.57],
            "static_filter": [0.40, 0.42, 0.39],
            "full_sentinel": [0.16, 0.15, 0.18],
        })
        (d / "results.json").write_text(json.dumps(res))

    loaded = load_runs(runs)
    assert set(loaded) == {"Qwen", "DeepSeek"}

    agg = aggregate(runs, tmp_path / "agg")
    assert agg["models"] == ["DeepSeek", "Qwen"] or set(agg["models"]) == {"Qwen", "DeepSeek"}
    assert "two_way_anova" in agg, agg.get("note")
    cond_term = [k for k in agg["two_way_anova"] if "B" in k and ":" not in k][0]
    assert agg["two_way_anova"][cond_term]["p"] < 0.01  # condition strongly significant
    assert "full_sentinel" in agg["effect_vs_vanilla_pooled"]
    assert (tmp_path / "agg" / "aggregate_results.json").exists()
