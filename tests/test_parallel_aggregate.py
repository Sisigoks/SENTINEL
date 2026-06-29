"""Tests for multi-GPU job planning and cross-model aggregation (no GPU needed)."""

from __future__ import annotations

import json
from pathlib import Path

from sentinel.aggregate import aggregate, load_runs
from sentinel.launcher import build_jobs, detect_gpus


def test_detect_gpus_is_nonnegative():
    assert detect_gpus() >= 0  # 0 on a CPU box, N on a GPU host


def test_build_jobs_shards_to_fill_gpus():
    models = ["qwen3_14b", "deepseek_r1_distill_14b", "mistral_small", "qwen3_32b"]
    # 8 GPUs, 4 models, seeds [0,1,2] -> replicas=2 -> 8 jobs (fills 8 GPUs)
    jobs = build_jobs(models, "conf/config.yaml", n_gpus=8, runs_dir="experiments/runs")
    assert len(jobs) == 8
    assert {j.model for j in jobs} == set(models)
    # first shard of each model runs evolution; later shards are grid-only
    def evo_off(j):
        return any("run_evolution=false" in o for o in j.overrides)
    first = [j for j in jobs if not evo_off(j)]
    later = [j for j in jobs if evo_off(j)]
    assert len(first) == 4 and len(later) == 4
    # 4 GPUs, 4 models -> one job per model (replicas=1)
    assert len(build_jobs(models, "conf/config.yaml", n_gpus=4, runs_dir="x")) == 4


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
    assert set(agg["models"]) == {"Qwen", "DeepSeek"}
    assert "two_way_anova" in agg, agg.get("note")
    cond_term = [k for k in agg["two_way_anova"] if "B" in k and ":" not in k][0]
    assert agg["two_way_anova"][cond_term]["p"] < 0.01  # condition strongly significant
    assert "full_sentinel" in agg["effect_vs_vanilla_pooled"]
    assert (tmp_path / "agg" / "aggregate_results.json").exists()


def test_load_runs_merges_seed_shards(tmp_path: Path):
    # two shard dirs for the SAME model (different seeds) must merge into one model entry
    runs = tmp_path / "runs"
    (runs / "_shard0" / "Qwen").mkdir(parents=True)
    (runs / "_shard1" / "Qwen").mkdir(parents=True)
    (runs / "_shard0" / "Qwen" / "results.json").write_text(
        json.dumps(_fake_run("Qwen", {"vanilla": [0.6, 0.58]})))
    (runs / "_shard1" / "Qwen" / "results.json").write_text(
        json.dumps(_fake_run("Qwen", {"vanilla": [0.62]})))
    merged = load_runs(runs)
    assert set(merged) == {"Qwen"}
    assert len(merged["Qwen"]["conditions"]["vanilla"]["final_asr_samples"]) == 3  # 2 + 1 seeds
