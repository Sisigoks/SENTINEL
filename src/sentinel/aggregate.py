"""Cross-model aggregation (paper §6.2: model x condition ANOVA).

After per-model runs complete (ideally in parallel, one model per GPU), this reads every
``experiments/runs/<model>/results.json`` and produces the multi-model evidence:

  * Two-way ANOVA (model x condition) on final ASR — tests the model-agnostic claim and the
    model x condition interaction.
  * Tukey HSD post-hoc across defense conditions (pooled over models).
  * Effect sizes (Cohen's d / Hedges g) for each condition vs Vanilla, pooled over models.
  * A model x condition ASR heatmap (color + B&W).

It needs >= 2 models, each with raw per-seed ``final_asr_samples`` (saved by the experiment
driver). With one model it still emits the per-condition summary and notes ANOVA needs >= 2.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import viz
from .core.logging import get_logger
from .stats import tests as st

log = get_logger(__name__)

_CONDITION_ORDER = ["vanilla", "static_filter", "reflection_defense", "meta_defense", "full_sentinel"]


def _merge_model_results(parts: list[dict]) -> dict:
    """Merge several results.json for the SAME model (e.g. seed-shards from run-parallel):
    concatenate per-condition samples and recompute means; keep richest detection/extras."""
    if len(parts) == 1:
        return parts[0]
    merged = dict(parts[0])
    merged_conds: dict[str, dict] = {}
    cond_names = {c for p in parts for c in p.get("conditions", {})}
    for cond in cond_names:
        fa: list[float] = []
        au: list[float] = []
        recall = []
        for p in parts:
            c = p.get("conditions", {}).get(cond)
            if not c:
                continue
            fa += c.get("final_asr_samples", [c.get("final_asr_mean")] if "final_asr_mean" in c else [])
            au += c.get("asr_aulc_samples", [c.get("asr_aulc_mean")] if "asr_aulc_mean" in c else [])
            if "detection_recall" in c:
                recall.append(c["detection_recall"])
        merged_conds[cond] = {
            "final_asr_mean": float(np.mean(fa)) if fa else float("nan"),
            "asr_aulc_mean": float(np.mean(au)) if au else float("nan"),
            "final_asr_samples": fa,
            "asr_aulc_samples": au,
            "detection_recall": float(np.mean(recall)) if recall else None,
        }
    merged["conditions"] = merged_conds
    # prefer a part that actually computed detection / evolution
    for key in ("detection", "evolution", "ablation", "cross_threat_transfer"):
        for p in parts:
            if p.get(key):
                merged[key] = p[key]
                break
    return merged


def load_runs(runs_dir: str | Path) -> dict[str, dict]:
    """Recursively load every results.json and group by the model field, merging shards."""
    by_model: dict[str, list[dict]] = {}
    for f in sorted(Path(runs_dir).rglob("results.json")):
        try:
            res = json.loads(f.read_text())
        except Exception as exc:  # skip a half-written file
            log.warning("skip unreadable results", path=str(f), error=str(exc))
            continue
        model = res.get("model", f.parent.name)
        by_model.setdefault(model, []).append(res)
    return {m: _merge_model_results(parts) for m, parts in by_model.items()}


def aggregate(runs_dir: str | Path, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(runs_dir)
    if not runs:
        raise SystemExit(f"no results.json found under {runs_dir}")

    models = list(runs)
    conditions = [c for c in _CONDITION_ORDER
                  if any(c in r.get("conditions", {}) for r in runs.values())]

    # long-form samples for ANOVA / effect sizes
    values: list[float] = []
    f_model: list[str] = []
    f_cond: list[str] = []
    for model, res in runs.items():
        for cond in conditions:
            c = res.get("conditions", {}).get(cond)
            if not c:
                continue
            samples = c.get("final_asr_samples") or [c.get("final_asr_mean", float("nan"))]
            for v in samples:
                values.append(float(v))
                f_model.append(model)
                f_cond.append(cond)

    agg: dict = {"models": models, "conditions": conditions}

    # final-ASR matrix (rows = models, cols = conditions) for the heatmap + table
    matrix = np.full((len(models), len(conditions)), np.nan)
    for i, model in enumerate(models):
        for j, cond in enumerate(conditions):
            c = runs[model].get("conditions", {}).get(cond)
            if c:
                matrix[i, j] = c.get("final_asr_mean", np.nan)
    agg["final_asr_matrix"] = matrix.tolist()
    viz.plot_cross_transfer_matrix(
        np.nan_to_num(matrix), models, [_CONDITION_LABEL(c) for c in conditions],
        out, name="fig_model_condition_asr", title="Final ASR by model x condition")

    # two-way ANOVA (needs >= 2 models and >= 2 conditions with replication)
    if len(set(f_model)) >= 2 and len(set(f_cond)) >= 2 and len(values) >= 8:
        try:
            res_anova = st.two_way_anova(values, f_model, f_cond)
            agg["two_way_anova"] = res_anova.factors
            tukey = st.tukey_hsd(values, f_cond)
            agg["tukey_hsd"] = str(tukey)
        except Exception as exc:
            agg["two_way_anova_error"] = str(exc)
    else:
        agg["note"] = "two-way ANOVA needs >= 2 models with per-seed samples; run more models"

    # effect sizes vs vanilla, pooled across models
    base = [v for v, c in zip(values, f_cond, strict=True) if c == "vanilla"]
    agg["effect_vs_vanilla_pooled"] = {}
    pvals, names = [], []
    for cond in conditions:
        if cond == "vanilla":
            continue
        samp = [v for v, c in zip(values, f_cond, strict=True) if c == cond]
        if len(base) >= 2 and len(samp) >= 2:
            agg["effect_vs_vanilla_pooled"][cond] = {
                "cohens_d": st.cohens_d(base, samp), "hedges_g": st.hedges_g(base, samp),
                "mean_asr": float(np.mean(samp)),
            }
            try:
                from scipy.stats import ttest_ind
                _, p = ttest_ind(base, samp, equal_var=False)
                pvals.append(float(p))
                names.append(cond)
            except Exception:
                pass
    if pvals:
        holm = st.holm_bonferroni(pvals)
        bh = st.benjamini_hochberg(pvals)
        agg["multiple_comparison"] = {
            n: {"p": p, "holm_reject": h, "bh_reject": b}
            for n, p, h, b in zip(names, pvals, holm, bh, strict=True)
        }

    (out / "aggregate_results.json").write_text(json.dumps(agg, indent=2, default=str))
    log.info("aggregation complete", models=len(models), out=str(out))
    return agg


def _CONDITION_LABEL(c: str) -> str:
    return {
        "vanilla": "Vanilla", "static_filter": "Static", "reflection_defense": "Reflection",
        "meta_defense": "Meta-Def", "full_sentinel": "Full SENTINEL",
    }.get(c, c)
