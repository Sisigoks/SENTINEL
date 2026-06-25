"""Publication figures (paper §8 + brief figure list).

All figures use a consistent, conference-ready style and save vector PDF + PNG.
Functions take already-computed metric data (so they are testable without a GPU) and
return the saved path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / server
import matplotlib.pyplot as plt
import numpy as np

from ..core.types import ThreatClass

_CONDITION_ORDER = ["vanilla", "static_filter", "reflection_defense", "meta_defense", "full_sentinel"]
_CONDITION_LABEL = {
    "vanilla": "Vanilla",
    "static_filter": "Static Filter",
    "reflection_defense": "Reflection-Defense",
    "meta_defense": "Meta-Defense",
    "full_sentinel": "Full SENTINEL",
}


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "font.family": "serif",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _save(fig, out: Path, name: str) -> Path:
    # PNG only, at 300 DPI (set in set_paper_style) — publication-quality raster.
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{name}.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return png


def plot_asr_curves(
    curves: dict[str, list[float]], out: Path, name: str = "fig1_asr_curves",
    ci: dict[str, tuple[list[float], list[float]]] | None = None,
) -> Path:
    """THE page-1 figure: ASR vs defensive experience, one curve per condition."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond in _CONDITION_ORDER:
        if cond not in curves:
            continue
        y = curves[cond]
        x = np.arange(len(y))
        ax.plot(x, y, marker="o", ms=3, lw=2, label=_CONDITION_LABEL[cond])
        if ci and cond in ci:
            lo, hi = ci[cond]
            ax.fill_between(x, lo, hi, alpha=0.15)
    ax.set_xlabel("Defensive experience (encounter window)")
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("ASR vs. defensive experience")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    return _save(fig, out, name)


def plot_migration_heatmap(matrix: np.ndarray, out: Path, name: str = "fig_migration") -> Path:
    set_paper_style()
    labels = [c.value for c in ThreatClass]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Destination class")
    ax.set_ylabel("Source class")
    ax.set_title("Attack migration")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="migrated mass")
    return _save(fig, out, name)


def plot_defense_evolution(history: list[dict], out: Path, name: str = "fig_evolution") -> Path:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    gens = range(len(history))
    ax.plot(gens, [h["best_asr"] for h in history], marker="o", label="best ASR")
    ax.plot(gens, [h["pop_mean_asr"] for h in history], marker="s", label="pop mean ASR")
    ax2 = ax.twinx()
    ax2.plot(gens, [h["archive_coverage"] for h in history], color="green", ls="--", label="QD coverage")
    ax.set_xlabel("Generation")
    ax.set_ylabel("ASR")
    ax2.set_ylabel("MAP-Elites coverage")
    ax.set_title("Defensive architecture evolution")
    ax.legend(loc="upper right")
    return _save(fig, out, name)


def plot_recurrence(recurrence: dict[str, list[int]], out: Path, name: str = "fig_recurrence") -> Path:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for cls, cycles in recurrence.items():
        if not cycles:
            continue
        hist, edges = np.histogram(cycles, bins=min(20, max(len(cycles), 2)))
        ax.plot(edges[:-1], hist, label=cls, lw=1.2)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Threat frequency")
    ax.set_title("Threat recurrence over time")
    ax.legend(fontsize=6, ncol=2)
    return _save(fig, out, name)


def plot_security_utility_pareto(
    points: dict[str, tuple[float, float]], out: Path, name: str = "fig_pareto"
) -> Path:
    """x = clean-task accuracy (utility), y = 1-ASR (security). Upper-right is best."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for cond, (util, sec) in points.items():
        ax.scatter(util, sec, s=80)
        ax.annotate(_CONDITION_LABEL.get(cond, cond), (util, sec), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Clean-task accuracy (utility)")
    ax.set_ylabel("Security (1 − ASR)")
    ax.set_title("Security–utility tradeoff")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return _save(fig, out, name)


def plot_cross_transfer_matrix(
    matrix: np.ndarray, row_labels: list[str], col_labels: list[str],
    out: Path, name: str = "fig_transfer", title: str = "Cross transfer (ΔASR)",
) -> Path:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, out, name)


def plot_signature_drift(drift_by_class: dict[str, list[float]], out: Path, name: str = "fig_drift") -> Path:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for cls, series in drift_by_class.items():
        ax.plot(range(len(series)), series, label=cls, lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Signature drift (cosine)")
    ax.set_title("Behavioral signature drift")
    ax.legend(fontsize=6, ncol=2)
    return _save(fig, out, name)


def plot_stability(history: list[dict], out: Path, name: str = "fig_stability") -> Path:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(history)), [h["module_churn"] for h in history], marker="o")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Module churn")
    ax.set_title("Defense stability (lower = converged)")
    return _save(fig, out, name)


def plot_ablation(asr_by_config: dict[str, float], out: Path, name: str = "fig_ablation") -> Path:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    configs = list(asr_by_config)
    vals = [asr_by_config[c] for c in configs]
    ax.barh(configs, vals, color="steelblue")
    ax.set_xlabel("Attack Success Rate")
    ax.set_title("Ablation: marginal contribution of each component")
    ax.invert_yaxis()
    return _save(fig, out, name)
