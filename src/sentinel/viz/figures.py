"""Publication figures (paper §8 + brief figure list).

Design goals (driven by real-run feedback):
  * **Axes adapt to the recorded values** — ASR axes autoscale with headroom instead of a
    fixed 0..1 box, so small effects are visible rather than crushed into a corner.
  * **Overlapping series stay distinguishable** — every condition gets a distinct
    (color, linestyle, marker) so curves that share a value (e.g. several defenses at the
    same ASR) can still be told apart.
  * **Color AND black-and-white** — every figure is written twice: ``<name>.png`` (color)
    and ``<name>_bw.png`` (grayscale-safe, relies on linestyle/marker/hatch), so the same
    artifact suits color screens and B&W print.

All figures take already-computed metric data (testable without a GPU) and return the
color PNG path. Saved at 300 DPI.
"""

from __future__ import annotations

from collections.abc import Callable
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

# Distinct visual encodings so series remain separable in color and in grayscale.
_LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]


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


def _series_style(idx: int, mode: str) -> dict:
    """Per-series plot kwargs. Color mode uses the colormap; bw uses black + linestyle/marker."""
    ls = _LINESTYLES[idx % len(_LINESTYLES)]
    mk = _MARKERS[idx % len(_MARKERS)]
    if mode == "bw":
        return dict(color="0.15", linestyle=ls, marker=mk, markerfacecolor="white",
                    markeredgecolor="0.15", markersize=5.5, linewidth=1.8, markeredgewidth=1.0)
    return dict(color=plt.get_cmap("tab10")(idx % 10), linestyle=ls, marker=mk,
                markersize=5.0, linewidth=2.0)


def _cond_idx(cond: str) -> int:
    return _CONDITION_ORDER.index(cond) if cond in _CONDITION_ORDER else 0


def _autoscale_rate(ax, values: list[float]) -> None:
    """Set a 0-based y-limit with headroom so a rate fills the plot (not crushed at the edge)."""
    vmax = max(values) if values else 1.0
    top = min(1.0, max(vmax * 1.25 + 0.02, 0.08))  # floor so all-zero curves are still visible
    ax.set_ylim(0.0, top)


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{name}.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return png


def _dual(out: Path, name: str, figsize: tuple[float, float],
          draw: Callable[[object, object, str], None]) -> Path:
    """Render ``draw`` twice — color (<name>.png) and grayscale (<name>_bw.png)."""
    color_path = None
    for mode in ("color", "bw"):
        set_paper_style()
        fig, ax = plt.subplots(figsize=figsize)
        draw(fig, ax, mode)
        path = _save(fig, out, name if mode == "color" else f"{name}_bw")
        if mode == "color":
            color_path = path
    assert color_path is not None
    return color_path


# --------------------------------------------------------------------------- ASR curves
def plot_asr_curves(
    curves: dict[str, list[float]], out: Path, name: str = "fig1_asr_curves",
    ci: dict[str, tuple[list[float], list[float]]] | None = None,
) -> Path:
    """THE page-1 figure: ASR vs defensive experience, one curve per condition."""
    def draw(fig, ax, mode):
        allv: list[float] = []
        for cond in _CONDITION_ORDER:
            if cond not in curves:
                continue
            y = list(curves[cond])
            x = np.arange(len(y))
            allv += y
            stl = _series_style(_cond_idx(cond), mode)
            ax.plot(x, y, label=_CONDITION_LABEL[cond], **stl)
            if ci and cond in ci:
                lo, hi = ci[cond]
                ax.fill_between(x, lo, hi, alpha=0.10, color=stl["color"], linewidth=0)
        ax.set_xlabel("Defensive experience (encounter window)")
        ax.set_ylabel("Attack Success Rate")
        ax.set_title("ASR vs. defensive experience")
        _autoscale_rate(ax, allv)
        ax.margins(x=0.03)
        ax.legend(loc="best", fontsize=8, ncol=2)
    return _dual(out, name, (6.4, 4.3), draw)


# --------------------------------------------------------------------------- heatmaps
def _heatmap(matrix, row_labels, col_labels, out, name, title, cbar_label, annotate):
    def draw(fig, ax, mode):
        cmap = "magma" if mode == "color" else "gray_r"
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_title(title)
        if annotate:
            vmax = float(np.max(matrix)) if matrix.size else 1.0
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = matrix[i, j]
                    # contrast-aware text color
                    dark = (val / (vmax + 1e-9)) > 0.5
                    color = "white" if (dark and mode == "color") else ("white" if dark else "black")
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    return _dual(out, name, (6.6, 5.6), draw)


def plot_migration_heatmap(matrix: np.ndarray, out: Path, name: str = "fig_migration") -> Path:
    labels = [c.value for c in ThreatClass]
    return _heatmap(np.asarray(matrix), labels, labels, out, name,
                    "Attack migration", "migrated mass", annotate=False)


def plot_cross_transfer_matrix(
    matrix: np.ndarray, row_labels: list[str], col_labels: list[str],
    out: Path, name: str = "fig_transfer", title: str = "Cross transfer (ΔASR)",
) -> Path:
    return _heatmap(np.asarray(matrix), row_labels, col_labels, out, name, title,
                    "ASR", annotate=True)


# --------------------------------------------------------------------------- evolution
def plot_defense_evolution(history: list[dict], out: Path, name: str = "fig_evolution") -> Path:
    def draw(fig, ax, mode):
        gens = list(range(len(history)))
        best = [h["best_asr"] for h in history]
        mean = [h["pop_mean_asr"] for h in history]
        cov = [h["archive_coverage"] for h in history]
        s0, s1 = _series_style(0, mode), _series_style(1, mode)
        ax.plot(gens, best, label="best ASR", **s0)
        ax.plot(gens, mean, label="pop mean ASR", **s1)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Attack Success Rate")
        _autoscale_rate(ax, best + mean)
        ax2 = ax.twinx()
        cov_style = _series_style(2, mode)
        cov_style["color"] = "0.45" if mode == "bw" else "tab:green"
        ax2.plot(gens, cov, label="QD coverage", **cov_style)
        ax2.set_ylabel("MAP-Elites coverage")
        ax2.set_ylim(0, 1.02)
        ax2.grid(False)
        ax.set_title("Defensive architecture evolution")
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], loc="best", fontsize=8)
    return _dual(out, name, (6.2, 4.2), draw)


def plot_stability(history: list[dict], out: Path, name: str = "fig_stability") -> Path:
    def draw(fig, ax, mode):
        churn = [h["module_churn"] for h in history]
        ax.plot(range(len(history)), churn, **_series_style(0, mode))
        ax.set_xlabel("Generation")
        ax.set_ylabel("Module churn")
        ax.set_title("Defense stability (lower = converged)")
        ax.set_ylim(bottom=0)
        ax.margins(x=0.03)
    return _dual(out, name, (6.0, 4.0), draw)


# --------------------------------------------------------------------------- recurrence / drift
def plot_recurrence(recurrence: dict[str, list[int]], out: Path, name: str = "fig_recurrence") -> Path:
    def draw(fig, ax, mode):
        for i, (cls, cycles) in enumerate(sorted(recurrence.items())):
            if not cycles:
                continue
            bins = min(20, max(len(set(cycles)), 2))
            hist, edges = np.histogram(cycles, bins=bins)
            ax.plot(edges[:-1], hist, label=cls, **_series_style(i, mode))
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Threat frequency")
        ax.set_title("Threat recurrence over time")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=6, ncol=2, loc="best")
    return _dual(out, name, (6.6, 4.2), draw)


def plot_signature_drift(drift_by_class: dict[str, list[float]], out: Path,
                         name: str = "fig_drift") -> Path:
    def draw(fig, ax, mode):
        allv: list[float] = []
        for i, (cls, series) in enumerate(sorted(drift_by_class.items())):
            allv += list(series)
            ax.plot(range(len(series)), series, label=cls, **_series_style(i, mode))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Signature drift (cosine)")
        ax.set_title("Behavioral signature drift")
        if allv:
            ax.set_ylim(0, max(allv) * 1.2 + 1e-3)
        ax.legend(fontsize=6, ncol=2, loc="best")
    return _dual(out, name, (6.4, 4.2), draw)


# --------------------------------------------------------------------------- pareto / ablation / bars
def plot_security_utility_pareto(
    points: dict[str, tuple[float, float]], out: Path, name: str = "fig_pareto"
) -> Path:
    """x = clean-task accuracy (utility), y = 1-ASR (security). Upper-right is best.
    Axes autoscale to the data with padding so points are not stuck in a corner."""
    def draw(fig, ax, mode):
        xs = [p[0] for p in points.values()]
        ys = [p[1] for p in points.values()]
        for i, (cond, (util, sec)) in enumerate(points.items()):
            stl = _series_style(_cond_idx(cond), mode)
            ax.scatter(util, sec, s=110, color=stl["color"], marker=stl["marker"],
                       edgecolor="black", linewidth=0.8, zorder=3,
                       label=_CONDITION_LABEL.get(cond, cond))
            ax.annotate(_CONDITION_LABEL.get(cond, cond), (util, sec), fontsize=7,
                        xytext=(6, 4), textcoords="offset points")
        # pad limits around the data so nothing sits on an edge
        def lim(vals):
            lo, hi = min(vals), max(vals)
            pad = max((hi - lo) * 0.25, 0.05)
            return max(0.0, lo - pad), min(1.02, hi + pad)
        ax.set_xlim(*lim(xs))
        ax.set_ylim(*lim(ys))
        ax.set_xlabel("Clean-task accuracy (utility)")
        ax.set_ylabel("Security (1 − ASR)")
        ax.set_title("Security–utility tradeoff")
        ax.legend(fontsize=7, loc="best")
    return _dual(out, name, (5.8, 4.6), draw)


def plot_ablation(asr_by_config: dict[str, float], out: Path, name: str = "fig_ablation") -> Path:
    def draw(fig, ax, mode):
        configs = list(asr_by_config)
        vals = [asr_by_config[c] for c in configs]
        if mode == "bw":
            ax.barh(configs, vals, color="0.7", edgecolor="black", hatch="//")
        else:
            ax.barh(configs, vals, color="steelblue", edgecolor="black")
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.01, i, f"{v:.2f}", va="center", fontsize=7)
        ax.set_xlabel("Attack Success Rate")
        ax.set_xlim(0, max(vals) * 1.18 + 1e-3)
        ax.set_title("Ablation: marginal contribution of each component")
        ax.invert_yaxis()
    return _dual(out, name, (6.6, 4.2), draw)


def plot_transfer_bars(transfer: dict[str, float], out: Path, name: str = "fig_cross_threat") -> Path:
    """Leave-one-class-out cross-threat transfer: held-out class vs ASR (bar)."""
    def draw(fig, ax, mode):
        classes = list(transfer)
        vals = [transfer[c] for c in classes]
        if mode == "bw":
            ax.bar(classes, vals, color="0.7", edgecolor="black", hatch="\\\\")
        else:
            ax.bar(classes, vals, color="indianred", edgecolor="black")
        ax.set_ylabel("ASR on held-out class")
        ax.set_xlabel("Held-out threat class")
        ax.set_title("Cross-threat transfer (leave-one-class-out)")
        ax.set_ylim(0, (max(vals) if vals else 1.0) * 1.2 + 1e-3)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    return _dual(out, name, (6.6, 4.2), draw)
