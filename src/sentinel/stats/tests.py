"""Statistical tests for conference-grade validation (brief: "no metric without a test").

Provides: 10k-bootstrap CIs, two-way ANOVA (model x condition), Tukey HSD post-hoc,
effect sizes (Cohen's d, Hedges g, eta-squared), multiple-comparison corrections
(Holm-Bonferroni, Benjamini-Hochberg), and two-sample power analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float
    variance: float


def bootstrap_ci(
    data: list[float] | np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    x = np.asarray(data, dtype=float)
    n = x.shape[0]
    if n == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), float("nan"))
    means = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(float(x.mean()), float(lo), float(hi), float(x.var(ddof=1) if n > 1 else 0.0))


def cohens_d(a: list[float], b: list[float]) -> float:
    a_, b_ = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a_), len(b_)
    sp = np.sqrt(((na - 1) * a_.var(ddof=1) + (nb - 1) * b_.var(ddof=1)) / max(na + nb - 2, 1))
    return float((a_.mean() - b_.mean()) / (sp + 1e-12))


def hedges_g(a: list[float], b: list[float]) -> float:
    d = cohens_d(a, b)
    n = len(a) + len(b)
    correction = 1 - (3 / (4 * n - 9)) if n > 3 else 1.0
    return float(d * correction)


def eta_squared(groups: list[list[float]]) -> float:
    grand = np.concatenate([np.asarray(g, float) for g in groups])
    gm = grand.mean()
    ss_between = sum(len(g) * (np.mean(g) - gm) ** 2 for g in groups)
    ss_total = float(np.sum((grand - gm) ** 2))
    return float(ss_between / (ss_total + 1e-12))


@dataclass(slots=True)
class AnovaResult:
    factors: dict[str, dict[str, float]]  # name -> {F, p, eta_sq}


def two_way_anova(
    values: list[float], factor_a: list[str], factor_b: list[str]
) -> AnovaResult:
    """Two-way ANOVA via OLS + type-II sums of squares (statsmodels).

    factor_a = model family, factor_b = defense condition (paper §6.2).
    """
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    df = pd.DataFrame({"y": values, "A": factor_a, "B": factor_b})
    model = ols("y ~ C(A) + C(B) + C(A):C(B)", data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    ss_total = table["sum_sq"].sum()
    out: dict[str, dict[str, float]] = {}
    for term in table.index:
        if term == "Residual":
            continue
        out[term] = {
            "F": float(table.loc[term, "F"]),
            "p": float(table.loc[term, "PR(>F)"]),
            "eta_sq": float(table.loc[term, "sum_sq"] / ss_total),
        }
    return AnovaResult(factors=out)


def tukey_hsd(values: list[float], groups: list[str], alpha: float = 0.05):
    """Tukey HSD post-hoc after a significant ANOVA."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    res = pairwise_tukeyhsd(np.asarray(values, float), np.asarray(groups), alpha=alpha)
    return res


def holm_bonferroni(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    order = np.argsort(pvals)
    m = len(pvals)
    reject = [False] * m
    for rank, idx in enumerate(order):
        if pvals[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject


def benjamini_hochberg(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    m = len(pvals)
    order = np.argsort(pvals)
    reject = [False] * m
    max_k = -1
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= alpha * rank / m:
            max_k = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= max_k:
            reject[idx] = True
    return reject


def power_two_sample(effect_size: float, n: int, alpha: float = 0.05) -> float:
    """Approximate power of a two-sample t-test for a given Cohen's d and per-group n."""
    from scipy.stats import nct, t

    df = 2 * n - 2
    ncp = effect_size * np.sqrt(n / 2.0)
    crit = t.ppf(1 - alpha / 2, df)
    return float(1 - nct.cdf(crit, df, ncp) + nct.cdf(-crit, df, ncp))


def required_n(effect_size: float, alpha: float = 0.05, target_power: float = 0.8) -> int:
    """Smallest per-group n achieving target power for a two-sample t-test."""
    for n in range(3, 5000):
        if power_two_sample(effect_size, n, alpha) >= target_power:
            return n
    return 5000
