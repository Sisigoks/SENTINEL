"""Statistics-engine tests (bootstrap, effect size, corrections, power, ANOVA)."""

from __future__ import annotations

import numpy as np

from sentinel.stats import tests as st


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(0.3, 0.05, size=50).tolist()
    r = st.bootstrap_ci(data, n_boot=2000)
    assert r.ci_low < r.mean < r.ci_high


def test_cohens_d_sign_and_magnitude():
    a = [0.6, 0.62, 0.58, 0.61]
    b = [0.2, 0.18, 0.22, 0.19]
    d = st.cohens_d(a, b)
    assert d > 2.0  # large effect
    g = st.hedges_g(a, b)
    assert abs(g) <= abs(d)


def test_holm_and_bh():
    pvals = [0.001, 0.04, 0.20, 0.5]
    holm = st.holm_bonferroni(pvals)
    bh = st.benjamini_hochberg(pvals)
    assert holm[0] and not holm[-1]
    assert bh[0] and not bh[-1]


def test_power_and_required_n():
    p = st.power_two_sample(0.8, 30)
    assert 0.0 <= p <= 1.0
    n = st.required_n(0.8)
    assert 5 <= n <= 60


def test_two_way_anova_detects_condition_effect():
    rng = np.random.default_rng(1)
    values, fa, fb = [], [], []
    for model in ["m1", "m2", "m3"]:
        for cond, mean in [("vanilla", 0.6), ("sentinel", 0.15)]:
            for _ in range(8):
                values.append(float(rng.normal(mean, 0.03)))
                fa.append(model)
                fb.append(cond)
    res = st.two_way_anova(values, fa, fb)
    # condition factor should be highly significant
    cond_term = [k for k in res.factors if "B" in k and ":" not in k][0]
    assert res.factors[cond_term]["p"] < 0.001
