"""Publication-grade statistics: bootstrap, ANOVA, post-hoc, effect size, corrections."""

from .tests import (
    benjamini_hochberg,
    bootstrap_ci,
    cohens_d,
    eta_squared,
    hedges_g,
    holm_bonferroni,
    power_two_sample,
    tukey_hsd,
    two_way_anova,
)

__all__ = [
    "bootstrap_ci",
    "two_way_anova",
    "tukey_hsd",
    "cohens_d",
    "hedges_g",
    "eta_squared",
    "holm_bonferroni",
    "benjamini_hochberg",
    "power_two_sample",
]
