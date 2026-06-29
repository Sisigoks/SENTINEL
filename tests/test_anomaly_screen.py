"""Regression test for the anomaly screen calibration (the FPR=1.0 bug).

Using IsolationForest.score_samples (offset so inliers are ~-0.5) flagged everything as
novel -> FPR=1.0. decision_function (>0 normal, <0 anomaly) fixes it. This guards that.
"""

from __future__ import annotations

import numpy as np

from sentinel.sentinel_layer.anomaly_screen import AnomalyScreen


def test_anomaly_screen_not_overflagging():
    rng = np.random.default_rng(0)
    benign = rng.normal(0, 1, size=(200, 16))
    a = AnomalyScreen()
    a.fit(benign)
    in_dist_flagged = sum(a.screen(rng.normal(0, 1, 16)).flagged for _ in range(200))
    ood_flagged = sum(a.screen(rng.normal(6, 1, 16)).flagged for _ in range(50))
    assert in_dist_flagged / 200 < 0.25, "in-distribution false-positive rate too high"
    assert ood_flagged / 50 > 0.8, "out-of-distribution attacks missed"
