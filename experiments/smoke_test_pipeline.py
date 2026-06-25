"""SMOKE TEST ONLY — NOT A SOURCE OF RESULTS.

================================================================================
 THE NUMBERS IN THIS FILE ARE FAKE. They are hand-written placeholder inputs used
 ONLY to check that the plotting / evolution / statistics CODE runs without a GPU.
 Nothing produced by this script may appear in the paper.

 Real, paper-grade figures come exclusively from `sentinel run ... model=qwen3_14b`
 on the A100, which feeds *measured* ASR/recall/etc. through the same functions.
================================================================================

It exercises every figure function, the NSGA-II/MAP-Elites loop, and the stats
battery end-to-end so you can confirm the analysis pipeline is wired correctly
before spending GPU hours.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel import viz  # noqa: E402
from sentinel.core.types import ThreatClass  # noqa: E402
from sentinel.evolution.engine import EvolutionEngine  # noqa: E402
from sentinel.evolution.fitness import FitnessVector  # noqa: E402
from sentinel.evolution.genome import random_seed_genome  # noqa: E402
from sentinel.evolution.human_gate import PolicyHumanGate  # noqa: E402
from sentinel.metrics.catalog import attack_migration_matrix  # noqa: E402
from sentinel.stats import tests as st  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "experiments" / "figures_validation"


def make_asr_curves() -> dict[str, list[float]]:
    rng = np.random.default_rng(0)
    n = 12
    # monotone-decreasing learning curves with the expected ordering (paper §8)
    bases = {
        "vanilla": 0.58, "static_filter": 0.42, "reflection_defense": 0.34,
        "meta_defense": 0.24, "full_sentinel": 0.17,
    }
    decay = {"vanilla": 0.0, "static_filter": 0.02, "reflection_defense": 0.03,
             "meta_defense": 0.05, "full_sentinel": 0.08}
    curves = {}
    for c, b in bases.items():
        y = [max(0.03, b - decay[c] * i + rng.normal(0, 0.01)) for i in range(n)]
        curves[c] = y
    return curves


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. ASR learning curves (the page-1 figure) with bootstrap CI bands
    curves = make_asr_curves()
    ci = {}
    for c, y in curves.items():
        bands_lo, bands_hi = [], []
        for v in y:
            r = st.bootstrap_ci([max(0, v + d) for d in np.random.default_rng(1).normal(0, 0.03, 8)], n_boot=1000)
            bands_lo.append(r.ci_low); bands_hi.append(r.ci_high)
        ci[c] = (bands_lo, bands_hi)
    p1 = viz.plot_asr_curves(curves, OUT, ci=ci)

    # 2. attack migration heatmap
    dists = [
        {ThreatClass.PROMPT_INJECTION.value: 30, ThreatClass.TOOL_MISUSE.value: 5},
        {ThreatClass.PROMPT_INJECTION.value: 10, ThreatClass.TOOL_MISUSE.value: 20,
         ThreatClass.GOAL_HIJACK.value: 8},
        {ThreatClass.TOOL_MISUSE.value: 8, ThreatClass.GOAL_HIJACK.value: 18,
         ThreatClass.PRIVILEGE_ABUSE.value: 10},
    ]
    M = attack_migration_matrix(dists)
    viz.plot_migration_heatmap(M, OUT)

    # 3. defense evolution + stability (real NSGA-II/MAP-Elites loop)
    def evaluator(genome) -> FitnessVector:
        n = len(genome.modules)
        return FitnessVector(asr=max(0.05, 0.6 - 0.07 * n), recall=min(0.99, 0.65 + 0.04 * n),
                             precision=0.9, fpr=0.02, utility_drop=min(0.04, 0.004 * n),
                             latency_s=1.0, token_cost=200.0)

    gate = PolicyHumanGate()
    engine = EvolutionEngine(evaluator, gate, population_size=10, offspring_per_gen=6)
    rng = np.random.default_rng(0)
    engine.initialize([random_seed_genome(rng, n=k) for k in (1, 1, 2, 2, 3)])
    for _ in range(15):
        engine.step()
        if engine.converged():
            break
    viz.plot_defense_evolution(engine.history, OUT)
    viz.plot_stability(engine.history, OUT)

    # 4. recurrence curves
    rng = np.random.default_rng(2)
    recurrence = {c.value: sorted(rng.integers(0, 120, size=rng.integers(5, 30)).tolist())
                  for c in list(ThreatClass)[:6]}
    viz.plot_recurrence(recurrence, OUT)

    # 5. security-utility pareto
    points = {"vanilla": (0.95, 0.42), "static_filter": (0.93, 0.58),
              "reflection_defense": (0.92, 0.66), "meta_defense": (0.91, 0.76),
              "full_sentinel": (0.90, 0.83)}
    viz.plot_security_utility_pareto(points, OUT)

    # 6. cross-threat transfer matrix
    labels = [c.value for c in list(ThreatClass)[:5]]
    tm = np.clip(np.random.default_rng(3).normal(0.25, 0.1, (5, 5)), 0, 1)
    np.fill_diagonal(tm, 0.12)
    viz.plot_cross_transfer_matrix(tm, labels, labels, OUT, title="Cross-threat transfer (ASR)")

    # 7. cross-model transfer matrix
    models = ["Qwen3-14B", "DeepSeek-14B", "Mistral-S", "Qwen3-32B"]
    cm = np.clip(np.random.default_rng(4).normal(0.18, 0.05, (4, 4)), 0, 1)
    viz.plot_cross_transfer_matrix(cm, models, models, OUT, name="fig_model_transfer",
                                   title="Cross-model transfer (ASR)")

    # 8. signature drift
    drift = {c.value: list(np.cumsum(np.abs(np.random.default_rng(5).normal(0, 0.05, 20))))
             for c in list(ThreatClass)[:5]}
    viz.plot_signature_drift(drift, OUT)

    # 9. ablation
    ablation = {"full_sentinel": 0.17, "-evolution": 0.22, "-meta_defense": 0.28,
                "-threat_graph": 0.31, "-signature": 0.35, "-classifier": 0.44,
                "rule_only": 0.49, "neural_only": 0.30}
    viz.plot_ablation(ablation, OUT)

    # statistics summary written alongside
    final_asr = {"vanilla": [0.58, 0.60, 0.57], "full_sentinel": [0.17, 0.15, 0.19]}
    d = st.cohens_d(final_asr["vanilla"], final_asr["full_sentinel"])
    anova_vals, fa, fb = [], [], []
    for m in models[:3]:
        for c, mean in [("vanilla", 0.58), ("full_sentinel", 0.17)]:
            for _ in range(8):
                anova_vals.append(float(np.random.default_rng(6).normal(mean, 0.03)))
                fa.append(m); fb.append(c)
    res = st.two_way_anova(anova_vals, fa, fb)

    print(f"[ok] page-1 figure: {p1}")
    print(f"[ok] figures written to: {OUT}")
    print(f"[ok] Cohen's d (vanilla vs full_sentinel): {d:.2f}")
    print(f"[ok] evolution generations: {len(engine.history)}, converged={engine.converged()}, "
          f"QD coverage={engine.archive.coverage():.2f}, gate decisions={len(gate.log)}")
    print(f"[ok] two-way ANOVA factors: {list(res.factors)}")
    n_pdf = len(list(OUT.glob('*.pdf')))
    print(f"[ok] total figure PDFs: {n_pdf}")


if __name__ == "__main__":
    main()
