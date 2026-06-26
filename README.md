# SENTINEL

**Self-Evolving Neural Threat Intelligence through Neuroadaptive Evolutionary Learning**

A defensive-only, self-hardening agent-security research framework. SENTINEL treats an
incoming attack as a structured signal: it **detects** the threat, **classifies** its
OWASP type and behavioral signature, **learns** which countermeasure neutralizes it, and
**evolves** its own defensive architecture in response — under a human gate, with
evolution confined to defensive modules. It is built on the FGAE (Failure-Guided
Architecture Evolution) substrate, substituting a *threat* signal for FGAE's
*reasoning-failure* signal.

> Strictly defensive. SENTINEL never generates attacks, probes external systems, or
> develops offensive capability. Adversarial probes come only from published red-team
> corpora (read-only, provenance-hashed). See [docs/01-architecture-and-design.md](docs/01-architecture-and-design.md) §0.

## Why it's defensible (machine-checked safety)

Five invariants are enforced in the type system and tests, not by policy:

| Invariant | Enforcement |
|---|---|
| I1 No attack synthesis | no code path emits attacks; probes are read-only |
| I2 Evolution = defensive only | genome holds only `DefensiveModuleSpec`; closed mutation set |
| I3 Human gate before retention | `PENDING→DEPLOYED` requires an approved `HumanGateDecision` |
| I4 Oversight immutable | objectives/oversight are not mutation targets |
| I5 Probe provenance | every probe carries a SHA-256 source hash |

`pytest tests/test_invariants.py` is the guarantee these hold.

## Architecture (see [docs/01-architecture-and-design.md](docs/01-architecture-and-design.md))

```
Base Agent (FGAE) → Sentinel cascade (Rule→Anomaly→NeuralClassifier→Signature)
   → Threat Graph ↔ Meta-Defense (contextual bandit)
   → Evolution Engine (NSGA-II + MAP-Elites, human-gated)
   → Eval / Metrics / Stats / Viz
```

Key design choices (with rejected alternatives) are documented per component:
hybrid behavioral signatures (6 interpretable axes ⊕ learned residual), LinUCB/Thompson
meta-defense, NSGA-II + MAP-Elites evolution over a constrained multi-objective fitness,
and per-class programmatic ASR oracles with canary tokens.

## Real models only (A100 / vLLM)

No mock backends. Inference is real: in-process `vllm` serving of local quantized weights,
or an OpenAI-compatible endpoint. Model choice is a config switch (Qwen3-14B,
DeepSeek-R1-Distill-14B, Mistral-Small, Qwen3-32B) — no call-site changes.

### Install (GPU host)

```bash
pip install -e ".[dev]"          # pulls torch, vllm, sentence-transformers, ...
```

### Google Colab / Lightning AI

```bash
!bash scripts/colab_setup.sh      # pins a vLLM<->transformers combo that avoids known crashes
%env HF_TOKEN=hf_xxx              # faster / gated downloads
!sentinel run model=qwen3_14b
```

`colab_setup.sh` installs a tested serving stack and prints your GPU's compute capability
(the full grid needs an A100/L4/H100; a T4 works only for a smoke run). If a model fails to
load, the error prints an actionable hint — see [docs/02-troubleshooting.md](docs/02-troubleshooting.md).

### Run the full study

```bash
sentinel run --config conf/config.yaml model=qwen3_14b
sentinel run-all-models          # three families + Qwen3-32B scale check
```

This runs the adversarial grid (5 conditions × seeds), the human-gated evolution loop,
ablations, the robustness suite (multi-turn / context-length / channel poisoning),
cross-threat transfer, the full statistics battery, and writes publication figures +
a reproducibility manifest to `experiments/runs/<model>/`.

**Live progress.** The run shows nested progress bars (via `tqdm.auto`, which renders as real
bars in Colab): a top-level **phase** bar (adversarial grid → evolution → ablation → robustness
→ transfer → utility → statistics → figures), a per-condition×seed bar, and an inner per-probe
bar with running ASR/recall so you can watch the agent harden in real time. First, try the
**smoke run** in [docs/02-troubleshooting.md](docs/02-troubleshooting.md#quick-smoke-run-verify-the-whole-pipeline-end-to-end-cheaply)
to confirm the model loads before committing GPU hours.

> **Figures and results come ONLY from real model runs.** There is no synthetic/CPU path
> that fabricates graphs — `sentinel run` on the A100 measures ASR/recall/etc. from the actual
> models and feeds them through the metric → stats → figure pipeline. Each figure is written as
> PNG (300 DPI) in **two variants** to `experiments/runs/<model>/figures/`: `<name>.png` (color)
> and `<name>_bw.png` (grayscale, print-safe — series stay distinguishable by linestyle + marker).
> Axes autoscale to the recorded values, and every defense condition gets a distinct marker so
> curves that share an ASR value remain separable.

## Logic tests (no GPU, no fabricated data)

The unit suite verifies the *logic* of the GPU-independent components (safety invariants,
NSGA-II dominance, bandit updates, threat-graph queries, metric math, statistics) using
synthetic arrays as test fixtures only — it never produces figures or paper artifacts:

```bash
pip install numpy scipy scikit-learn matplotlib networkx pydantic \
            structlog omegaconf statsmodels pandas pytest
pytest tests/                       # safety invariants + component logic
```

## Metrics (four research dimensions)

Security effectiveness (ASR, **ASR-AULC** flagship, time-to-hardening, recall, FPR, F1),
threat behavior (migration KL + matrix, recurrence, signature drift, Shannon diversity,
novelty), defense evolution (security fitness, efficiency, stability, convergence,
module utility, reuse, cross-threat transfer), and utility preservation
(security-utility tradeoff, clean-task accuracy).

## Statistics

Bootstrap CIs (10k), two-way ANOVA (model × condition), Tukey HSD, Cohen's d / Hedges g /
η², Holm–Bonferroni + Benjamini–Hochberg, power analysis (≥0.8) with sample-size derivation.

## Layout

```
docs/        architecture & design rationale
conf/        Hydra configs (model + experiment)
src/sentinel/
  core/ models/ substrate/ sentinel_layer/ graph/ meta_defense/
  defenses/ evolution/ corpora/ eval/ metrics/ stats/ viz/
tests/       GPU-independent test suite (incl. safety invariants)
experiments/ runner + figure validation harness
```

## License

Apache-2.0. Defensive-security research artifact.
