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
or an OpenAI-compatible endpoint. Model choice is a config switch — no call-site changes.

The six-model roster is designed as a factorial study, not a grab-bag:

| Config | Model | Size | Axis it contributes |
|---|---|---|---|
| `llama3_1_8b` | Llama-3.1-8B-Instruct (AWQ) | 8B | low-scale anchor; Meta lineage |
| `phi4_14b` | Phi-4 | 14B | instruct-vs-reasoning contrast at matched scale; Microsoft lineage |
| `deepseek_r1_distill_14b` | DeepSeek-R1-Distill-Qwen-14B | 14B | reasoning-distilled, scale-matched to Phi-4 |
| `mistral_small_24b` | Mistral-Small-24B-Instruct-2501 | 24B | Mistral alignment lineage |
| `qwen3_32b` | Qwen3-32B-AWQ | 32B | hybrid-reasoning flagship; Alibaba lineage |
| `llama3_3_70b` | Llama-3.3-70B-Instruct (AWQ) | 70B | within-family scale axis vs the 8B; frontier check |

Five vendor lineages test the model-agnostic claim; the Llama 8B↔70B pair isolates scale
from family; the Phi-4 ↔ R1-distill pair isolates reasoning post-training from scale.
Llama repos are gated (accept the Meta license, set `HF_TOKEN`); Phi-4 (MIT) and
Mistral (Apache-2.0) are open.

### Install (one script, every GPU and lab)

A single setup script works on any CUDA GPU (T4 / L4 / A100 / H100) and any environment
(Colab, Lightning AI, Kaggle, bare cloud box). It lets vLLM resolve a matching torch +
tokenizers, pins vLLM 0.10.0 + transformers 4.53.3, removes mismatched FlashInfer wheels
(the usual cause of `cudaErrorInsufficientDriver`), and prints your GPU's compute capability.

```bash
bash scripts/setup.sh
export HF_TOKEN=hf_xxx            # faster / gated downloads (Colab: %env HF_TOKEN=hf_xxx)
sentinel run model=llama3_1_8b
```

(On Colab/Lightning prefix with `!`/`%env`.) The full grid wants an A100/L4/H100; a T4 works
only for a smoke run. If a model fails to load, the error prints an actionable hint — see
[docs/02-troubleshooting.md](docs/02-troubleshooting.md).

> SENTINEL disables the FlashInfer sampler by default (`VLLM_USE_FLASHINFER_SAMPLER=0`) so a
> mismatched FlashInfer wheel can't crash the run — vLLM uses FlashAttention + the native
> sampler. Re-enable with `model.use_flashinfer=true` once your wheel matches the driver.

### Run the full study (one adaptive config, any GPU)

There is a single config — `conf/config.yaml`. It **auto-tunes to the GPU** (batch size, serving
params scale from T4 to B200) and **auto-detects GPU count**:

```bash
sentinel run                     # one GPU, saturated via batched inference
sentinel run-parallel            # detects ALL GPUs, shards across them, then aggregates
sentinel aggregate               # cross-model two-way ANOVA + figures from completed runs
```

`run-parallel` shards the six-model roster (and their seeds) to fill every GPU; see
[docs/03-multi-gpu-runbook.md](docs/03-multi-gpu-runbook.md). Trim for free tier with overrides:
`sentinel run 'experiment.seeds=[0,1]' corpus.repeat=5 experiment.run_robustness=false`.

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
experiments/ run outputs (runs/, aggregate/ — created at runtime)
```

## License

Apache-2.0. Defensive-security research artifact.
