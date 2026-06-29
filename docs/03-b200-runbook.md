# B200 (or any multi-GPU) runbook — one short, fully-utilized run

SENTINEL auto-detects every GPU and fans model runs across all of them (`sentinel run-parallel`).
With 8× B200 this runs all four model families *and* their evolution/ablation jobs at once.

## The pipeline parallelism model
vLLM owns one GPU per process, so SENTINEL parallelizes **one process per GPU**, not
tensor-parallel. `run-parallel` detects `torch.cuda.device_count()` and starts that many
workers; each worker pins `CUDA_VISIBLE_DEVICES` to one GPU and pulls `model × config` jobs
from a queue. 8 GPUs + 8 jobs ⇒ full utilization. Within a process the run is latency-bound
(one probe at a time), so keep `solve.max_tokens` small — that, not the GPU, sets wall-clock.

## Step 0 — install (once)
```bash
bash scripts/setup.sh
export HF_TOKEN=hf_xxx
```

## Step 1 — WARM UP before the clock matters (critical)
Pre-download weights and build the compile cache on every GPU so the real run is pure inference:
```bash
sentinel run-parallel --config conf/config_warmup.yaml --no-evo
```
This loads all four models across the GPUs, runs a trivial 1-probe grid, and exits. Caches
persist (`~/.cache/huggingface`, `~/.cache/vllm`). Do this *outside* your timed window if you can.

## Step 2 — the real run (uses all GPUs)
```bash
sentinel run-parallel
```
Defaults to **8 jobs**: the 4 families on the flagship config (`conf/config_b200.yaml`) +
the 4 families on the evolution/ablation config (`conf/config_b200_evo.yaml`). On 8 GPUs all
run concurrently. It writes:
- `experiments/runs/<model>/` — flagship: ASR curves, ASR-AULC, detection (recall/FPR/per-class F1), security-utility, transfer, figures (color + B&W), `results.json`.
- `experiments/runs_evo/<model>/` — evolution trajectory + ablation, figures, `results.json`.
- `experiments/runs/_logs/<job>.log` — per-job stdout.
- `experiments/aggregate/aggregate_results.json` + `fig_model_condition_asr.*` — **cross-model two-way ANOVA (model × condition)**, Tukey HSD, pooled effect sizes, multiple-comparison correction.

Each phase checkpoints `results.json`, so partial results survive if you run out of time.

## What you get (PhD checklist)
- H1: ASR-AULC curves per model × cross-model ANOVA significance ✓
- Detector quality: recall, **FPR**, per-class precision/recall/F1 ✓
- H2: ablation (−evolution, −meta-defense, −classifier, …) ✓ (evo runs)
- Defensive evolution: trajectory, stability, QD coverage, human-gate log ✓
- H4: security-utility tradeoff (Pareto) ✓
- H3 (cross-threat transfer): leave-one-class-out ✓
- Statistics: bootstrap CIs, two-way ANOVA, Tukey, Cohen's d/Hedges g, Holm/BH ✓

Robustness (multi-turn / context-length / channel poisoning) is **off** in the B200 configs
because it is latency-heavy; run it later on free compute (Kaggle) with
`experiment.run_robustness=true`.

## Tuning for your exact time budget
- Faster: `--no-evo` (flagship only, 4 jobs), or lower `corpus.repeat` / `experiment.seeds`.
- More complete: add `experiment.run_robustness=true` to `conf/config_b200_evo.yaml`.
- Fewer GPUs than jobs is fine — jobs queue automatically.

## Blackwell (B200 / SM100) note
B200 needs CUDA 12.8 + a recent vLLM. If the engine errors on graph capture or a kernel,
add `enforce_eager=true` to the model config (skips `torch.compile`; ~10–20% slower decode but
maximally compatible) and ensure `torch.version.cuda == 12.8` (`scripts/setup.sh` handles this).
