# Multi-GPU runbook — one config, every GPU, fully utilized

SENTINEL adapts to whatever it runs on. There is **one config** (`conf/config.yaml`); it
auto-tunes to the GPU and auto-fans across all GPUs.

## How "use the GPU to the max" works
1. **Batched inference.** The LLM solve for many probes is submitted to vLLM in one batched
   call (`solve.batch_size`), so a single GPU is saturated instead of running one probe at a
   time. `batch_size: auto` scales it to VRAM: T4→8, L4→24, A100-40→48, A100/H100-80→96,
   B200/H200→192. ([models/autotune.py](../src/sentinel/models/autotune.py))
2. **Auto-tuned serving.** `max_model_len` and `gpu_memory_utilization` resolve from detected
   VRAM (the `auto` fields in the model configs).
3. **Auto multi-GPU.** `sentinel run-parallel` detects `torch.cuda.device_count()` and shards
   the run across every GPU (one vLLM process per GPU).

## Install (once)
```bash
bash scripts/setup.sh
export HF_TOKEN=hf_xxx
```

## Single GPU — saturates that GPU
```bash
sentinel run                      # full study on one GPU, auto batch size
```

## All GPUs — auto-detect and fan out
```bash
sentinel run-parallel             # detects N GPUs, shards across all of them, then aggregates
```
With the six-model roster and seeds `[0,1,2]`:
- **12 GPUs** → `replicas = 12//6 = 2` → each model split into 2 seed-shards → **12 jobs, all GPUs busy**.
- **6 GPUs** → one model per GPU.
- **2 GPUs** → 4 jobs, 2 at a time (queued).

Each model's first shard also runs evolution + ablation; later shards are grid-only. Shards
write to `experiments/runs/_shard<k>/` and are **merged by model at aggregation** (each
`results.json` carries the model name), so you still get per-model multi-seed stats and the
cross-model two-way ANOVA. Output:
- `experiments/runs/_shard*/<model>/` — ASR curves, ASR-AULC, detection (recall/**FPR**/per-class F1), evolution, ablation, security-utility, transfer, figures (color + B&W), `results.json` (per-phase checkpointed).
- `experiments/runs/_logs/<job>.log` — per-job stdout.
- `experiments/aggregate/` — **two-way ANOVA (model × condition)**, Tukey HSD, pooled effect sizes, multiple-comparison correction, model×condition ASR heatmap.

## Warm up first (recommended for a short rented window)
Pre-download weights + build the vLLM compile cache so the timed run is pure inference:
```bash
sentinel run-parallel --gpus 8 --no-aggregate \
  --config conf/config.yaml
# ...or a trivial single pass per model to populate caches, then the real run-parallel.
```
A cheaper warm pass overrides the one config:
```bash
sentinel run 'experiment.conditions=[vanilla]' 'experiment.seeds=[0]' corpus.repeat=1 \
  solve.max_tokens=8 experiment.run_evolution=false experiment.run_ablation=false \
  experiment.run_robustness=false
```
Caches persist (`~/.cache/huggingface`, `~/.cache/vllm`); the real run then skips download + compile.

## Trim for time / free tier (still one config, just overrides)
```bash
sentinel run 'experiment.seeds=[0,1]' corpus.repeat=5 experiment.run_robustness=false
```

## Blackwell (B200 / SM100)
Needs CUDA 12.8 + recent vLLM (setup.sh installs torch cu128). If graph capture errors on a
kernel, add `enforce_eager=true` to the model config (skips `torch.compile`; ~10-20% slower
decode, maximally compatible).
