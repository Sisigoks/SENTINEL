# Troubleshooting — model & serving errors

SENTINEL runs real models via vLLM, so most failures are environment/version issues, not
SENTINEL bugs. The vLLM backend ([src/sentinel/models/backend.py](../src/sentinel/models/backend.py))
catches engine-init failures and prints an actionable hint; this page is the full reference.
Known version skews are auto-patched in [src/sentinel/models/compat.py](../src/sentinel/models/compat.py),
but pinning the dependency set is the clean fix.

---

## 1. `Qwen2Tokenizer has no attribute all_special_tokens_extended`

```
AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
  ... vllm/transformers_utils/tokenizer.py ... get_cached_tokenizer
```

**Cause.** A transformers <-> vLLM version skew. vLLM's tokenizer cache reads
`all_special_tokens_extended`; some newer transformers builds dropped that property from
the *slow* tokenizer classes (e.g. `Qwen2Tokenizer`). Seen on Colab where transformers is
newer than the vLLM build expects.

**Fixes (either one):**

1. **Automatic shim (already applied).** `apply_model_compat_patches()` restores the property
   before the engine loads. If you see a `compat: patched missing tokenizer...` warning, the
   shim handled it and the run continues.
2. **Clean pin (recommended for the paper run).** Match transformers to your vLLM build:
   ```bash
   pip install "transformers==4.53.3"     # compatible with vLLM 0.10.x
   ```
   If unsure of the exact pin, reinstall vLLM so it pulls its tested transformers:
   ```bash
   pip install --force-reinstall "vllm==0.10.0"
   ```

---

## 2. `Compute Capability < 8.0 is not supported by the V1 Engine. Falling back to V0.`

**Not fatal** — vLLM automatically uses the V0 engine. It means your GPU is **pre-Ampere**
(e.g. Colab **T4 = CC 7.5**, V100 = CC 7.0), not an A100 (CC 8.0). Consequences:

- AWQ needs CC ≥ 7.5, so T4 works but is **slow**; V100 (7.0) cannot run AWQ.
- The full 5×3×10 grid is impractical on a T4. Use it only for a smoke run; do the real
  grid on an **A100/L4/H100** (Lightning AI / Colab Pro A100).

To sanity-check on a small GPU, shrink the run:
```bash
sentinel run model=qwen3_14b max_model_len=4096 \
  'experiment.seeds=[0]' corpus.repeat=2 experiment.run_evolution=false
```

---

## 2b. `cudaErrorInsufficientDriver (35)` in FlashInfer (the big one on cloud A100s)

```
RuntimeError: CUDA Runtime Error: cudaErrorInsufficientDriver (35):
CUDA driver version is insufficient for CUDA runtime version
  ... flashinfer/sampling.py ... top_k_mask_logits()
```

**Cause.** The model loads fine; the crash is in **FlashInfer's sampler kernel**. A FlashInfer
wheel compiled for a *newer* CUDA runtime than your driver/torch (commonly a CUDA-13 build on a
CUDA-12.8 box) reports the driver as "insufficient" even when `nvidia-smi` shows a perfectly new
driver (e.g. 570.x / CUDA 12.8 on an A100). **It is not your GPU, driver, vLLM, AWQ, or transformers.**

**Fix (already applied in code).** SENTINEL sets `VLLM_USE_FLASHINFER_SAMPLER=0` before vLLM
imports, so vLLM uses its native PyTorch sampler and never calls FlashInfer. SENTINEL decodes at
temperature 0 (greedy), so this costs nothing. To re-enable FlashInfer once your wheel matches:
```bash
sentinel run model=qwen3_14b model.use_flashinfer=true
```

**Clean dependency resolution (recommended on the A100).** Remove the mismatched FlashInfer and
install a consistent CUDA-12.8 stack:
```bash
bash scripts/setup_a100_cuda128.sh
```
This uninstalls all `flashinfer*` wheels, installs torch cu128 + vLLM 0.10.0 + transformers 4.53.3,
and verifies `torch.version.cuda`. vLLM runs on the A100 with FlashAttention + the native sampler —
FlashInfer is optional and not required.

Diagnostics to confirm the mismatch:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # expect 2.7.x / 12.8
pip show flashinfer-python 2>/dev/null | grep -i version || echo "flashinfer not installed (good)"
```

---

## 3. GPU out-of-memory (OOM) / KV-cache errors

```
ValueError: No available memory for the cache blocks ...
torch.OutOfMemoryError: CUDA out of memory
```

Lower memory pressure in the model config or via overrides:
```bash
sentinel run model=qwen3_14b max_model_len=4096 model.gpu_memory_utilization=0.80
```
- Drop `max_model_len` (8192 → 4096 → 2048).
- Drop `gpu_memory_utilization` (0.90 → 0.80).
- Use an AWQ/quantized repo, or a smaller model.

---

## 4. Quantization mismatch

```
ValueError: ... is not quantized with awq ...
```

**Cause.** Forcing `quantization: awq` on a non-AWQ repo. SENTINEL now defaults
`quantization: null` so vLLM **auto-detects** from `config.json` — keep it null unless you
deliberately point at an AWQ/GPTQ repo.

---

## 5. Mistral models fail to load the tokenizer

Some Mistral repos ship only Mistral's native tokenizer. Set in the model config:
```yaml
tokenizer_mode: mistral
```
The provided [conf/model/mistral_small.yaml](../conf/model/mistral_small.yaml) uses the
HF-format repo (`tokenizer_mode: auto`), which avoids this.

---

## 6. Hugging Face rate limits / gated models

```
Warning: You are sending unauthenticated requests to the HF Hub ...
```
Set a token before running (also speeds downloads):
```bash
export HF_TOKEN=hf_xxx        # Colab: import os; os.environ["HF_TOKEN"]="hf_xxx"
```
For gated repos you must also accept the license on the model page.

---

## 7. `Could not load libtorchcodec` / `libavutil.so... cannot open shared object file`

```
RuntimeError: Could not load libtorchcodec ...
OSError: libavutil.so.60: cannot open shared object file: No such file or directory
  ... from sentence_transformers import SentenceTransformer
```

**Cause.** Recent `sentence-transformers` imports `torchcodec` (audio/video modality), which
needs FFmpeg native libs that Colab/minimal images lack. **SENTINEL no longer uses
sentence-transformers** — the encoder ([src/sentinel/models/encoder.py](../src/sentinel/models/encoder.py))
computes embeddings directly via `transformers`. If you hit this, you have an old build:
```bash
pip uninstall -y sentence-transformers torchcodec   # not needed by SENTINEL
```
(Re-run `scripts/colab_setup.sh`, which no longer installs sentence-transformers.)

---

## 8. Encoder CUDA OOM when sharing the GPU with the served LLM

The vLLM engine reserves `gpu_memory_utilization` of VRAM (default 0.90) *first*; the frozen
encoder then loads onto the same GPU. On small GPUs (e.g. 14.5 GB T4) little is left.
SENTINEL loads the encoder in **fp16** and **auto-falls back to CPU** if the GPU load fails,
so it won't crash. To keep it on GPU, leave headroom:
```bash
sentinel run model=qwen3_14b model.gpu_memory_utilization=0.70
```
Or force CPU encoding (the encoder is small; fine for the probe set):
```bash
sentinel run model=qwen3_14b encoder.device=cpu
```

---

## 9. `Requesting 3-fold cross-validation but provided less than 3 examples`

```
ValueError: Requesting 3-fold cross-validation but provided less than 3 examples
            for at least one class.
  ... CalibratedClassifierCV ... classifier.py
```

**Cause.** A tiny corpus (e.g. `corpus.repeat=2`) gives fewer samples per class than the
calibration folds. **Fixed** — the classifier now adapts the fold count to the smallest
class and falls back to an *uncalibrated* head when there are too few samples, so both the
smoke run and the full grid work. (Also removed `multi_class="multinomial"`, which sklearn
≥1.7 dropped.) No action needed; for meaningful calibration use the full corpus (`repeat=60`).

Related: with a **single seed** (`experiment.seeds=[0]`) effect sizes / significance tests are
undefined and are skipped with a note — use ≥2 seeds (the default is `[0,1,2]`) for statistics.

---

## 10. `trust_remote_code` required

A few repos need custom modeling code. Set per-model:
```yaml
trust_remote_code: true
```
Only enable this for repos you trust.

---

## Quick smoke run (verify the whole pipeline end-to-end, cheaply)

```bash
sentinel run model=qwen3_14b \
  max_model_len=4096 \
  'experiment.seeds=[0]' \
  corpus.repeat=2 \
  experiment.run_robustness=false \
  experiment.run_evolution=false \
  experiment.run_ablation=false
```
This loads the real model, runs one seed of the adversarial grid + transfer + stats, and
writes PNG figures to `experiments/runs/<model>/figures/`. Scale up once it's green.
