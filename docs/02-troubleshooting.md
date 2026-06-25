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

## 7. `trust_remote_code` required

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
