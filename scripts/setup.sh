#!/usr/bin/env bash
# SENTINEL — universal setup. One script for every GPU (T4 / L4 / A100 / H100 …) and
# every environment (Colab, Lightning AI, Kaggle, bare cloud box).
#
# Run from the repo root:   bash scripts/setup.sh
#
# Design for uniformity & robustness:
#   * Let vLLM pull the torch / tokenizers it was built against — no per-CUDA index-URL
#     guessing, so the same script works regardless of the box's CUDA minor version.
#   * Remove FlashInfer entirely (the usual cause of `cudaErrorInsufficientDriver (35)`).
#     vLLM runs without it (FlashAttention on Ampere+, XFormers on older GPUs, native
#     PyTorch sampler everywhere). SENTINEL also disables the FlashInfer sampler in code.
#   * Pin only the two things that actually matter for compatibility: vLLM 0.10.0 and
#     transformers 4.53.3 (avoids the `all_special_tokens_extended` tokenizer skew).
#   * No sentence-transformers (pulls torchcodec -> needs FFmpeg, breaks on minimal images).
set -euo pipefail

# ----- progress helpers -------------------------------------------------------
TOTAL_STEPS=6
BAR_W=30
_progress() {  # _progress <current> <total> <label>
  local cur=$1 tot=$2 label=$3
  local filled=$(( cur * BAR_W / tot )) i bar=""
  for ((i=0; i<BAR_W; i++)); do [ $i -lt $filled ] && bar+="█" || bar+="░"; done
  printf "\r\033[1m[%d/%d]\033[0m [%s] %3d%%  %-28s" "$cur" "$tot" "$bar" $(( cur*100/tot )) "$label"
}
step() {  # step <n> <message>; prints a banner + overall progress bar
  echo; echo "════════════════════════════════════════════════════════════════"
  _progress "$1" "$TOTAL_STEPS" "$2"; echo; echo "────────────────────────────────────────────────────────────────"
}
START=$SECONDS

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SENTINEL setup — universal (any GPU / any lab)               ║"
echo "╚══════════════════════════════════════════════════════════════╝"

step 1 "Detecting GPU / driver"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
  echo "   (nvidia-smi not found — CPU-only box; vLLM needs a CUDA GPU to serve models)"
fi

step 2 "Removing mismatched FlashInfer wheels"
# (cu13 build on a cu12 box etc. — the usual cudaErrorInsufficientDriver culprit)
pip uninstall -y flashinfer flashinfer-python flashinfer-cubin flashinfer-jit-cache 2>/dev/null || true

step 3 "Installing vLLM 0.10.0 + transformers 4.53.3  (downloads show their own bars)"
# pip's per-wheel download progress bars are left ON (no -q) so you can watch it. vLLM
# resolves a compatible torch + tokenizers. Override torch only via SENTINEL_TORCH_INDEX.
if [ -n "${SENTINEL_TORCH_INDEX:-}" ]; then
  echo "   (using custom torch index: ${SENTINEL_TORCH_INDEX})"
  pip install --progress-bar on torch --index-url "${SENTINEL_TORCH_INDEX}"
fi
pip install --progress-bar on "vllm==0.10.0" "transformers==4.53.3" accelerate

step 4 "Installing SENTINEL framework deps (analysis / stats / figures)"
pip install --progress-bar on "pydantic>=2.6" numpy scipy scikit-learn statsmodels pandas \
  networkx matplotlib structlog omegaconf hydra-core rich tqdm hnswlib

step 5 "Installing SENTINEL (editable, --no-deps)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pip install -e "${REPO_ROOT}" --no-deps || pip install -e . --no-deps || true

step 6 "Sanity check"
python - <<'PY'
import torch
print("torch:", torch.__version__, "| torch.cuda:", torch.version.cuda,
      "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    print("GPU:", torch.cuda.get_device_name(0), "| compute capability:", f"{cc[0]}.{cc[1]}")
    if cc[0] < 8:
        print("NOTE: pre-Ampere GPU (CC<8.0, e.g. T4). vLLM uses the V0 engine; the full grid "
              "is slow — fine for a smoke run, use an A100/L4/H100 for the real study.")
try:
    import flashinfer  # noqa: F401
    print("NOTE: flashinfer still importable — SENTINEL disables its sampler, but you may "
          "`pip uninstall -y flashinfer flashinfer-python` to be fully safe.")
except Exception:
    print("flashinfer: not installed (good — vLLM uses FlashAttention/XFormers + native sampler).")
PY

_progress "$TOTAL_STEPS" "$TOTAL_STEPS" "complete"; echo
echo "   setup finished in $(( SECONDS - START ))s"

cat <<'EOF'

==> Done. FlashInfer sampler is disabled automatically (VLLM_USE_FLASHINFER_SAMPLER=0).

Set a token for faster/gated downloads, then run:
    export HF_TOKEN=hf_xxx
    sentinel run model=llama3_1_8b        # one model (8B — cheapest)
    sentinel run-all-models               # six-model roster (5 lineages + 8B->70B scale axis)

Cheap end-to-end smoke test first (any GPU):
    sentinel run model=llama3_1_8b max_model_len=4096 'experiment.seeds=[0]' corpus.repeat=2 \
      experiment.run_robustness=false experiment.run_evolution=false experiment.run_ablation=false

Troubleshooting: docs/02-troubleshooting.md
EOF
