#!/usr/bin/env bash
# SENTINEL — A100 / CUDA 12.8 setup (driver 570+).
#
# Resolves the vLLM <-> FlashInfer <-> torch <-> transformers dependency tangle that
# causes `cudaErrorInsufficientDriver (35)` in FlashInfer's sampler kernel.
#
# Strategy: install a mutually-consistent CUDA-12.8 stack and REMOVE FlashInfer entirely.
# vLLM runs fine on an A100 without it (FlashAttention for attention + native PyTorch
# sampler). SENTINEL also disables the FlashInfer sampler in code, so this is belt-and-braces.
#
# Run:  bash scripts/setup_a100_cuda128.sh
set -euo pipefail

echo "==> Verifying driver / CUDA…"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# 1) Remove any mismatched FlashInfer wheels (the usual culprit: a cu13 build).
echo "==> Removing FlashInfer (not needed; avoids CUDA-runtime/driver mismatch)…"
pip uninstall -y flashinfer flashinfer-python flashinfer-cubin flashinfer-jit-cache 2>/dev/null || true

# 2) Pin a consistent CUDA-12.8 serving stack.
echo "==> Installing torch (cu128) + vLLM 0.10.0 + transformers 4.53.3…"
pip install -q torch==2.7.* --index-url https://download.pytorch.org/whl/cu128
pip install -q "vllm==0.10.0" "transformers==4.53.3" "tokenizers>=0.19,<0.20" accelerate

# 3) SENTINEL framework deps (no sentence-transformers / torchcodec).
echo "==> Installing SENTINEL analysis deps…"
pip install -q "pydantic>=2.6" numpy scipy scikit-learn statsmodels pandas \
  networkx matplotlib structlog omegaconf hydra-core rich tqdm hnswlib

# 4) Install SENTINEL itself without pulling its deps (keep the pins above).
pip install -q -e . --no-deps 2>/dev/null || pip install -q -e /teamspace/studios/this_studio/SENTINEL --no-deps 2>/dev/null || true

echo "==> Sanity check:"
python - <<'PY'
import torch
print("torch:", torch.__version__, "| torch.cuda:", torch.version.cuda,
      "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    print("GPU:", torch.cuda.get_device_name(0), "| compute capability:", f"{cc[0]}.{cc[1]}")
try:
    import flashinfer  # noqa: F401
    print("WARNING: flashinfer still importable — SENTINEL disables its sampler, but you can "
          "`pip uninstall -y flashinfer flashinfer-python` to be fully safe.")
except Exception:
    print("flashinfer: not installed (good — vLLM uses FlashAttention + native sampler).")
PY

cat <<'EOF'

==> Done. SENTINEL disables the FlashInfer sampler automatically (VLLM_USE_FLASHINFER_SAMPLER=0).

Run the study:
    export HF_TOKEN=hf_xxx                 # faster / gated downloads
    sentinel run model=qwen3_14b
    sentinel run-all-models

Optional (only if you specifically want FlashInfer and your CUDA matches the driver):
    pip install flashinfer-python flashinfer-cubin
    pip install flashinfer-jit-cache --extra-index-url https://flashinfer.ai/whl/cu128
    sentinel run model=qwen3_14b model.use_flashinfer=true
EOF
