#!/usr/bin/env bash
# SENTINEL — Google Colab / Lightning AI setup.
# Run once at the top of your notebook:  !bash scripts/colab_setup.sh
#
# Installs a vLLM <-> transformers combination known to avoid the
# `all_special_tokens_extended` tokenizer crash (see docs/02-troubleshooting.md).
set -euo pipefail

echo "==> Installing SENTINEL dependencies (this can take a few minutes)…"

# Pin a compatible serving stack. vLLM 0.10.x pairs with transformers 4.53.x.
pip install -q \
  "vllm==0.10.0" \
  "transformers==4.53.3" \
  "tokenizers>=0.19,<0.20" \
  "sentence-transformers>=2.6" \
  "hnswlib>=0.8"

# SENTINEL's analysis / framework deps.
pip install -q \
  "pydantic>=2.6" numpy scipy scikit-learn statsmodels pandas \
  networkx matplotlib structlog omegaconf hydra-core rich tqdm

# Install SENTINEL itself (editable, no deps so the pins above are kept).
pip install -q -e . --no-deps || pip install -q -e /content/SENTINEL --no-deps || true

echo "==> Done. Verify the GPU:"
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"GPU: {name}  compute capability: {cc[0]}.{cc[1]}")
    if cc[0] < 8:
        print("NOTE: pre-Ampere GPU (CC<8.0). vLLM uses the V0 engine; the full grid is slow. "
              "Use an A100/L4/H100 for the real run (see docs/02-troubleshooting.md).")
PY

echo "==> Set your HF token for faster/gated downloads:  export HF_TOKEN=hf_xxx"
echo "==> Then run:  sentinel run model=qwen3_14b"
