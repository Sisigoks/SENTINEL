"""Tests for the model-compat shim and vLLM error diagnostics (no GPU / no transformers)."""

from __future__ import annotations

from sentinel.models.backend import VLLMBackend
from sentinel.models.compat import apply_model_compat_patches


def test_compat_patch_is_safe_without_transformers():
    # transformers is not installed in CI; the shim must no-op, not crash, and be idempotent.
    apply_model_compat_patches()
    apply_model_compat_patches()


def test_diagnose_tokenizer_skew():
    msg = VLLMBackend._diagnose(
        AttributeError("Qwen2Tokenizer has no attribute all_special_tokens_extended"),
        "Qwen/Qwen3-14B-AWQ",
    )
    assert "transformers" in msg and "pin" in msg.lower()
    assert "Qwen/Qwen3-14B-AWQ" in msg


def test_diagnose_oom():
    msg = VLLMBackend._diagnose(RuntimeError("CUDA out of memory"), "x")
    assert "max_model_len" in msg


def test_diagnose_quantization():
    msg = VLLMBackend._diagnose(ValueError("model is not quantized with awq"), "x")
    assert "quantization" in msg.lower()


def test_diagnose_compute_capability():
    msg = VLLMBackend._diagnose(RuntimeError("Compute Capability < 8.0 is not supported"), "x")
    assert "Ampere" in msg or "V0" in msg
