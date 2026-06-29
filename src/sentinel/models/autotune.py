"""Hardware auto-tuning so ONE config maxes out ANY GPU.

Detects the GPU (count, name, VRAM, compute capability) and derives serving + batching
parameters that saturate whatever it runs on — a tiny T4 or 8x B200 — with no per-machine
config edits. Resolves the ``auto`` sentinels used throughout the configs.
"""

from __future__ import annotations

from ..core.logging import get_logger

log = get_logger(__name__)


def gpu_profile() -> dict:
    """Best-effort GPU facts. Returns count=0 on a CPU box (callers fall back to defaults)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"count": 0, "name": "cpu", "vram_gb": 0.0, "cc": (0, 0)}
        props = torch.cuda.get_device_properties(0)
        return {
            "count": torch.cuda.device_count(),
            "name": props.name,
            "vram_gb": round(props.total_memory / (1024**3), 1),
            "cc": (props.major, props.minor),
        }
    except Exception as exc:
        log.warning("gpu_profile failed; assuming CPU", error=str(exc))
        return {"count": 0, "name": "unknown", "vram_gb": 0.0, "cc": (0, 0)}


def autotune_batch_size(vram_gb: float) -> int:
    """Generation batch size scaled to VRAM headroom (after ~10GB for a 14B-AWQ + encoder).

    The model+encoder take ~11GB; the rest is KV cache, which is what bounds concurrency.
    Bigger GPU -> far more concurrent sequences -> higher utilization.
    """
    if vram_gb <= 0:        # CPU / unknown -> conservative
        return 8
    if vram_gb <= 16:       # T4 / V100 16GB
        return 8
    if vram_gb <= 32:       # L4 24GB, V100 32GB
        return 24
    if vram_gb <= 50:       # A100/L40S 40-48GB
        return 48
    if vram_gb <= 96:       # A100/H100 80GB
        return 96
    return 192              # H200 141GB / B200 180GB+ -> very large batches


def resolve_serving(model_cfg: dict, profile: dict | None = None) -> dict:
    """Resolve ``auto`` serving fields in a model config from the detected GPU."""
    prof = profile or gpu_profile()
    cfg = dict(model_cfg)
    vram = prof["vram_gb"]

    if str(cfg.get("gpu_memory_utilization", "auto")) == "auto":
        # leave headroom for the co-located encoder; high on big cards, safer on tiny ones
        cfg["gpu_memory_utilization"] = 0.85 if 0 < vram <= 16 else 0.90
    if str(cfg.get("max_model_len", "auto")) == "auto":
        cfg["max_model_len"] = 4096 if 0 < vram <= 16 else 8192
    log.info("autotuned serving", gpu=prof["name"], vram_gb=vram, count=prof["count"],
             gpu_mem_util=cfg["gpu_memory_utilization"], max_model_len=cfg["max_model_len"])
    return cfg


def autotune_summary() -> str:
    p = gpu_profile()
    if p["count"] == 0:
        return "no CUDA GPU detected (CPU box)"
    return (f"{p['count']}x {p['name']} ({p['vram_gb']}GB, CC {p['cc'][0]}.{p['cc'][1]}) "
            f"-> batch_size={autotune_batch_size(p['vram_gb'])}")
