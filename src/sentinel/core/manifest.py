"""Reproducibility manifest (paper §reproducibility requirements).

Captures everything needed to reproduce a run: seeds, config snapshot, dataset hash,
model identity, and hardware/runtime metadata.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RunManifest:
    run_id: str
    seed: int
    config: dict
    dataset_hash: str
    model_name: str
    encoder_name: str
    python: str = field(default_factory=lambda: sys.version)
    platform: str = field(default_factory=platform.platform)
    timestamp: float = field(default_factory=time.time)
    git_commit: str | None = None
    gpu: str | None = None

    def __post_init__(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                self.gpu = torch.cuda.get_device_name(0)
        except Exception:
            self.gpu = None

    @staticmethod
    def hash_dataset(hashes: list[str]) -> str:
        h = hashlib.sha256()
        for x in sorted(hashes):
            h.update(x.encode())
        return h.hexdigest()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))
