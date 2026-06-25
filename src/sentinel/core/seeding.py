"""Deterministic seeding for reproducibility (paper §reproducibility requirements).

Every experiment captures and restores RNG state across ``random``, ``numpy``,
and ``torch`` (incl. CUDA) so a run is bit-reproducible given a seed and the
recorded environment manifest.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed all RNGs. Torch is seeded if available (it is, on the A100 path)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is a hard dep; this guard only helps doc builds
        pass
