"""
Reproducibility — INSIDE-OUT / EMHA.

set_seed() is the single entrypoint for seeding every source of randomness
the pipeline touches. Call it once, at the start of a run, before building
datasets, models, or dataloaders.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed random, numpy, torch, and torch.cuda for a reproducible run.

    Also sets PYTHONHASHSEED, which only affects hash() in subprocesses
    started after this call (e.g. spawned DataLoader workers) — it cannot
    change hash randomization in the already-running process.

    Args:
        seed: seed value applied to every RNG.
        deterministic: if True, force cuDNN into deterministic mode and
            enable torch's deterministic-algorithms check in warn-only mode
            (some ops have no deterministic CUDA kernel; warn_only logs
            instead of raising for those).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
