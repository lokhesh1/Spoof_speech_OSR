#!/usr/bin/env python3
"""Shared helpers for the ``osr_gate/`` package (mirrors hier_spec/common.py)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger("ser.osr_gate")


def add_repo_to_path() -> None:
    """Make ``protocols_mlaad`` / ``extract_mlaad`` importable from osr_gate/."""
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto"):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("numba", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
