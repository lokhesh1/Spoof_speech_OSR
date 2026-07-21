#!/usr/bin/env python3
"""Shared helpers for the ``cicmm/`` package."""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

logger = logging.getLogger("ser.cicmm")

REPO_ROOT = Path(__file__).resolve().parent.parent


def setup_paths() -> None:
    """Make ``protocols_mlaad`` and ``osr_gate/data`` importable from cicmm/.

    Repo root is inserted early (for ``protocols_mlaad``).  ``osr_gate/`` is
    *appended* so that same-named modules in cicmm/ (model, train, eval, …)
    are found first by the CWD entry that Python already placed at the front.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    osr = str(REPO_ROOT / "osr_gate")
    if osr not in sys.path:
        sys.path.append(osr)


def seed_everything(seed: int) -> None:
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
