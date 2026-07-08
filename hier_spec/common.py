#!/usr/bin/env python3
"""Shared configuration and helpers for the Hier-Spec system.

Hier-Spec is the *second* source-tracing system in this repo (after
``gmm_baseline/``). It is a hierarchical, architecture-then-model classifier:

    cached Wav2Vec2-Base layer-5 frames (T, 768)
        -> AASIST graph backend            (aasist.py)
        -> L2-normalised 160-d embedding
        -> ArcFace architecture head + per-architecture model heads (model.py)
        -> two-stage Mahalanobis OOD gate  (calibrate.py / eval.py)

Feature extraction is done up front by ``extract_feats.py`` (mirroring the
``extract_mlaad.py`` output layout), so training/eval read ``.npz`` frames and
never run Wav2Vec2 themselves.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("hier_spec.common")

EMBED_DIM = 160          # AASIST last_hidden = 5 * gat_dims[1] = 5 * 32
UNKNOWN_LABEL = -1


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HierSpecConfig:
    # feature front-end (fixed by spec)
    hf_id: str = "facebook/wav2vec2-base"
    layer: int = 5
    feat_dim: int = 768
    sample_rate: int = 16_000
    # feature-frame window used for batching (clips are cached at full length;
    # this window is applied at train/eval time and can change without
    # re-extracting features)
    max_frames: int = 200                # ~4 s of 20 ms Wav2Vec2 frames
    # ArcFace
    arcface_s: float = 30.0
    arcface_m: float = 0.3
    # Mahalanobis ridge regularisation: Sigma + lambda * mean(diag(Sigma)) * I
    ridge_lambda: float = 1e-3
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Repo import helper (mirrors gmm_baseline/common.py:add_repo_to_path)
# --------------------------------------------------------------------------- #
def add_repo_to_path() -> None:
    """Make ``protocols_mlaad`` / ``extract_mlaad`` importable from hier_spec/."""
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    import random

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
        format="%(levelname)s %(name)s: %(message)s",
    )


# --------------------------------------------------------------------------- #
# Feature-frame window (crop / repeat-pad to a fixed number of frames)
# --------------------------------------------------------------------------- #
def fit_frames(feat: np.ndarray, n_frames: int, *, train: bool,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Crop or repeat-pad an ``(T, D)`` feature sequence to exactly ``n_frames``.

    ``train`` uses a random crop / random pad offset; otherwise the crop is
    centred. Short sequences are tiled (repeat-padded), matching common
    Wav2Vec2-AASIST practice.
    """
    t = feat.shape[0]
    if t == n_frames:
        return feat
    if t > n_frames:
        if train:
            start = int((rng or np.random).integers(0, t - n_frames + 1)) \
                if rng is not None else np.random.randint(0, t - n_frames + 1)
        else:
            start = (t - n_frames) // 2
        return feat[start:start + n_frames]
    # t < n_frames: tile then crop
    reps = int(np.ceil(n_frames / t))
    tiled = np.tile(feat, (reps, 1))
    return tiled[:n_frames]


# --------------------------------------------------------------------------- #
# Mahalanobis statistics (shared pooled covariance per hierarchy level)
# --------------------------------------------------------------------------- #
def estimate_gaussian_stats(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    ridge_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Class means and a Cholesky factor of the ridge-regularised pooled cov.

    Returns ``(means (n_classes, D), chol (D, D))`` where ``chol`` is the lower
    Cholesky factor ``L`` of ``Sigma_reg = Sigma + lambda*mean(diag(Sigma))*I``
    and ``Sigma`` is the pooled within-class covariance. Mahalanobis distance
    is then ``||solve(L, x - mu_c)||`` (see :func:`mahalanobis`).
    """
    x = embeddings.astype(np.float64, copy=False)
    d = x.shape[1]
    means = np.zeros((n_classes, d), dtype=np.float64)
    for c in range(n_classes):
        m = labels == c
        if not np.any(m):
            logger.warning("class %d has no samples for mean estimation.", c)
            continue
        means[c] = x[m].mean(axis=0)
    centered = x - means[labels]
    sigma = (centered.T @ centered) / max(len(x), 1)
    sigma += ridge_lambda * float(np.mean(np.diag(sigma))) * np.eye(d)
    chol = np.linalg.cholesky(sigma)
    return means, chol


def mahalanobis(x: np.ndarray, means: np.ndarray, chol: np.ndarray) -> np.ndarray:
    """Mahalanobis distance from each row of ``x`` to every class mean.

    ``x`` is ``(N, D)``, ``means`` ``(C, D)``, ``chol`` the ``(D, D)`` lower
    factor. Returns ``(N, C)`` distances. Solves ``L y = (x - mu)`` per class
    via triangular solve (Cholesky), so no explicit inverse is formed.
    """
    from scipy.linalg import solve_triangular

    x = x.astype(np.float64, copy=False)
    n, d = x.shape
    c = means.shape[0]
    out = np.empty((n, c), dtype=np.float64)
    for j in range(c):
        diff = (x - means[j]).T                       # (D, N)
        y = solve_triangular(chol, diff, lower=True)  # (D, N)
        out[:, j] = np.sqrt(np.sum(y * y, axis=0))
    return out


# --------------------------------------------------------------------------- #
# EER / AUROC for the OOD gates (positive class = OOD / "reject")
# --------------------------------------------------------------------------- #
def eer_threshold(scores: np.ndarray, is_ood: np.ndarray) -> Tuple[float, float]:
    """Equal-error-rate threshold on OOD scores (higher score = more OOD).

    ``is_ood`` is a boolean array (True = should be rejected). Returns
    ``(threshold, eer)``; accept iff ``score <= threshold``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    is_ood = np.asarray(is_ood, dtype=bool)
    if is_ood.all() or (~is_ood).all():
        logger.warning("EER undefined (only one class present); using median.")
        return float(np.median(scores)), float("nan")
    n_ood = int(is_ood.sum())
    n_id = int((~is_ood).sum())
    # Candidate thresholds = the sorted unique scores; accept iff score <= thr,
    # so FAR = frac(OOD accepted), FRR = frac(ID rejected). Pick the threshold
    # minimising |FAR - FRR| and report EER = (FAR + FRR) / 2 there.
    candidates = np.unique(scores)
    best_thr, best_gap, best_eer = float(candidates[0]), np.inf, float("nan")
    for thr in candidates:
        accepted = scores <= thr
        far = float(np.sum(accepted & is_ood)) / n_ood
        frr = float(np.sum(~accepted & ~is_ood)) / n_id
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap, best_thr, best_eer = gap, float(thr), (far + frr) / 2.0
    return best_thr, float(best_eer)


def auroc(scores: np.ndarray, is_ood: np.ndarray) -> float:
    """AUROC of OOD scores (higher = more OOD) as an OOD detector."""
    from sklearn.metrics import roc_auc_score

    is_ood = np.asarray(is_ood, dtype=bool)
    if is_ood.all() or (~is_ood).all():
        return float("nan")
    return float(roc_auc_score(is_ood.astype(int), np.asarray(scores)))
