#!/usr/bin/env python3
"""Per-class GMM density estimation + conformal threshold calibration.

Two GMM covariance modes:
    * ``full``     -- 5-component full-covariance for every class (baseline).
    * ``adaptive`` -- full for classes with >= ``min_samples_full`` training
                      samples, diagonal for the rest.

Calibration uses conformal prediction on the dev set: a sample is assigned to
the class with the lowest NLL only if that NLL is below the class threshold;
otherwise it is rejected as ``unknown``.

Run from ``cicmm/``::

    python gmm_fit.py --artifacts-dir artifacts/cicmm_e256_manual \\
                      --gmm-covariance full
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

from common import configure_logging, setup_paths

setup_paths()

logger = logging.getLogger("ser.cicmm.gmm_fit")

N_CLASSES = 24


# ---------------------------------------------------------------------- #
# GMM fitting
# ---------------------------------------------------------------------- #
def fit_gmms(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_classes: int = N_CLASSES,
    n_components: int = 5,
    covariance_type: str = "full",
    min_samples_full: int = 300,
    seed: int = 0,
) -> Dict[int, GaussianMixture]:
    """Fit one GMM per known class on training embeddings."""
    gmms: Dict[int, GaussianMixture] = {}
    for k in range(n_classes):
        X = embeddings[labels == k]
        if len(X) == 0:
            logger.warning("class %d has no training samples; skipped.", k)
            continue

        if covariance_type == "adaptive":
            cov = "full" if len(X) >= min_samples_full else "diag"
        else:
            cov = covariance_type

        nc = min(n_components, len(X))
        gmm = GaussianMixture(
            n_components=nc, covariance_type=cov,
            random_state=seed, max_iter=200, n_init=3,
        )
        gmm.fit(X)
        gmms[k] = gmm
        logger.debug("class %d: %d samples, %d comps, cov=%s, bic=%.1f",
                      k, len(X), nc, cov, gmm.bic(X))
    return gmms


# ---------------------------------------------------------------------- #
# NLL scoring
# ---------------------------------------------------------------------- #
def score_nll(gmms: Dict[int, GaussianMixture],
              embeddings: np.ndarray) -> np.ndarray:
    """Return ``(N, K)`` NLL matrix; ``nll[i, k] = -log p(x_i | GMM_k)``."""
    N = len(embeddings)
    K = max(gmms.keys()) + 1
    nll = np.full((N, K), np.inf, dtype=np.float64)
    for k, gmm in gmms.items():
        nll[:, k] = -gmm.score_samples(embeddings)
    return nll


# ---------------------------------------------------------------------- #
# Conformal threshold calibration
# ---------------------------------------------------------------------- #
def calibrate_thresholds(
    gmms: Dict[int, GaussianMixture],
    embeddings: np.ndarray,
    labels: np.ndarray,
    is_known: np.ndarray,
    coverage: float = 0.08,
) -> Dict[int, float]:
    """Per-class NLL thresholds via conformal prediction on the dev set.

    ``coverage`` is the target *rejection* fraction: ``1 - coverage`` of
    known dev samples per class should fall below the threshold.
    """
    thresholds: Dict[int, float] = {}
    for k, gmm in gmms.items():
        mask = (labels == k) & is_known
        if not mask.any():
            thresholds[k] = float("inf")
            continue
        nlls = -gmm.score_samples(embeddings[mask])
        q = min(1.0, (1 + 1.0 / len(nlls)) * (1 - coverage))
        thresholds[k] = float(np.quantile(nlls, q))
    return thresholds


# ---------------------------------------------------------------------- #
# Prediction with rejection
# ---------------------------------------------------------------------- #
def predict_with_rejection(
    nll: np.ndarray,
    thresholds: Dict[int, float],
) -> np.ndarray:
    """Assign to the lowest-NLL class if below threshold; else unknown (-1)."""
    best_class = nll.argmin(axis=1)
    best_nll = nll.min(axis=1)
    predictions = np.full(len(nll), -1, dtype=np.int64)
    for i in range(len(nll)):
        k = int(best_class[i])
        if best_nll[i] <= thresholds.get(k, float("inf")):
            predictions[i] = k
    return predictions


# ---------------------------------------------------------------------- #
# Grid search over coverage
# ---------------------------------------------------------------------- #
def grid_search_coverage(
    gmms: Dict[int, GaussianMixture],
    dev_emb: np.ndarray,
    dev_labels: np.ndarray,
    dev_is_known: np.ndarray,
    grid: Sequence[float] = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20),
) -> Tuple[float, Dict[int, float]]:
    """Search over ``coverage`` values, picking the one that maximises dev MacroF1."""
    from sklearn.metrics import f1_score

    nll = score_nll(gmms, dev_emb)
    dev_true = np.where(dev_is_known, dev_labels, N_CLASSES)
    best_f1, best_cov, best_thresh = -1.0, 0.08, {}

    for cov in grid:
        thresh = calibrate_thresholds(gmms, dev_emb, dev_labels, dev_is_known, cov)
        pred = predict_with_rejection(nll, thresh)
        pred_mapped = np.where(pred >= 0, pred, N_CLASSES)
        f1 = f1_score(dev_true, pred_mapped, average="macro", zero_division=0)
        logger.info("  coverage=%.3f  dev-MacroF1=%.4f", cov, f1)
        if f1 > best_f1:
            best_f1, best_cov, best_thresh = f1, cov, thresh

    logger.info("Best dev coverage=%.3f  MacroF1=%.4f", best_cov, best_f1)
    return best_cov, best_thresh


# ---------------------------------------------------------------------- #
# Driver
# ---------------------------------------------------------------------- #
def run(
    *,
    artifacts_dir: str,
    gmm_components: int = 5,
    gmm_covariance: str = "full",
    gmm_min_samples_full: int = 300,
    omega_s: float = 0.08,
    grid_search: bool = True,
    seed: int = 0,
) -> Path:
    art = Path(artifacts_dir)

    # Load cached embeddings
    train = np.load(art / "emb_train.npz", allow_pickle=True)
    dk = np.load(art / "emb_dev_known.npz", allow_pickle=True)
    du = np.load(art / "emb_dev_unknown.npz", allow_pickle=True)

    # Fit GMMs on training embeddings
    gmms = fit_gmms(
        train["emb"], train["labels"], N_CLASSES,
        n_components=gmm_components, covariance_type=gmm_covariance,
        min_samples_full=gmm_min_samples_full, seed=seed,
    )

    # Calibrate thresholds on dev (known + unknown)
    dev_emb = np.concatenate([dk["emb"], du["emb"]])
    dev_labels = np.concatenate([dk["labels"], du["labels"]])
    dev_is_known = np.concatenate([dk["is_known"], du["is_known"]])

    if grid_search:
        best_cov, thresholds = grid_search_coverage(
            gmms, dev_emb, dev_labels, dev_is_known)
    else:
        thresholds = calibrate_thresholds(
            gmms, dev_emb, dev_labels, dev_is_known, coverage=omega_s)
        best_cov = omega_s

    # Save
    joblib.dump({
        "gmms": gmms, "thresholds": thresholds,
        "gmm_covariance": gmm_covariance, "gmm_components": gmm_components,
        "best_coverage": best_cov,
    }, art / "gmm_bundle.pkl")
    logger.info("Saved gmm_bundle.pkl -> %s", art)
    return art


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit per-class GMMs + calibrate conformal thresholds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--artifacts-dir", required=True)
    p.add_argument("--gmm-components", type=int, default=5)
    p.add_argument("--gmm-covariance", choices=["full", "adaptive"], default="full")
    p.add_argument("--gmm-min-samples-full", type=int, default=300,
                   help="Threshold for switching to diagonal (adaptive mode).")
    p.add_argument("--omega-s", type=float, default=0.08)
    p.add_argument("--no-grid-search", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    run(
        artifacts_dir=args.artifacts_dir,
        gmm_components=args.gmm_components,
        gmm_covariance=args.gmm_covariance,
        gmm_min_samples_full=args.gmm_min_samples_full,
        omega_s=args.omega_s,
        grid_search=not args.no_grid_search,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
