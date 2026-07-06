#!/usr/bin/env python3
"""Train the LFCC + per-class GMM + one-class-SVM source-tracing baseline.

Pipeline (raw MLAAD audio -> artifacts, no PTM embeddings)::

    1. LFCC per clip (20-dim, 30 ms / 59 % overlap frames).
    2. One GaussianMixture(512, diagonal) per known class fit on that class's
       pooled LFCC frames.
    3. Each train clip -> K-dim vector of per-GMM mean log-likelihoods.
    4. StandardScaler + single global OneClassSVM on those vectors -> the
       known/unknown gate.

Only the 24 models in ``train.csv`` are known classes (indexed 0..23). Saves a
single joblib bundle consumed by ``eval.py``.

Example (smoke test on 200 clips, CPU)::

    python train.py \\
        --mlaad-root /path/to/Mlaad_v5/mlaad_v5 \\
        --out artifacts/model.joblib \\
        --cache-dir cache/lfcc --limit 200
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

import common
from common import (
    LFCCConfig, build_lfcc_transform, lfcc_cached, score_vector,
    save_artifacts, add_repo_to_path,
)

logger = logging.getLogger("gmm_baseline.train")

DEFAULT_PROTOCOL_DIR = Path(__file__).resolve().parent.parent / "mlaad4sourcetracing"


def _fit_class_gmm(
    frames: np.ndarray,
    *,
    n_components: int,
    max_iter: int,
    tol: float,
    reg_covar: float,
    seed: int,
):
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        max_iter=max_iter,
        tol=tol,
        reg_covar=reg_covar,
        random_state=seed,
    )
    # float64 for numerical stability (avoids collapsed-component Cholesky errors)
    gmm.fit(frames.astype(np.float64, copy=False))
    return gmm


def train(
    *,
    mlaad_root: Path,
    protocol_dir: Path,
    out_path: Path,
    cache_dir: Optional[Path],
    cfg: LFCCConfig,
    n_components: int,
    max_iter: int,
    tol: float,
    reg_covar: float,
    max_frames_per_class: Optional[int],
    ocsvm_nu: float,
    ocsvm_gamma: str,
    seed: int,
    limit: Optional[int],
    per_class_limit: Optional[int],
    log_every: int,
) -> None:
    add_repo_to_path()
    from protocols_mlaad import load_split, build_label_map

    label_map = build_label_map(protocol_dir)
    classes = [m for m, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    K = len(classes)
    logger.info("Known classes: %d", K)

    clips = load_split(protocol_dir, "train", mlaad_root=mlaad_root,
                       label_map=label_map)
    if per_class_limit:
        seen: dict[int, int] = defaultdict(int)
        kept = []
        for c in clips:
            if seen[c.label] < per_class_limit:
                seen[c.label] += 1
                kept.append(c)
        clips = kept
    if limit:
        clips = clips[:limit]
    logger.info("Train clips: %d", len(clips))

    transform = build_lfcc_transform(cfg)

    # ---- gather LFCC frames per class -------------------------------------- #
    by_class: dict[int, List[np.ndarray]] = defaultdict(list)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for i, c in enumerate(clips):
        feat = lfcc_cached(c.rel, c.path, transform, cfg, cache_dir)
        if feat.shape[0] == 0:
            logger.warning("Empty LFCC (skipped): %s", c.rel)
            continue
        by_class[c.label].append(feat)
        if (i + 1) % log_every == 0 or (i + 1) == len(clips):
            logger.info("LFCC %d/%d (%.1f clip/s)", i + 1, len(clips),
                        (i + 1) / (time.time() - t0))

    present = sorted(by_class)
    if len(present) < K:
        missing = [classes[j] for j in range(K) if j not in by_class]
        logger.warning("%d classes have no clips (limit too small?): %s",
                       len(missing), missing)

    # ---- fit one GMM per class -------------------------------------------- #
    gmms: List[Optional[object]] = [None] * K
    for label in present:
        frames = np.vstack(by_class[label])
        if max_frames_per_class and frames.shape[0] > max_frames_per_class:
            idx = rng.choice(frames.shape[0], max_frames_per_class, replace=False)
            frames = frames[idx]
        n_comp = min(n_components, frames.shape[0])
        if n_comp < n_components:
            logger.warning("class %d (%s): only %d frames, reducing components %d->%d",
                           label, classes[label], frames.shape[0], n_components, n_comp)
        t = time.time()
        gmms[label] = _fit_class_gmm(
            frames, n_components=n_comp, max_iter=max_iter, tol=tol,
            reg_covar=reg_covar, seed=seed,
        )
        logger.info("GMM class %2d/%d (%s): %d frames, %d comps, %.1fs",
                    label, K, classes[label], frames.shape[0], n_comp, time.time() - t)

    if any(g is None for g in gmms):
        raise RuntimeError("Some classes have no GMM; cannot build score vectors. "
                           "Increase --limit or check the data.")

    # ---- score vectors + one-class SVM gate ------------------------------- #
    logger.info("Building score vectors for %d train clips...", len(clips))
    C = np.empty((len(clips), K), dtype=np.float64)
    for i, c in enumerate(clips):
        feat = lfcc_cached(c.rel, c.path, transform, cfg, cache_dir)
        C[i] = score_vector(feat, gmms)

    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM

    scaler = StandardScaler().fit(C)
    ocsvm = OneClassSVM(kernel="rbf", nu=ocsvm_nu, gamma=ocsvm_gamma)
    ocsvm.fit(scaler.transform(C))
    inlier_frac = float(np.mean(ocsvm.predict(scaler.transform(C)) == 1))
    logger.info("OneClassSVM (nu=%.3f, gamma=%s): %.1f%% of train kept as inliers",
                ocsvm_nu, ocsvm_gamma, 100 * inlier_frac)

    save_artifacts(out_path, {
        "config": cfg.to_dict(),
        "label_map": label_map,
        "classes": classes,
        "gmms": gmms,
        "scaler": scaler,
        "ocsvm": ocsvm,
        "ocsvm_params": {"nu": ocsvm_nu, "gamma": ocsvm_gamma},
        "gmm_params": {"n_components": n_components, "covariance_type": "diag",
                       "max_iter": max_iter, "tol": tol, "reg_covar": reg_covar},
    })


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the LFCC+GMM+OCSVM source-tracing baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mlaad-root", required=True,
                   help="MLAAD root containing fake/<lang>/<model>/ (raw .wav).")
    p.add_argument("--protocol-dir", default=str(DEFAULT_PROTOCOL_DIR),
                   help="Path to mlaad4sourcetracing/.")
    p.add_argument("--out", default="artifacts/model.joblib",
                   help="Output joblib bundle.")
    p.add_argument("--cache-dir", default=None,
                   help="Optional dir to cache per-clip LFCC .npy (recommended).")
    # GMM
    p.add_argument("--n-components", type=int, default=512)
    p.add_argument("--max-iter", type=int, default=400)
    p.add_argument("--tol", type=float, default=0.005)
    p.add_argument("--reg-covar", type=float, default=1e-3,
                   help="Variance floor per component; prevents collapsed "
                        "(zero-width) components with 512 mixtures.")
    p.add_argument("--max-frames-per-class", type=int, default=None,
                   help="Subsample frames per class before GMM fit (default: all).")
    # OCSVM
    p.add_argument("--ocsvm-nu", type=float, default=0.1,
                   help="OneClassSVM nu (upper bound on training outlier fraction).")
    p.add_argument("--ocsvm-gamma", default="scale", help="OneClassSVM gamma.")
    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="Use only the first N train clips (smoke test).")
    p.add_argument("--per-class-limit", type=int, default=None,
                   help="Keep at most N clips per class (stratified smoke test).")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    train(
        mlaad_root=Path(args.mlaad_root),
        protocol_dir=Path(args.protocol_dir),
        out_path=Path(args.out),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        cfg=LFCCConfig(),
        n_components=args.n_components,
        max_iter=args.max_iter,
        tol=args.tol,
        reg_covar=args.reg_covar,
        max_frames_per_class=args.max_frames_per_class,
        ocsvm_nu=args.ocsvm_nu,
        ocsvm_gamma=args.ocsvm_gamma,
        seed=args.seed,
        limit=args.limit,
        per_class_limit=args.per_class_limit,
        log_every=args.log_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
