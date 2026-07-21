#!/usr/bin/env python3
"""The known/unknown gate: descriptors + calibrated logistic score (§5).

Pure array math over the caches written by :mod:`embed` -- the network is never
run here. For each clip a descriptor vector ``phi(x)`` is built, z-scored with
**dev** statistics, and fed to a tiny logistic regression giving the signed
score ``s(x) = w . phi + b`` (``>= 0`` known, ``< 0`` unknown).

Descriptors (all oriented so higher = more known):

    p_max, -H(p), p1-p2, energy(logsumexp z),           # from the 24 logits
    nearest-class term, -Mahalanobis,                    # from geometry
    hyperbolic radius                                    # busemann head only

So the hyperbolic head uses 7 descriptors, the Euclidean control 6. The z-score
mean/std are fit once on dev and reused on eval (no test leakage), and are
stored in ``gate.pkl`` next to the logistic weights.

Run from ``osr_gate/`` after :mod:`embed`::

    python gate.py --artifacts-dir artifacts/busemann_layer05
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from common import add_repo_to_path, configure_logging
from embed import maha_space, mahalanobis_min
from hyperbolic import poincare_radius

logger = logging.getLogger("ser.osr_gate.gate")

N_CLASSES = 24


# --------------------------------------------------------------------------- #
# Cache / stats IO
# --------------------------------------------------------------------------- #
def load_cache(art: Path, split: str) -> Dict[str, np.ndarray]:
    with np.load(art / f"emb_{split}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def load_stats(art: Path) -> Dict:
    with np.load(art / "gauss_stats.npz", allow_pickle=True) as z:
        return {"means": z["means"], "precision": z["precision"],
                "head": str(z["head"]), "prototypes": z["prototypes"]}


# --------------------------------------------------------------------------- #
# Descriptors
# --------------------------------------------------------------------------- #
def _softmax_stats(logits: np.ndarray):
    m = logits.max(axis=1, keepdims=True)
    ex = np.exp(logits - m)
    ssum = ex.sum(axis=1, keepdims=True)
    p = ex / ssum
    logsumexp = (m[:, 0] + np.log(ssum[:, 0]))          # energy, T=1
    return p, logsumexp


def build_descriptors(
    cache: Dict[str, np.ndarray], stats: Dict, is_busemann: bool
) -> Tuple[np.ndarray, List[str]]:
    """Return ``(phi (N, D), names)`` with all columns oriented higher = known."""
    emb = cache["emb"].astype(np.float32)
    logits = cache["logits"].astype(np.float32)
    p, energy = _softmax_stats(logits)

    p_max = p.max(axis=1)
    neg_entropy = (p * np.log(p + 1e-12)).sum(axis=1)   # = -H(p)
    psort = np.sort(p, axis=1)
    margin = psort[:, -1] - psort[:, -2]

    # nearest-class term (higher = closer to some known class)
    if is_busemann:
        # logits = -Busemann, so max logit = -min_k Busemann
        near = logits.max(axis=1)
    else:
        d = np.linalg.norm(emb[:, None, :] - stats["means"][None, :, :], axis=2)
        near = -d.min(axis=1)

    neg_maha = -mahalanobis_min(maha_space(emb, is_busemann),
                                stats["means"], stats["precision"])

    cols = [p_max, neg_entropy, margin, energy, near, neg_maha]
    names = ["p_max", "neg_entropy", "margin", "energy", "near", "neg_maha"]
    if is_busemann:
        radius = poincare_radius(torch.from_numpy(emb)).numpy()
        cols.append(radius)
        names.append("radius")
    return np.stack(cols, axis=1).astype(np.float32), names


# --------------------------------------------------------------------------- #
# Fit / apply
# --------------------------------------------------------------------------- #
def fit_gate(art: Path) -> Dict:
    """Fit the logistic gate on dev; save ``gate.pkl``; return the bundle."""
    import joblib
    from sklearn.linear_model import LogisticRegression

    stats = load_stats(art)
    is_busemann = stats["head"] == "busemann"

    dk = load_cache(art, "dev_known")
    du = load_cache(art, "dev_unknown")
    Xk, names = build_descriptors(dk, stats, is_busemann)
    Xu, _ = build_descriptors(du, stats, is_busemann)

    X = np.concatenate([Xk, Xu], axis=0)
    y = np.concatenate([np.ones(len(Xk)), np.zeros(len(Xu))]).astype(int)  # 1=known

    zmean = X.mean(axis=0)
    zstd = X.std(axis=0) + 1e-8
    Xz = (X - zmean) / zstd

    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(Xz, y)

    # §5.3 zero-training fallback midpoint (busemann radius only)
    r0 = None
    if is_busemann:
        rk = Xk[:, names.index("radius")]
        ru = Xu[:, names.index("radius")]
        r0 = float((rk.mean() + ru.mean()) / 2.0)

    bundle = {"clf": clf, "zmean": zmean, "zstd": zstd, "names": names,
              "head": stats["head"], "r0": r0}
    joblib.dump(bundle, art / "gate.pkl")

    train_auroc = _auroc(clf.predict_proba(Xz)[:, 1], y)
    logger.info("gate fit on dev (%d known / %d unknown), %d descriptors; "
                "dev-resub AUROC %.4f -> %s",
                len(Xk), len(Xu), len(names), train_auroc, art / "gate.pkl")
    return bundle


def load_gate(art: Path) -> Dict:
    import joblib
    return joblib.load(art / "gate.pkl")


def score_split(art: Path, split: str, bundle: Optional[Dict] = None):
    """Return ``(s, labels, is_known)`` for a split under the fitted gate.

    ``s`` is ``P(known)`` (sigmoid of the logistic gate's decision function).
    ``s >= 0.5`` -> known, ``s < 0.5`` -> unknown.
    """
    bundle = bundle or load_gate(art)
    stats = load_stats(art)
    cache = load_cache(art, split)
    X, _ = build_descriptors(cache, stats, bundle["head"] == "busemann")
    Xz = (X - bundle["zmean"]) / bundle["zstd"]
    s = bundle["clf"].predict_proba(Xz)[:, 1]
    return s, cache["labels"], cache["is_known"].astype(bool)


# --------------------------------------------------------------------------- #
# Small AUROC helper (self-contained; eval.py has the full metric suite)
# --------------------------------------------------------------------------- #
def _auroc(score: np.ndarray, known: np.ndarray) -> float:
    """AUROC for known(1) vs unknown(0) given a higher-is-known score."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(known)) < 2:
        return float("nan")
    return float(roc_auc_score(known, score))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit the known/unknown gate on cached descriptors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--artifacts-dir", required=True,
                   help="Dir with emb_*.npz + gauss_stats.npz from embed.py.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    add_repo_to_path()
    configure_logging(verbose=args.verbose)
    art = Path(args.artifacts_dir)
    fit_gate(art)
    # quick held-out sanity on eval (not the full metric suite)
    s, _, known = score_split(art, "eval")
    logger.info("eval known/unknown AUROC %.4f", _auroc(s, known))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
