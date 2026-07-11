#!/usr/bin/env python3
"""Cache embeddings + estimate Gaussian stats from a trained head.

After :mod:`train`, this embeds every clip (full sequence, ``eval()`` mode) once
and writes the results to disk so :mod:`gate` and :mod:`eval` are pure array
math -- the network is never run again.

Per split it caches ``emb`` and ``logits``:

    * **busemann** head -- ``emb`` is the Poincare-ball point ``z``; ``logits``
      are the margin-free ``-Busemann(z, P)``. The Mahalanobis space is the
      tangent-at-origin ``logmap0(z)``; the distance descriptor uses the fixed
      ideal prototypes ``P``.
    * **euclidean** head -- ``emb`` is the L2-normalized embedding ``e``;
      ``logits`` are the classifier outputs. Mahalanobis space is ``e`` itself;
      the distance descriptor uses the estimated class means.

Gaussian stats (class means + one shared Ledoit-Wolf precision) are estimated
from **train** only (principle P1) and saved to ``gauss_stats.npz``.

Run from ``osr_gate/``::

    python embed.py --feat-root ../feats_xlsr --layer 5 --head busemann
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from common import (add_repo_to_path, configure_logging, resolve_device,
                    seed_everything)
from data import eval_loader, load_osr_splits
from hyperbolic import logmap0
from train import N_CLASSES, PROTO_DIR_DEFAULT, load_head

logger = logging.getLogger("ser.osr_gate.embed")


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_embeddings(
    head, loss_fn, clips, device, is_busemann: bool, batch_size: int = 16,
    num_workers: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Return ``(emb, logits, labels, is_known, rels)`` for ``clips``.

    Full-sequence, ``eval()`` mode. ``emb`` is ``z`` (busemann) or ``e``
    (euclidean); ``logits`` are ``(N, 24)`` margin-free class scores.
    """
    head.eval()
    pin = device.type == "cuda"
    loader = eval_loader(clips, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=pin)
    embs, logits_all, labels_all = [], [], []
    for feats, mask, labels in loader:
        feats, mask = feats.to(device), mask.to(device)
        if is_busemann:
            z = head(feats, mask)
            logits = loss_fn(z, None)
            emb = z
        else:
            emb, logits = head(feats, mask)
        embs.append(emb.cpu().numpy())
        logits_all.append(logits.cpu().numpy())
        labels_all.append(labels.numpy())
    emb = np.concatenate(embs).astype(np.float32)
    logits = np.concatenate(logits_all).astype(np.float32)
    labels = np.concatenate(labels_all).astype(np.int64)
    is_known = np.array([c.is_known for c in clips], dtype=bool)
    rels = [c.rel for c in clips]
    return emb, logits, labels, is_known, rels


# --------------------------------------------------------------------------- #
# Shared-covariance Gaussian
# --------------------------------------------------------------------------- #
def maha_space(emb: np.ndarray, is_busemann: bool) -> np.ndarray:
    """Map cached embeddings into the space Mahalanobis is measured in.

    busemann -> tangent at origin (``logmap0``); euclidean -> identity.
    """
    if not is_busemann:
        return emb
    return logmap0(torch.from_numpy(emb)).numpy().astype(np.float32)


def estimate_shared_gaussian(
    x: np.ndarray, labels: np.ndarray, n_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-class means and one shared Ledoit-Wolf precision (inverse cov).

    Samples are centered by their class mean, then a single shrinkage
    covariance is fit on the pooled residuals (assume_centered).
    """
    from sklearn.covariance import LedoitWolf

    d = x.shape[1]
    means = np.zeros((n_classes, d), dtype=np.float64)
    for k in range(n_classes):
        xk = x[labels == k]
        if len(xk) == 0:
            logger.warning("class %d has no samples for Gaussian stats", k)
            continue
        means[k] = xk.mean(0)
    centered = x - means[labels]
    precision = LedoitWolf(assume_centered=True).fit(centered).precision_
    return means.astype(np.float32), precision.astype(np.float32)


def mahalanobis_min(x: np.ndarray, means: np.ndarray,
                    precision: np.ndarray) -> np.ndarray:
    """``min_k (x-m_k)^T P (x-m_k)`` for each row of ``x`` -> ``(N,)``."""
    # (N, K, d) diffs -> quadratic form via einsum
    diff = x[:, None, :] - means[None, :, :]          # (N, K, d)
    m = np.einsum("nkd,de,nke->nk", diff, precision, diff)  # (N, K)
    return m.min(axis=1)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(
    *,
    feat_root: str,
    layer: int,
    head_kind: str,
    artifacts_dir: str,
    protocol_dir: str = PROTO_DIR_DEFAULT,
    batch_size: int = 16,
    num_workers: int = 2,
    device: str = "auto",
    seed: int = 0,
) -> Path:
    seed_everything(seed)
    dev = resolve_device(device)
    is_busemann = head_kind == "busemann"

    art = Path(artifacts_dir)
    head, loss_fn, ckpt = load_head(art / "best.pt", device=dev)
    logger.info("loaded %s head (in_dim=%d) from %s", head_kind,
                ckpt["in_dim"], art / "best.pt")

    train_k, dev_k, dev_u, eval_all = load_osr_splits(protocol_dir, feat_root, layer)
    splits = {"train": train_k, "dev_known": dev_k,
              "dev_unknown": dev_u, "eval": eval_all}

    cached = {}
    for name, clips in splits.items():
        emb, logits, labels, is_known, rels = compute_embeddings(
            head, loss_fn, clips, dev, is_busemann, batch_size, num_workers)
        np.savez(art / f"emb_{name}.npz", emb=emb, logits=logits,
                 labels=labels, is_known=is_known, rels=np.asarray(rels))
        cached[name] = (emb, labels)
        logger.info("%s: %d clips embedded -> emb_%s.npz", name, len(labels), name)

    # Gaussian stats from train only
    tr_emb, tr_labels = cached["train"]
    x = maha_space(tr_emb, is_busemann)
    means, precision = estimate_shared_gaussian(x, tr_labels, N_CLASSES)
    prototypes = ckpt.get("prototypes")
    np.savez(
        art / "gauss_stats.npz",
        means=means, precision=precision, head=head_kind,
        prototypes=(prototypes.numpy() if prototypes is not None
                    else np.zeros(0, dtype=np.float32)),
    )
    logger.info("gauss_stats.npz: means %s, precision %s",
                means.shape, precision.shape)
    return art


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cache embeddings + Gaussian stats from a trained head.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--feat-root", required=True)
    p.add_argument("--layer", type=int, required=True, choices=[1, 2, 5])
    p.add_argument("--head", choices=["busemann", "euclidean"], default="busemann")
    p.add_argument("--artifacts-dir", default=None,
                   help="Defaults to artifacts/<head>_layer<NN>/.")
    p.add_argument("--protocol-dir", default=PROTO_DIR_DEFAULT)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    add_repo_to_path()
    configure_logging(verbose=args.verbose)
    art = args.artifacts_dir or f"artifacts/{args.head}_layer{args.layer:02d}"
    run(
        feat_root=args.feat_root, layer=args.layer, head_kind=args.head,
        artifacts_dir=art, protocol_dir=args.protocol_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        device=args.device, seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
