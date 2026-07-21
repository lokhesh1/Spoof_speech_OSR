#!/usr/bin/env python3
"""Cache embeddings from a frozen C-ICMM model for all splits.

Writes ``emb_<split>.npz`` containing ``emb``, ``logits``, ``labels``,
``is_known``, and ``rels`` for each split.  Also computes per-class sample
counts (needed by ``gmm_fit.py`` for the adaptive-covariance decision).

Run from ``cicmm/``::

    python embed.py --feat-root ../feats_xlsr --layer 5 \\
                    --artifacts-dir artifacts/cicmm_e256_manual
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from common import (configure_logging, resolve_device, seed_everything,
                    setup_paths)

setup_paths()

from data import eval_loader, load_osr_splits  # osr_gate

from train import PROTO_DIR_DEFAULT, N_CLASSES, load_model

logger = logging.getLogger("ser.cicmm.embed")


@torch.no_grad()
def compute_embeddings(
    model, clips, device, batch_size: int = 16, num_workers: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Return ``(emb, logits, labels, is_known, rels)``."""
    model.eval()
    pin = device.type == "cuda"
    loader = eval_loader(clips, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=pin)
    embs, logits_all, labels_all = [], [], []
    for feats, mask, labels in loader:
        feats, mask = feats.to(device), mask.to(device)
        emb, logits = model(feats, mask)
        embs.append(emb.cpu().numpy())
        logits_all.append(logits.cpu().numpy())
        labels_all.append(labels.numpy())
    emb = np.concatenate(embs).astype(np.float32)
    logits = np.concatenate(logits_all).astype(np.float32)
    labels = np.concatenate(labels_all).astype(np.int64)
    is_known = np.array([c.is_known for c in clips], dtype=bool)
    rels = [c.rel for c in clips]
    return emb, logits, labels, is_known, rels


def run(
    *,
    feat_root: str,
    layer: int,
    artifacts_dir: str,
    protocol_dir: str = PROTO_DIR_DEFAULT,
    batch_size: int = 16,
    num_workers: int = 2,
    device: str = "auto",
    seed: int = 0,
) -> Path:
    seed_everything(seed)
    dev = resolve_device(device)

    art = Path(artifacts_dir)
    model, ckpt = load_model(art / "best.pt", device=dev)
    logger.info("Loaded model (in_dim=%d, embed=%dD) from %s",
                ckpt["in_dim"], ckpt["embed_dim"], art / "best.pt")

    train_k, dev_k, dev_u, eval_all = load_osr_splits(protocol_dir, feat_root, layer)
    splits = {"train": train_k, "dev_known": dev_k,
              "dev_unknown": dev_u, "eval": eval_all}

    class_counts = {}
    for name, clips in splits.items():
        emb, logits, labels, is_known, rels = compute_embeddings(
            model, clips, dev, batch_size, num_workers)
        np.savez(art / f"emb_{name}.npz", emb=emb, logits=logits,
                 labels=labels, is_known=is_known, rels=np.asarray(rels))
        if name == "train":
            for k in range(N_CLASSES):
                class_counts[k] = int((labels == k).sum())
        logger.info("%s: %d clips embedded -> emb_%s.npz", name, len(labels), name)

    (art / "class_counts.json").write_text(
        json.dumps(class_counts, indent=2), encoding="utf-8")
    logger.info("class_counts.json: %s", class_counts)
    return art


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cache embeddings from a frozen C-ICMM model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--feat-root", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--artifacts-dir", required=True)
    p.add_argument("--protocol-dir", default=PROTO_DIR_DEFAULT)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    run(
        feat_root=args.feat_root, layer=args.layer,
        artifacts_dir=args.artifacts_dir, protocol_dir=args.protocol_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        device=args.device, seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
