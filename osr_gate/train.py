#!/usr/bin/env python3
"""Train the 24-way head (hyperbolic Busemann or Euclidean CE control).

Single invocation, one layer = one experiment (run three times for layers
1/2/5). Pipeline:

1. Build fixed ideal prototypes (Busemann head) and the head over cached
   ``(T, D)`` features; class-balanced sampling, 4 s random crops.
2. Train with AdamW + cosine LR; select the checkpoint by **dev-known top-1
   accuracy** (margin-free ``-Busemann`` argmax, or CE logits argmax).
3. Save ``best.pt`` / ``last.pt`` / ``meta.json`` (+ frozen prototypes) under
   ``artifacts/<head>_layer<NN>/``.

Run from ``osr_gate/`` (sibling imports ``common`` / ``model`` / ``data`` /
``hyperbolic``)::

    python train.py --feat-root ../feats_xlsr --layer 5 --head busemann
    python train.py --feat-root ../feats_mlaad --layer 5 --head euclidean
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from common import (add_repo_to_path, configure_logging, resolve_device,
                    seed_everything)
from data import DEFAULT_CROP_FRAMES, eval_loader, load_osr_splits, train_loader
from hyperbolic import PenalizedBusemannLoss, ideal_prototypes
from model import BusemannHead, EuclideanHead

logger = logging.getLogger("ser.osr_gate.train")

N_CLASSES = 24
#: Protocol dir resolved against the repo root, so cwd does not matter.
PROTO_DIR_DEFAULT = str(Path(__file__).resolve().parent.parent / "mlaad4sourcetracing")


# --------------------------------------------------------------------------- #
# Forward helpers (unify the two heads)
# --------------------------------------------------------------------------- #
def _busemann_logits(head, loss_fn, feats, mask):
    """Margin-free gate logits ``-Busemann(z)`` for the hyperbolic head."""
    return loss_fn(head(feats, mask), label=None)


def _euclidean_logits(head, feats, mask):
    _, logits = head(feats, mask)
    return logits


@torch.no_grad()
def dev_known_accuracy(head, loss_fn, loader, device, is_busemann) -> float:
    head.eval()
    correct = total = 0
    for feats, mask, labels in loader:
        feats, mask, labels = feats.to(device), mask.to(device), labels.to(device)
        if is_busemann:
            logits = _busemann_logits(head, loss_fn, feats, mask)
        else:
            logits = _euclidean_logits(head, feats, mask)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_head(
    *,
    feat_root: str,
    layer: int,
    head_kind: str = "busemann",
    protocol_dir: str = PROTO_DIR_DEFAULT,
    ball_dim: int = 16,
    phi: float = 1.1,
    embed_dim: int = 256,
    crop_frames: int = DEFAULT_CROP_FRAMES,
    batch_size: int = 32,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    device: str = "auto",
    out_dir: Optional[str] = None,
    seed: int = 0,
    max_steps: Optional[int] = None,
) -> Path:
    """Train one head on one layer; return the artifacts directory."""
    seed_everything(seed)
    dev = resolve_device(device)
    is_busemann = head_kind == "busemann"

    train_k, dev_k, _dev_u, _ev = load_osr_splits(protocol_dir, feat_root, layer)
    in_dim = _feature_dim(train_k)
    logger.info("layer %d | feat dim %d | train %d | dev-known %d | head %s",
                layer, in_dim, len(train_k), len(dev_k), head_kind)

    pin = dev.type == "cuda"   # pin only for GPU transfer (else forces a CUDA alloc)
    tl = train_loader(train_k, batch_size=batch_size, crop_frames=crop_frames,
                      num_workers=num_workers, pin_memory=pin)
    dl = eval_loader(dev_k, batch_size=batch_size, num_workers=num_workers,
                     pin_memory=pin)

    if is_busemann:
        prototypes = ideal_prototypes(N_CLASSES, ball_dim, seed=seed)
        head = BusemannHead(in_dim, ball_dim=ball_dim).to(dev)
        loss_fn = PenalizedBusemannLoss(prototypes, phi=phi).to(dev)
    else:
        prototypes = None
        head = EuclideanHead(in_dim, N_CLASSES, embed_dim=embed_dim).to(dev)
        loss_fn = torch.nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    out = Path(out_dir) if out_dir else Path("artifacts") / f"{head_kind}_layer{layer:02d}"
    out.mkdir(parents=True, exist_ok=True)

    best_acc, best_epoch = -1.0, -1
    step = 0
    for epoch in range(epochs):
        head.train()
        t0 = time.time()
        run_loss = 0.0
        n_batches = 0
        for feats, mask, labels in tl:
            feats, mask, labels = feats.to(dev), mask.to(dev), labels.to(dev)
            opt.zero_grad()
            if is_busemann:
                loss = loss_fn(head(feats, mask), labels)
            else:
                _, logits = head(feats, mask)
                loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            opt.step()
            run_loss += loss.item()
            n_batches += 1
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        sched.step()

        acc = dev_known_accuracy(head, loss_fn if is_busemann else None,
                                 dl, dev, is_busemann)
        logger.info("epoch %d/%d  loss %.4f  dev-known acc %.4f  (%.1fs)",
                    epoch + 1, epochs, run_loss / max(n_batches, 1), acc,
                    time.time() - t0)
        _save_ckpt(out / "last.pt", head, prototypes, head_kind, in_dim,
                   ball_dim, phi, embed_dim)
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch + 1
            _save_ckpt(out / "best.pt", head, prototypes, head_kind, in_dim,
                       ball_dim, phi, embed_dim)
        if max_steps is not None and step >= max_steps:
            break

    (out / "meta.json").write_text(json.dumps({
        "head": head_kind, "layer": layer, "feat_root": feat_root,
        "in_dim": in_dim, "ball_dim": ball_dim, "phi": phi,
        "embed_dim": embed_dim, "crop_frames": crop_frames,
        "epochs": epochs, "lr": lr, "best_epoch": best_epoch,
        "best_dev_known_acc": best_acc, "seed": seed,
    }, indent=2), encoding="utf-8")
    logger.info("Best dev-known acc %.4f @ epoch %d -> %s",
                best_acc, best_epoch, out)
    return out


def _feature_dim(clips) -> int:
    import numpy as np
    with np.load(clips[0].path, allow_pickle=False) as d:
        return int(d["features"].shape[-1])


def _save_ckpt(path, head, prototypes, head_kind, in_dim, ball_dim, phi,
               embed_dim) -> None:
    torch.save({
        "state_dict": head.state_dict(),
        "prototypes": prototypes,
        "head": head_kind, "in_dim": in_dim, "ball_dim": ball_dim,
        "phi": phi, "embed_dim": embed_dim,
    }, path)


def load_head(ckpt_path, device="cpu"):
    """Reload a trained head + (prototypes, loss_fn) from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt["head"] == "busemann":
        head = BusemannHead(ckpt["in_dim"], ball_dim=ckpt["ball_dim"])
        loss_fn = PenalizedBusemannLoss(ckpt["prototypes"], phi=ckpt["phi"])
    else:
        head = EuclideanHead(ckpt["in_dim"], N_CLASSES, embed_dim=ckpt["embed_dim"])
        loss_fn = None
    head.load_state_dict(ckpt["state_dict"])
    head.to(device).eval()
    if loss_fn is not None:
        loss_fn.to(device)
    return head, loss_fn, ckpt


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the OSR-gate 24-way head on one layer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--feat-root", required=True,
                   help="Feature root, e.g. ../feats_xlsr or ../feats_mlaad.")
    p.add_argument("--layer", type=int, required=True, choices=[1, 2, 5])
    p.add_argument("--head", choices=["busemann", "euclidean"], default="busemann")
    p.add_argument("--protocol-dir", default=PROTO_DIR_DEFAULT)
    p.add_argument("--ball-dim", type=int, default=16)
    p.add_argument("--phi", type=float, default=1.1)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--crop-frames", type=int, default=DEFAULT_CROP_FRAMES)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Cap total optimizer steps (smoke test).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    add_repo_to_path()
    configure_logging(verbose=args.verbose)
    train_head(
        feat_root=args.feat_root, layer=args.layer, head_kind=args.head,
        protocol_dir=args.protocol_dir, ball_dim=args.ball_dim, phi=args.phi,
        embed_dim=args.embed_dim, crop_frames=args.crop_frames,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, num_workers=args.num_workers,
        device=args.device, out_dir=args.out_dir, seed=args.seed,
        max_steps=args.max_steps,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
