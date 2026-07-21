#!/usr/bin/env python3
"""Train the NovelModel with C-ICMM (CE + SupCon + curriculum ICMM).

The warm phase (epochs 1..warm_epochs) uses tight omega [0.45, 0.55];
the expansion phase (warm_epochs+1..epochs) widens to [0.2, 0.8].
Epoch ``warm_epochs`` is typically the peak-performance checkpoint.

If ``--icmm-weighting auto`` is selected, the pair-weight matrix is
recomputed from learned centroids at the end of the warm phase.

Run from ``cicmm/``::

    python train.py --feat-root ../feats_xlsr --layer 5 \\
                    --embed-dim 256 --icmm-weighting manual
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional, Sequence

import torch

from common import (configure_logging, resolve_device, seed_everything,
                    setup_paths)

setup_paths()

from data import eval_loader, load_osr_splits, train_loader          # osr_gate
from protocols_mlaad import build_label_map                          # repo root

from losses import CentroidTracker, SupConLoss, generate_synthetic, icmm_loss
from model import NovelModel
from pair_weights import auto_pair_weights, manual_pair_weights

logger = logging.getLogger("ser.cicmm.train")

N_CLASSES = 24
PROTO_DIR_DEFAULT = str(Path(__file__).resolve().parent.parent / "mlaad4sourcetracing")


# ---------------------------------------------------------------------- #
# Dev-known accuracy (fast; for checkpoint selection)
# ---------------------------------------------------------------------- #
@torch.no_grad()
def dev_known_accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for feats, mask, labels in loader:
        feats, mask, labels = feats.to(device), mask.to(device), labels.to(device)
        _, logits = model(feats, mask)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


# ---------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------- #
def train_cicmm(
    *,
    feat_root: str,
    layer: int,
    protocol_dir: str = PROTO_DIR_DEFAULT,
    # Model
    embed_dim: int = 256,
    hidden_dim: int = 512,
    n_heads: int = 4,
    # Schedule
    epochs: int = 60,
    warm_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    crop_frames: int = 200,
    # Loss weights
    w_ce: float = 0.5,
    w_sc: float = 0.5,
    lambda_icmm: float = 0.5,
    supcon_tau: float = 0.05,
    # ICMM
    n_synthetic: int = 32,
    icmm_weighting: str = "manual",
    centroid_momentum: float = 0.99,
    centroid_mode: str = "ema",
    # System
    num_workers: int = 4,
    device: str = "auto",
    seed: int = 0,
    out_dir: str | None = None,
    max_steps: int | None = None,
) -> Path:
    """Train one C-ICMM experiment; return the artifacts directory."""
    seed_everything(seed)
    dev = resolve_device(device)

    # Data
    label_map = build_label_map(protocol_dir)
    train_k, dev_k, _dev_u, _ev = load_osr_splits(protocol_dir, feat_root, layer)
    in_dim = _feature_dim(train_k)
    logger.info("layer %d | feat %dD | train %d | dev-known %d | embed %dD | icmm %s | centroid %s",
                layer, in_dim, len(train_k), len(dev_k), embed_dim, icmm_weighting, centroid_mode)

    pin = dev.type == "cuda"
    tl = train_loader(train_k, batch_size=batch_size, crop_frames=crop_frames,
                      num_workers=num_workers, pin_memory=pin)
    dl = eval_loader(dev_k, batch_size=batch_size, num_workers=num_workers,
                     pin_memory=pin)

    # Model
    model = NovelModel(in_dim, hidden_dim, embed_dim, n_heads, N_CLASSES).to(dev)

    # Losses
    ce_fn = torch.nn.CrossEntropyLoss()
    sc_fn = SupConLoss(supcon_tau)

    # Pair weights
    if icmm_weighting == "manual":
        pair_matrix = manual_pair_weights(label_map).to(dev)
    else:
        pair_matrix = torch.ones(N_CLASSES, N_CLASSES, device=dev)
        pair_matrix.fill_diagonal_(0)

    # Centroids
    centroids = CentroidTracker(N_CLASSES, embed_dim,
                                momentum=centroid_momentum,
                                mode=centroid_mode, device=dev)

    # Optimiser
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    out = Path(out_dir) if out_dir else Path("artifacts") / f"cicmm_e{embed_dim}_{icmm_weighting}"
    out.mkdir(parents=True, exist_ok=True)

    best_acc, best_epoch = -1.0, -1
    step = 0
    for epoch in range(epochs):
        omega_range = (0.45, 0.55) if epoch < warm_epochs else (0.2, 0.8)

        model.train()
        t0 = time.time()
        run_loss = run_ce = run_sc = run_icmm = 0.0
        n_batches = 0

        for feats, mask, labels in tl:
            feats, mask, labels = feats.to(dev), mask.to(dev), labels.to(dev)

            emb, logits = model(feats, mask)

            loss_ce = ce_fn(logits, labels)
            loss_sc = sc_fn(emb, labels)

            synthetic, syn_weights = generate_synthetic(emb, labels, omega_range,
                                                        n_synthetic, pair_matrix)
            if synthetic is not None and centroids.all_initialized():
                loss_ic = icmm_loss(synthetic, centroids.get(), syn_weights)
            else:
                loss_ic = torch.tensor(0.0, device=dev)

            loss = w_ce * loss_ce + w_sc * loss_sc + lambda_icmm * loss_ic

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            centroids.update(emb.detach(), labels)

            run_loss += loss.item()
            run_ce += loss_ce.item()
            run_sc += loss_sc.item()
            run_icmm += loss_ic.item()
            n_batches += 1
            step += 1
            if max_steps is not None and step >= max_steps:
                break

        sched.step()

        # Auto pair weights: recompute once at the end of the warm phase
        if icmm_weighting == "auto" and epoch == warm_epochs - 1:
            pair_matrix = auto_pair_weights(centroids.get()).to(dev)
            logger.info("Auto pair weights recomputed from centroids.")

        acc = dev_known_accuracy(model, dl, dev)
        nb = max(n_batches, 1)
        logger.info(
            "epoch %d/%d  loss %.4f (ce %.4f sc %.4f icmm %.4f)  "
            "dev-acc %.4f  (%.1fs)",
            epoch + 1, epochs, run_loss / nb, run_ce / nb,
            run_sc / nb, run_icmm / nb, acc, time.time() - t0,
        )

        _save_ckpt(out / "last.pt", model, epoch + 1, embed_dim, in_dim)
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch + 1
            _save_ckpt(out / "best.pt", model, epoch + 1, embed_dim, in_dim)

        if max_steps is not None and step >= max_steps:
            break

    (out / "meta.json").write_text(json.dumps({
        "layer": layer, "feat_root": feat_root, "in_dim": in_dim,
        "embed_dim": embed_dim, "hidden_dim": hidden_dim, "n_heads": n_heads,
        "epochs": epochs, "warm_epochs": warm_epochs, "lr": lr,
        "w_ce": w_ce, "w_sc": w_sc, "lambda_icmm": lambda_icmm,
        "supcon_tau": supcon_tau, "n_synthetic": n_synthetic,
        "icmm_weighting": icmm_weighting, "centroid_mode": centroid_mode,
        "centroid_momentum": centroid_momentum,
        "best_epoch": best_epoch, "best_dev_known_acc": best_acc, "seed": seed,
    }, indent=2), encoding="utf-8")
    logger.info("Best dev-known acc %.4f @ epoch %d -> %s", best_acc, best_epoch, out)
    return out


def _feature_dim(clips) -> int:
    import numpy as np
    with np.load(clips[0].path, allow_pickle=False) as d:
        return int(d["features"].shape[-1])


def _save_ckpt(path, model, epoch, embed_dim, in_dim) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "embed_dim": embed_dim,
        "in_dim": in_dim,
        "n_classes": model.n_classes,
    }, path)


def load_model(ckpt_path, device="cpu"):
    """Reload a trained NovelModel from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta_path = Path(ckpt_path).parent / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    model = NovelModel(
        in_dim=ckpt["in_dim"],
        hidden_dim=meta.get("hidden_dim", 512),
        embed_dim=ckpt["embed_dim"],
        n_heads=meta.get("n_heads", 4),
        n_classes=ckpt["n_classes"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train C-ICMM (CE + SupCon + curriculum ICMM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--feat-root", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--protocol-dir", default=PROTO_DIR_DEFAULT)
    p.add_argument("--embed-dim", type=int, default=256, help="Fine head dim (256 or 512).")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--warm-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--crop-frames", type=int, default=200)
    p.add_argument("--w-ce", type=float, default=0.5)
    p.add_argument("--w-sc", type=float, default=0.5)
    p.add_argument("--lambda-icmm", type=float, default=0.5)
    p.add_argument("--supcon-tau", type=float, default=0.05)
    p.add_argument("--n-synthetic", type=int, default=32)
    p.add_argument("--icmm-weighting", choices=["manual", "auto"], default="manual")
    p.add_argument("--centroid-momentum", type=float, default=0.99)
    p.add_argument("--centroid-mode", choices=["ema", "batch"], default="ema",
                   help="ema: momentum-blended running centroid. "
                        "batch: recomputed fresh from each batch, no history.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    train_cicmm(
        feat_root=args.feat_root, layer=args.layer,
        protocol_dir=args.protocol_dir,
        embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        epochs=args.epochs, warm_epochs=args.warm_epochs,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, crop_frames=args.crop_frames,
        w_ce=args.w_ce, w_sc=args.w_sc, lambda_icmm=args.lambda_icmm,
        supcon_tau=args.supcon_tau,
        n_synthetic=args.n_synthetic, icmm_weighting=args.icmm_weighting,
        centroid_momentum=args.centroid_momentum, centroid_mode=args.centroid_mode,
        num_workers=args.num_workers, device=args.device,
        seed=args.seed, out_dir=args.out_dir, max_steps=args.max_steps,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
