#!/usr/bin/env python3
"""Train the Hier-Spec model and estimate its Mahalanobis statistics.

Pipeline (single invocation):

1. Train ``HierSpecModel`` on cached Wav2Vec2 features with teacher-forced
   routing and the joint objective ``L = 0.5*(L_arch + L_model)``, using
   inverse-frequency oversampling (``WeightedRandomSampler``). AdamW, lr 1e-3,
   ReduceLROnPlateau on a dev closed-set score. Best/last checkpoints saved.
2. After training, embed all train clips (centre crop, eval mode) and estimate
   per-architecture and per-model Gaussian means + a ridge-regularised pooled
   covariance (Cholesky factor) at each level -> ``stats.npz``.

Run from the ``hier_spec/`` directory (``import common`` / ``import model`` are
sibling modules, mirroring ``gmm_baseline/``).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import taxonomy as tax
from common import (HierSpecConfig, add_repo_to_path, configure_logging,
                    estimate_gaussian_stats, resolve_device, seed_everything)
from data import HierSpecDataset, collate, inverse_frequency_weights

logger = logging.getLogger("hier_spec.train")


def _load_clips(protocol_dir: str, split: str, feat_root: str, layer: int,
                label_map, subsplit: Optional[str] = None):
    add_repo_to_path()
    from protocols_mlaad import load_split

    return load_split(protocol_dir, split, subsplit=subsplit,
                      feat_root=feat_root, layer=layer, label_map=label_map)


def evaluate_closed_set(model, loader, device) -> Dict[str, float]:
    """Closed-set arch and model accuracy on known dev clips (teacher-free routing)."""
    import torch

    model.eval()
    arch_correct = arch_total = 0
    model_correct = model_total = 0
    multi_idx = model._multi_arch_idx
    with torch.no_grad():
        for feats, arch_lab, model_local, _flat in loader:
            feats = feats.to(device, non_blocking=True)
            arch_logits, model_logits, _ = model.predict_logits(feats)
            arch_pred = arch_logits.argmax(1).cpu()
            known = arch_lab >= 0
            arch_correct += int(((arch_pred == arch_lab) & known).sum())
            arch_total += int(known.sum())
            # model acc: route by predicted arch; singleton preds are trivially
            # correct at model level, so score only multi-model-arch GT samples
            for a_idx, arch in multi_idx.items():
                m = (arch_lab == a_idx)
                if not bool(m.any()):
                    continue
                pred_local = model_logits[arch].argmax(1).cpu()[m]
                model_correct += int((pred_local == model_local[m]).sum())
                model_total += int(m.sum())
    return {
        "arch_acc": arch_correct / arch_total if arch_total else float("nan"),
        "model_acc": model_correct / model_total if model_total else float("nan"),
    }


def estimate_and_save_stats(model, clips, cfg: HierSpecConfig, device,
                            arch_label_map, label_map, out_path: Path,
                            batch_size: int, num_workers: int) -> None:
    import torch
    from torch.utils.data import DataLoader

    ds = HierSpecDataset(clips, max_frames=cfg.max_frames, train=False,
                         arch_label_map=arch_label_map,
                         model_label_maps=tax.model_label_maps(), seed=cfg.seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate)
    model.eval()
    embs: List[np.ndarray] = []
    arch_labels: List[np.ndarray] = []
    flat_labels: List[np.ndarray] = []
    with torch.no_grad():
        for feats, arch_lab, _model_local, flat in loader:
            x = model.embed(feats.to(device, non_blocking=True))
            embs.append(x.cpu().numpy())
            arch_labels.append(arch_lab.numpy())
            flat_labels.append(flat.numpy())
    emb = np.concatenate(embs).astype(np.float64)
    arch_lab = np.concatenate(arch_labels)
    flat_lab = np.concatenate(flat_labels)

    arch_means, arch_chol = estimate_gaussian_stats(
        emb, arch_lab, len(arch_label_map), cfg.ridge_lambda)
    model_means, model_chol = estimate_gaussian_stats(
        emb, flat_lab, len(label_map), cfg.ridge_lambda)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, arch_means=arch_means, arch_chol=arch_chol,
             model_means=model_means, model_chol=model_chol,
             ridge_lambda=np.asarray(cfg.ridge_lambda))
    logger.info("Saved Mahalanobis stats -> %s (arch %d, model %d classes)",
                out_path, len(arch_label_map), len(label_map))


def train(args) -> None:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from model import HierSpecModel

    cfg = HierSpecConfig(max_frames=args.max_frames, arcface_s=args.arcface_s,
                         arcface_m=args.arcface_m, ridge_lambda=args.ridge_lambda,
                         seed=args.seed)
    seed_everything(cfg.seed)
    device = resolve_device(args.device)
    logger.info("Device: %s", device)

    add_repo_to_path()
    from protocols_mlaad import build_label_map

    label_map = build_label_map(args.protocol_dir)        # 24 known models
    arch_label_map = tax.arch_label_map()                 # 13 archs
    model_label_maps = tax.model_label_maps()

    train_clips = _load_clips(args.protocol_dir, "train", args.feat_root,
                              cfg.layer, label_map)
    dev_clips = _load_clips(args.protocol_dir, "dev", args.feat_root,
                            cfg.layer, label_map)
    dev_known = [c for c in dev_clips if c.is_known]
    if args.limit:
        train_clips = train_clips[:args.limit]
        dev_known = dev_known[:max(1, args.limit // 4)]
    logger.info("Train clips: %d | dev known clips: %d", len(train_clips), len(dev_known))

    train_ds = HierSpecDataset(train_clips, max_frames=cfg.max_frames, train=True,
                               arch_label_map=arch_label_map,
                               model_label_maps=model_label_maps, seed=cfg.seed)
    weights = inverse_frequency_weights(train_clips)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                    num_samples=len(train_clips), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, collate_fn=collate,
                              drop_last=True, pin_memory=(device.type == "cuda"))

    dev_ds = HierSpecDataset(dev_known, max_frames=cfg.max_frames, train=False,
                             arch_label_map=arch_label_map,
                             model_label_maps=model_label_maps, seed=cfg.seed)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)

    model = HierSpecModel(arch_label_map, model_label_maps, tax.MULTI_MODEL_ARCHS,
                          feat_dim=cfg.feat_dim, s=cfg.arcface_s, m=cfg.arcface_m
                          ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=args.plateau_patience)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_meta = {
        "config": cfg.to_dict(),
        "arch_label_map": arch_label_map,
        "model_label_maps": model_label_maps,
        "label_map": label_map,
        "multi_model_archs": list(tax.MULTI_MODEL_ARCHS),
    }
    (out_dir / "meta.json").write_text(json.dumps(ckpt_meta, indent=2))

    best_score = -float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for feats, arch_lab, model_local, _flat in train_loader:
            feats = feats.to(device, non_blocking=True)
            arch_lab = arch_lab.to(device, non_blocking=True)
            model_local = model_local.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                l_arch, l_model, _ = model(feats, arch_lab, model_local)
                loss = 0.5 * (l_arch + l_model)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += loss.item()
            n_batches += 1
        train_loss = running / max(n_batches, 1)

        dev_metrics = evaluate_closed_set(model, dev_loader, device)
        score = np.nanmean([dev_metrics["arch_acc"], dev_metrics["model_acc"]])
        scheduler.step(score)
        logger.info("epoch %d/%d  loss %.4f  dev arch %.4f model %.4f  lr %.2e  (%.1fs)",
                    epoch, args.epochs, train_loss, dev_metrics["arch_acc"],
                    dev_metrics["model_acc"], optim.param_groups[0]["lr"],
                    time.time() - t0)

        torch.save({"model": model.state_dict(), **ckpt_meta,
                    "epoch": epoch, "dev_score": float(score)},
                   out_dir / "last.pt")
        # Save best whenever the score improves; also always seed best.pt on the
        # first epoch so a checkpoint exists even if dev has no known clips
        # (score == nan, e.g. a tiny smoke split) and nan never "improves".
        improved = np.isfinite(score) and score > best_score
        if epoch == 1 or improved:
            if improved:
                best_score = score
            torch.save({"model": model.state_dict(), **ckpt_meta,
                        "epoch": epoch, "dev_score": float(score)},
                       out_dir / "best.pt")
            logger.info("  saved best.pt (dev score %.4f)", score)

    # --- Mahalanobis statistics on the best checkpoint ---------------------- #
    best = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    estimate_and_save_stats(model, train_clips, cfg, device, arch_label_map,
                            label_map, out_dir / "stats.npz",
                            args.batch_size, args.num_workers)


def build_arg_parser() -> argparse.ArgumentParser:
    cfg = HierSpecConfig()
    p = argparse.ArgumentParser(
        description="Train Hier-Spec and estimate Mahalanobis statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--feat-root", required=True,
                   help="Feature root from extract_feats.py (has layer_05/).")
    p.add_argument("--protocol-dir", default="../mlaad4sourcetracing")
    p.add_argument("--out-dir", default="artifacts")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--plateau-patience", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=cfg.max_frames)
    p.add_argument("--arcface-s", type=float, default=cfg.arcface_s)
    p.add_argument("--arcface-m", type=float, default=cfg.arcface_m)
    p.add_argument("--ridge-lambda", type=float, default=cfg.ridge_lambda)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--limit", type=int, default=None, help="Smoke-test clip cap.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
