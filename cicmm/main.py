#!/usr/bin/env python3
"""Full C-ICMM pipeline: train -> embed -> GMM fit -> eval.

All experiment dimensions are CLI flags so any combination can be launched::

    # Baseline (matches the C-ICMM document)
    python main.py --feat-root ../feats_xlsr --layer 5

    # Trial: 512-D embeddings + auto ICMM + adaptive GMM
    python main.py --feat-root ../feats_xlsr --layer 5 \\
        --embed-dim 512 --icmm-weighting auto --gmm-covariance adaptive

See ``cicmm_run.sh`` at repo root for all six planned trial commands.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from common import configure_logging, setup_paths

setup_paths()

logger = logging.getLogger("ser.cicmm.main")

PROTO_DIR_DEFAULT = str(Path(__file__).resolve().parent.parent / "mlaad4sourcetracing")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the full C-ICMM pipeline (train -> embed -> GMM -> eval).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    p.add_argument("--feat-root", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--protocol-dir", default=PROTO_DIR_DEFAULT)
    p.add_argument("--out-dir", default=None,
                   help="Artifacts dir; auto-named from config if omitted.")

    # Model
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=4)

    # Training
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--warm-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--crop-frames", type=int, default=200)

    # Losses
    p.add_argument("--w-ce", type=float, default=0.5)
    p.add_argument("--w-sc", type=float, default=0.5)
    p.add_argument("--lambda-icmm", type=float, default=0.5)
    p.add_argument("--supcon-tau", type=float, default=0.05)

    # ICMM
    p.add_argument("--n-synthetic", type=int, default=32)
    p.add_argument("--icmm-weighting", choices=["manual", "auto"], default="manual")
    p.add_argument("--centroid-momentum", type=float, default=0.99)
    p.add_argument("--centroid-mode", choices=["ema", "batch"], default="ema",
                   help="ema: momentum-blended running centroid. "
                        "batch: recomputed fresh from each batch, no history.")

    # GMM
    p.add_argument("--gmm-components", type=int, default=5)
    p.add_argument("--gmm-covariance", choices=["full", "adaptive"], default="full")
    p.add_argument("--gmm-min-samples-full", type=int, default=300)
    p.add_argument("--omega-s", type=float, default=0.08)
    p.add_argument("--no-grid-search", action="store_true")

    # System
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("-v", "--verbose", action="store_true")

    # Stage selection
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training; reuse existing checkpoint.")
    p.add_argument("--skip-embed", action="store_true",
                   help="Skip embedding; reuse existing emb_*.npz.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)

    out = args.out_dir or (
        f"artifacts/cicmm_e{args.embed_dim}_{args.icmm_weighting}_{args.gmm_covariance}"
        + ("" if args.centroid_mode == "ema" else f"_{args.centroid_mode}")
    )

    # Stage 1: Train
    if not args.skip_train:
        from train import train_cicmm
        logger.info("=== Stage 1: Training ===")
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
            seed=args.seed, out_dir=out, max_steps=args.max_steps,
        )

    # Stage 2: Embed
    if not args.skip_embed:
        from embed import run as embed_run
        logger.info("=== Stage 2: Embedding ===")
        embed_run(
            feat_root=args.feat_root, layer=args.layer,
            artifacts_dir=out, protocol_dir=args.protocol_dir,
            device=args.device, seed=args.seed,
        )

    # Stage 3: GMM fit + calibration
    from gmm_fit import run as gmm_run
    logger.info("=== Stage 3: GMM fit + calibration ===")
    gmm_run(
        artifacts_dir=out,
        gmm_components=args.gmm_components,
        gmm_covariance=args.gmm_covariance,
        gmm_min_samples_full=args.gmm_min_samples_full,
        omega_s=args.omega_s,
        grid_search=not args.no_grid_search,
        seed=args.seed,
    )

    # Stage 4: Eval
    from eval import run as eval_run
    logger.info("=== Stage 4: Evaluation ===")
    results = eval_run(
        artifacts_dir=out, feat_root=args.feat_root,
        layer=args.layer, protocol_dir=args.protocol_dir,
    )

    logger.info("=== Done.  MacroF1 = %.4f ===", results["classification"]["macro_f1"])
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
