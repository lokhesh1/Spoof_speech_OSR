#!/usr/bin/env python3
"""Calibrate the two sequential OOD thresholds on the dev split (EER point).

* ``tau_arch``: Stage-1 gate. Score = min architecture Mahalanobis distance.
  Positive (should reject) = dev clip whose architecture family was not in
  training (``derive_arch`` -> unknown arch). EER over all dev clips.
* ``tau_model``: Stage-2 gate. Among dev clips whose GT architecture is a known
  **multi-model** arch, score = min Mahalanobis distance to that arch's known
  member models; positive = an unseen model of that arch. EER over that subset.

Thresholds (accept iff score <= tau) plus gate EER/AUROC are written to
``thresholds.json`` in the artifacts dir.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import taxonomy as tax
from common import auroc, configure_logging, eer_threshold, resolve_device
from hier_infer import embed_clips, load_bundle, load_split_clips

logger = logging.getLogger("hier_spec.calibrate")


def calibrate(args) -> None:
    device = resolve_device(args.device)
    bundle = load_bundle(Path(args.artifacts), device, checkpoint=args.checkpoint)

    clips = load_split_clips(args.protocol_dir, args.split, args.feat_root,
                             bundle.cfg.layer, bundle.label_map)
    if args.limit:
        clips = clips[:args.limit]
    emb, _arch_pred, _model_pred, metas = embed_clips(
        bundle, clips, batch_size=args.batch_size, num_workers=args.num_workers)
    logger.info("Embedded %d %s clips.", len(emb), args.split)

    known_archs = set(tax.known_archs())
    gt_arch = np.array([tax.derive_arch(m["model_name"]) for m in metas])

    # -- Stage 1: architecture gate ----------------------------------------- #
    s1 = bundle.stage1_scores(emb).min(axis=1)
    is_ood_arch = np.array([a not in known_archs for a in gt_arch])
    tau_arch, eer_arch = eer_threshold(s1, is_ood_arch)
    auroc_arch = auroc(s1, is_ood_arch)
    logger.info("Stage-1 arch gate: tau=%.4f  EER=%.4f  AUROC=%.4f  (OOD %d/%d)",
                tau_arch, eer_arch, auroc_arch, int(is_ood_arch.sum()), len(s1))

    # -- Stage 2: model gate over known multi-model archs ------------------- #
    known_models = set(bundle.label_map)
    s2 = np.full(len(emb), np.nan)
    in_scope = np.zeros(len(emb), dtype=bool)
    is_ood_model = np.zeros(len(emb), dtype=bool)
    for arch in bundle.multi_model_archs:
        sel = gt_arch == arch
        if not np.any(sel):
            continue
        s2[sel] = bundle.stage2_scores(emb[sel], arch).min(axis=1)
        in_scope |= sel
    for i, m in enumerate(metas):
        if in_scope[i]:
            is_ood_model[i] = m["model_name"] not in known_models
    if in_scope.any():
        tau_model, eer_model = eer_threshold(s2[in_scope], is_ood_model[in_scope])
        auroc_model = auroc(s2[in_scope], is_ood_model[in_scope])
    else:
        tau_model, eer_model, auroc_model = float("nan"), float("nan"), float("nan")
    logger.info("Stage-2 model gate: tau=%.4f  EER=%.4f  AUROC=%.4f  (OOD %d/%d in-scope)",
                tau_model, eer_model, auroc_model,
                int(is_ood_model[in_scope].sum()), int(in_scope.sum()))

    out = {
        "split": args.split,
        "tau_arch": tau_arch, "eer_arch": eer_arch, "auroc_arch": auroc_arch,
        "tau_model": tau_model, "eer_model": eer_model, "auroc_model": auroc_model,
        "n_clips": int(len(emb)),
    }
    out_path = Path(args.artifacts) / "thresholds.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Wrote thresholds -> %s", out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Calibrate Hier-Spec Stage-1/Stage-2 OOD thresholds on dev.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--feat-root", required=True)
    p.add_argument("--protocol-dir", default="../mlaad4sourcetracing")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--split", default="dev", choices=["dev", "eval"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    calibrate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
