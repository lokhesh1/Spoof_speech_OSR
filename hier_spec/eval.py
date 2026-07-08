#!/usr/bin/env python3
"""Evaluate Hier-Spec with full two-stage sequential OOD gating.

Per clip: embed -> Stage-1 arch gate (reject if min-arch Mahalanobis > tau_arch)
-> arch prediction from the ArcFace architecture head -> singleton bypass, or
Stage-2 model gate (reject if min-member Mahalanobis > tau_model) -> model
prediction from that architecture's ArcFace model head.

Reports flat open-set metrics using ``gmm_baseline/common.py:compute_metrics``
(identical definitions, so numbers are directly comparable to the GMM baseline),
plus architecture-level open-set metrics and the fraction of model-level errors
attributable to architecture misrouting (the spec's ~56% figure). Optional
``--fine`` breaks results down by the four seen/unseen subsplits.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import taxonomy as tax
from common import add_repo_to_path, configure_logging, resolve_device
from hier_infer import embed_clips, load_bundle, load_split_clips

logger = logging.getLogger("hier_spec.eval")

UNKNOWN_LABEL = -1


def _gmm_metrics():
    """Load compute_metrics/format_metrics from the GMM baseline (file import)."""
    repo = Path(__file__).resolve().parent.parent
    path = repo / "gmm_baseline" / "common.py"
    spec = importlib.util.spec_from_file_location("gmm_baseline_common", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass/typing machinery can resolve
    # cls.__module__ via sys.modules (Python 3.13 requires this).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.compute_metrics, mod.format_metrics


def decide(bundle, emb, arch_pred, model_pred_local, tau_arch, tau_model):
    """Run the sequential gate for every clip.

    Returns a dict of per-clip arrays: gated model-axis prediction, gate-free
    (closed-set) prediction, predicted architecture name, and the two stage
    scores / pass flags.
    """
    n = len(emb)
    s1 = bundle.stage1_scores(emb)                 # (N, 13)
    s1_min = s1.min(axis=1)

    pred_flat = np.full(n, UNKNOWN_LABEL, dtype=int)
    pred_known = np.zeros(n, dtype=bool)
    pred_arch_name = np.array(["__reject__"] * n, dtype=object)
    argmax_flat = np.full(n, UNKNOWN_LABEL, dtype=int)      # gate-free closed set
    argmax_arch_name = np.empty(n, dtype=object)
    s2_min = np.full(n, np.nan)
    stage1_pass = s1_min <= tau_arch

    singleton = bundle.singleton_model                     # arch -> model name
    for i in range(n):
        a_name = bundle.idx_to_arch[int(arch_pred[i])]
        argmax_arch_name[i] = a_name
        # gate-free closed-set model prediction
        if a_name in singleton:
            argmax_flat[i] = bundle.label_map[singleton[a_name]]
        else:
            local = int(model_pred_local[a_name][i])
            argmax_flat[i] = bundle.label_map[bundle.arch_member_names[a_name][local]]

        if not stage1_pass[i]:
            continue                                       # rejected at Stage 1
        pred_arch_name[i] = a_name
        if a_name in singleton:                            # singleton bypass
            pred_flat[i] = bundle.label_map[singleton[a_name]]
            pred_known[i] = True
        else:
            s2 = bundle.stage2_scores(emb[i:i + 1], a_name).min(axis=1)[0]
            s2_min[i] = s2
            if s2 <= tau_model:
                local = int(model_pred_local[a_name][i])
                pred_flat[i] = bundle.label_map[bundle.arch_member_names[a_name][local]]
                pred_known[i] = True
            # else: rejected at Stage 2 -> stays UNKNOWN
    return {
        "pred_flat": pred_flat, "pred_known": pred_known,
        "pred_arch_name": pred_arch_name, "argmax_flat": argmax_flat,
        "argmax_arch_name": argmax_arch_name,
        "s1_min": s1_min, "s2_min": s2_min, "stage1_pass": stage1_pass,
    }


def evaluate(args) -> None:
    device = resolve_device(args.device)
    bundle = load_bundle(Path(args.artifacts), device, checkpoint=args.checkpoint)
    compute_metrics, format_metrics = _gmm_metrics()

    thr = json.loads((Path(args.artifacts) / "thresholds.json").read_text())
    tau_arch, tau_model = thr["tau_arch"], thr["tau_model"]
    logger.info("Thresholds: tau_arch=%.4f  tau_model=%.4f", tau_arch, tau_model)

    clips = load_split_clips(args.protocol_dir, args.split, args.feat_root,
                             bundle.cfg.layer, bundle.label_map)
    if args.limit:
        clips = clips[:args.limit]
    if not clips:
        logger.error("No %s clips resolved under feat-root %s (nothing to "
                     "evaluate). Extract features for this split first.",
                     args.split, args.feat_root)
        return
    emb, arch_pred, model_pred_local, metas = embed_clips(
        bundle, clips, batch_size=args.batch_size, num_workers=args.num_workers)
    logger.info("Embedded %d %s clips.", len(emb), args.split)

    dec = decide(bundle, emb, arch_pred, model_pred_local, tau_arch, tau_model)

    # -- ground-truth arrays (model axis + arch axis) ----------------------- #
    arch_label_map = bundle.arch_label_map
    known_archs = set(tax.known_archs())
    true_flat = np.array([c.label for c in clips])
    true_known = np.array([c.is_known for c in clips])
    true_arch_name = np.array([tax.derive_arch(c.model_name) for c in clips], dtype=object)
    true_arch_known = np.array([a in known_archs for a in true_arch_name])
    true_arch_label = np.array([arch_label_map[a] if k else UNKNOWN_LABEL
                                for a, k in zip(true_arch_name, true_arch_known)])
    pred_arch_label = np.array([arch_label_map[a] if a in arch_label_map else UNKNOWN_LABEL
                                for a in dec["pred_arch_name"]])
    argmax_arch_label = np.array([arch_label_map[a] for a in dec["argmax_arch_name"]])

    results: Dict[str, dict] = {}

    def model_axis(mask=None):
        idx = slice(None) if mask is None else mask
        return compute_metrics(true_flat[idx], true_known[idx],
                               dec["pred_flat"][idx], dec["pred_known"][idx],
                               dec["argmax_flat"][idx])

    def arch_axis(mask=None):
        idx = slice(None) if mask is None else mask
        return compute_metrics(true_arch_label[idx], true_arch_known[idx],
                               pred_arch_label[idx], dec["stage1_pass"][idx],
                               argmax_arch_label[idx])

    results["overall_model"] = model_axis()
    results["overall_arch"] = arch_axis()
    print(format_metrics(f"{args.split} | model axis", results["overall_model"]))
    print(format_metrics(f"{args.split} | arch axis", results["overall_arch"]))

    # error decomposition: of gate-free model-level errors on known clips, what
    # fraction also mispredicted the architecture (spec ~56%).
    km = true_known
    model_wrong = km & (dec["argmax_flat"] != true_flat)
    arch_wrong = argmax_arch_label != true_arch_label
    n_model_wrong = int(model_wrong.sum())
    frac = float((arch_wrong & model_wrong).sum()) / n_model_wrong if n_model_wrong else float("nan")
    results["error_decomposition"] = {
        "n_known": int(km.sum()),
        "n_model_errors": n_model_wrong,
        "frac_model_errors_from_arch_misrouting": frac,
    }
    logger.info("Model errors from arch misrouting: %.1f%% (%d of %d known model errors)",
                100 * frac, int((arch_wrong & model_wrong).sum()), n_model_wrong)

    # -- fine subsplits ----------------------------------------------------- #
    if args.fine and args.split in ("dev", "eval"):
        add_repo_to_path()
        from protocols_mlaad import SUBSPLITS

        rel_index = {c.rel: i for i, c in enumerate(clips)}
        for sub in SUBSPLITS:
            sub_clips = load_split_clips(args.protocol_dir, args.split,
                                         args.feat_root, bundle.cfg.layer,
                                         bundle.label_map, subsplit=sub)
            mask = np.array([rel_index[c.rel] for c in sub_clips
                             if c.rel in rel_index], dtype=int)
            if len(mask) == 0:
                continue
            results[f"fine/{sub}/model"] = model_axis(mask)
            print(format_metrics(f"{args.split}/{sub} | model axis",
                                 results[f"fine/{sub}/model"]))

    # -- write outputs ------------------------------------------------------ #
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(results, indent=2))
        logger.info("Wrote metrics -> %s", args.out_json)

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        inv_label = {v: k for k, v in bundle.label_map.items()}
        with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["rel", "language", "true_model", "true_label",
                        "true_is_known", "true_arch", "pred_label", "pred_model",
                        "pred_is_known", "pred_arch", "s1_min", "s2_min",
                        "stage1_pass"])
            for i, c in enumerate(clips):
                pf = int(dec["pred_flat"][i])
                w.writerow([
                    c.rel, c.language, c.model_name, c.label, c.is_known,
                    true_arch_name[i], pf,
                    inv_label.get(pf, "UNKNOWN"), bool(dec["pred_known"][i]),
                    dec["pred_arch_name"][i], f"{dec['s1_min'][i]:.4f}",
                    "" if np.isnan(dec["s2_min"][i]) else f"{dec['s2_min'][i]:.4f}",
                    bool(dec["stage1_pass"][i]),
                ])
        logger.info("Wrote per-clip predictions -> %s", args.out_csv)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate Hier-Spec with two-stage OOD gating.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--feat-root", required=True)
    p.add_argument("--protocol-dir", default="../mlaad4sourcetracing")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--split", default="eval", choices=["dev", "eval"])
    p.add_argument("--fine", action="store_true",
                   help="Also report the four seen/unseen subsplits.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
