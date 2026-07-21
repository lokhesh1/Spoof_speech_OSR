#!/usr/bin/env python3
"""Held-out evaluation of the C-ICMM + GMM pipeline.

Reports:
    * MacroF1 (K+1 classes: 24 known + unknown).
    * Known-class top-1 attribution accuracy.
    * Open-set detection AUROC (known vs unknown).
    * Per-quadrant (lang_seen x model_seen) detection AUROC.
    * Per-class F1 breakdown.
    * Unknown rejection rate and known acceptance rate.

Run from ``cicmm/``::

    python eval.py --artifacts-dir artifacts/cicmm_e256_manual \\
                   --feat-root ../feats_xlsr --layer 5
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import joblib
import numpy as np
from sklearn.metrics import (classification_report, f1_score,
                             roc_auc_score, roc_curve)

from common import configure_logging, setup_paths

setup_paths()

from protocols_mlaad import SUBSPLITS, load_split, build_label_map

from gmm_fit import N_CLASSES, predict_with_rejection, score_nll
from train import PROTO_DIR_DEFAULT

logger = logging.getLogger("ser.cicmm.eval")


# ---------------------------------------------------------------------- #
# Metrics
# ---------------------------------------------------------------------- #
def _auroc(score: np.ndarray, known: np.ndarray) -> float:
    if len(np.unique(known)) < 2:
        return float("nan")
    return float(roc_auc_score(known, score))


def _eer(score: np.ndarray, known: np.ndarray) -> float:
    """Equal Error Rate on the known-vs-unknown detection score.

    FPR (known mistaken for unknown) and FNR=1-TPR (unknown mistaken for
    known) are monotone in opposite directions along the ROC threshold
    sweep, so the crossover point (nearest sampled threshold) is the EER.
    """
    if len(np.unique(known)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(known, score)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def detection_metrics(nll: np.ndarray, known: np.ndarray) -> Dict[str, float]:
    """AUROC + EER for known-vs-unknown using minimum NLL as the known-high score."""
    score = -nll.min(axis=1)
    return {"auroc": _auroc(score, known),
            "eer": _eer(score, known),
            "n_known": int(known.sum()),
            "n_unknown": int((~known).sum())}


def classification_metrics(
    predictions: np.ndarray, labels: np.ndarray, is_known: np.ndarray,
) -> Dict[str, float]:
    """MacroF1, known accuracy, unknown rejection rate, unknown-class F1.

    ``f1_unknown`` is the F1 of the single "unknown" class within the K+1
    scheme (predicted -> class ``N_CLASSES`` iff rejected by the gate),
    i.e. the per-class F1 sklearn would report for that label alone -- not
    to be confused with ``macro_f1`` which averages over all 25 classes.
    """
    true_mapped = np.where(is_known, labels, N_CLASSES)
    pred_mapped = np.where(predictions >= 0, predictions, N_CLASSES)
    macro_f1 = f1_score(true_mapped, pred_mapped, average="macro", zero_division=0)
    f1_unknown = f1_score(true_mapped, pred_mapped, labels=[N_CLASSES],
                          average="micro", zero_division=0)

    known_mask = is_known
    unknown_mask = ~is_known
    known_correct = (predictions[known_mask] == labels[known_mask]).sum() if known_mask.any() else 0
    known_accepted = (predictions[known_mask] >= 0).sum() if known_mask.any() else 0
    unknown_rejected = (predictions[unknown_mask] == -1).sum() if unknown_mask.any() else 0

    nk = int(known_mask.sum())
    nu = int(unknown_mask.sum())
    return {
        "macro_f1": float(macro_f1),
        "f1_unknown": float(f1_unknown),
        "known_top1": float(known_correct / max(nk, 1)),
        "known_accept_rate": float(known_accepted / max(nk, 1)),
        "unknown_reject_rate": float(unknown_rejected / max(nu, 1)),
    }


# ---------------------------------------------------------------------- #
# Per-quadrant evaluation
# ---------------------------------------------------------------------- #
def _subsplit_rels(protocol_dir, feat_root, layer, subsplit) -> set:
    clips = load_split(protocol_dir, "eval", subsplit=subsplit,
                       feat_root=feat_root, layer=layer)
    return {c.rel for c in clips}


# ---------------------------------------------------------------------- #
# Driver
# ---------------------------------------------------------------------- #
def run(
    *,
    artifacts_dir: str,
    feat_root: str,
    layer: int,
    protocol_dir: str = PROTO_DIR_DEFAULT,
) -> Dict:
    art = Path(artifacts_dir)

    bundle = joblib.load(art / "gmm_bundle.pkl")
    gmms = bundle["gmms"]
    thresholds = bundle["thresholds"]

    ev = np.load(art / "emb_eval.npz", allow_pickle=True)
    emb = ev["emb"]
    labels = ev["labels"]
    is_known = ev["is_known"].astype(bool)
    rels = [str(r) for r in ev["rels"]]

    nll = score_nll(gmms, emb)
    predictions = predict_with_rejection(nll, thresholds)

    results: Dict = {
        "gmm_covariance": bundle["gmm_covariance"],
        "gmm_components": bundle["gmm_components"],
        "best_coverage": bundle["best_coverage"],
        "layer": layer, "n_eval": len(emb),
    }

    results["detection"] = detection_metrics(nll, is_known)
    results["classification"] = classification_metrics(predictions, labels, is_known)

    # Per-class F1 (known classes only)
    known_idx = is_known
    if known_idx.any():
        label_map = build_label_map(protocol_dir)
        idx_to_model = {v: k for k, v in label_map.items()}
        class_names = [idx_to_model.get(i, f"class_{i}") for i in range(N_CLASSES)]
        kr = classification_report(
            labels[known_idx], predictions[known_idx],
            labels=list(range(N_CLASSES)),
            target_names=class_names,
            output_dict=True, zero_division=0,
        )
        results["per_class_f1"] = {
            name: round(kr[name]["f1-score"], 4) for name in class_names if name in kr
        }

    # Per-quadrant detection AUROC
    rel_set = set(rels)
    idx = {r: i for i, r in enumerate(rels)}
    quad = {}
    for sub in SUBSPLITS:
        sub_file = sub.replace("__", "___")
        try:
            qrels = _subsplit_rels(protocol_dir, feat_root, layer, sub) & rel_set
        except FileNotFoundError:
            continue
        if not qrels:
            continue
        qi = np.array([idx[r] for r in qrels])
        qnll = score_nll(gmms, emb[qi])
        qknown = is_known[qi]
        qpred = predict_with_rejection(qnll, thresholds)
        quad[sub] = {
            **detection_metrics(qnll, qknown),
            **classification_metrics(qpred, labels[qi], qknown),
        }
    results["quadrants"] = quad

    # Logit-based attribution accuracy (closed-set, no rejection)
    logits = ev["logits"]
    logit_pred = logits.argmax(axis=1)
    known_logit_acc = float((logit_pred[is_known] == labels[is_known]).mean()) if is_known.any() else float("nan")
    results["logit_top1_known"] = known_logit_acc

    _save_and_print(art, results)
    return results


def _save_and_print(art: Path, results: Dict) -> None:
    out = Path("results")
    out.mkdir(exist_ok=True)
    # Keyed off the artifacts dir name (unique per trial config), not just
    # gmm_covariance + layer -- multiple trials can share both (e.g. two
    # ICMM-weighting variants at full-cov), which previously overwrote
    # each other's result file.
    fname = out / f"{art.name}.json"
    fname.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    cls = results["classification"]
    det = results["detection"]
    logger.info("=== C-ICMM %s | layer %d | cov=%s | n=%d ===",
                art.name, results["layer"], results["gmm_covariance"],
                results["n_eval"])
    logger.info("MacroF1 (K+1) %.4f | F1(unknown) %.4f | Known-top1 %.4f | "
                "Accept %.4f | Unk-reject %.4f",
                cls["macro_f1"], cls["f1_unknown"], cls["known_top1"],
                cls["known_accept_rate"], cls["unknown_reject_rate"])
    logger.info("Detection AUROC %.4f | EER %.4f (n_known=%d, n_unknown=%d)",
                det["auroc"], det["eer"], det["n_known"], det["n_unknown"])
    logger.info("Logit top1 (known, no rejection) %.4f", results["logit_top1_known"])

    if results.get("quadrants"):
        logger.info("Quadrants:")
        for sub, m in results["quadrants"].items():
            logger.info("  %-40s AUROC %.4f  EER %.4f  MacroF1 %.4f",
                        sub, m["auroc"], m["eer"], m["macro_f1"])

    logger.info("Saved -> %s", fname)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate C-ICMM + GMM on the held-out eval set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--artifacts-dir", required=True)
    p.add_argument("--feat-root", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--protocol-dir", default=PROTO_DIR_DEFAULT)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    run(artifacts_dir=args.artifacts_dir, feat_root=args.feat_root,
        layer=args.layer, protocol_dir=args.protocol_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
