#!/usr/bin/env python3
"""Held-out evaluation of the two-stage cascade (§9).

Pure array math over the caches from :mod:`embed` and the gate from
:mod:`gate`. The eval split is scored once. Reports, as a JSON + console table:

    * Detection   : AUROC / AUPR known-vs-unknown, for all 43 eval-unknown
                    categories AND the 41 that exclude the dev-overlap models.
    * Joint       : OSCR curve (area + CCR at FPR=0.1).
    * Closed-set  : top-1 on eval-known, unconditioned and conditioned on gate
                    acceptance (what actually reaches Stage 2).
    * Memorization: dev-unknown vs eval-unknown AUROC (a large gap = the gate is
                    fitting the unknown categories it saw).
    * Quadrants   : per-``fine`` (lang_seen x model_seen) detection AUROC.
    * Ablation    : each descriptor alone as the gate score vs the learned combo.

Run from ``osr_gate/`` after :mod:`gate`::

    python eval.py --artifacts-dir artifacts/busemann_layer05 \\
                   --feat-root ../feats_xlsr --layer 5
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from common import add_repo_to_path, configure_logging
from gate import (_auroc, build_descriptors, load_cache, load_gate, load_stats,
                  score_split)
from train import N_CLASSES, PROTO_DIR_DEFAULT

logger = logging.getLogger("ser.osr_gate.eval")

#: numpy>=2.0 renamed trapz -> trapezoid.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

#: Models present in BOTH dev-unknown and eval-unknown (the only dev/eval leak).
OVERLAP_MODELS = {"WhisperSpeech", "tts_models/multilingual/multi-dataset/bark"}

SUBSPLITS = ("lang_seen__model_seen", "lang_seen__model_not_seen",
             "lang_not_seen__model_seen", "lang_not_seen__model_not_seen")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def eer(s: np.ndarray, known: np.ndarray) -> Dict[str, float]:
    """Equal error rate: the operating point where FAR == FRR.

    FAR = unknowns accepted, FRR = knowns rejected, sweeping the threshold on
    ``s`` (known-high). Threshold-free -- independent of the gate's ``s >= 0.5``
    boundary -- so it says how separable the two score distributions are.
    """
    if len(np.unique(known)) != 2:
        return {"eer": float("nan"), "eer_threshold": float("nan")}
    from sklearn.metrics import roc_curve

    far, tpr, thr = roc_curve(known, s)   # far = unknowns accepted at each thr
    frr = 1.0 - tpr                        # knowns rejected
    i = int(np.nanargmin(np.abs(frr - far)))
    return {"eer": float((far[i] + frr[i]) / 2.0), "eer_threshold": float(thr[i])}


def detection(s: np.ndarray, known: np.ndarray) -> Dict[str, float]:
    """AUROC + AUPR + EER (threshold-free) plus pointwise F1/accuracy at
    ``s >= 0.5`` (threshold-dependent), from a known-high score.
    """
    from sklearn.metrics import average_precision_score
    out = {"auroc": _auroc(s, known), "n_known": int(known.sum()),
           "n_unknown": int((~known).sum())}
    if len(np.unique(known)) == 2:
        out["aupr_known"] = float(average_precision_score(known, s))
        out["aupr_unknown"] = float(average_precision_score(~known, -s))
    else:
        out["aupr_known"] = out["aupr_unknown"] = float("nan")
    out.update(eer(s, known))
    out.update(detection_pointwise(s, known))
    return out


def oscr(s: np.ndarray, correct: np.ndarray, known: np.ndarray) -> Dict[str, float]:
    """Open-set classification rate curve summary.

    CCR = fraction of *all* knowns that are both accepted and correctly
    classified; FPR = fraction of unknowns accepted. Sweeps the threshold from
    high to low via a cumulative pass (O(n log n)).
    """
    nk, nu = int(known.sum()), int((~known).sum())
    if nk == 0 or nu == 0:
        return {"au_oscr": float("nan"), "ccr_at_fpr10": float("nan")}
    order = np.argsort(-s, kind="mergesort")
    corr_o = (correct & known)[order]
    un_o = (~known)[order]
    ccr = np.cumsum(corr_o) / nk
    fpr = np.cumsum(un_o) / nu
    ccr = np.concatenate([[0.0], ccr])
    fpr = np.concatenate([[0.0], fpr])
    return {"au_oscr": float(_trapz(ccr, fpr)),
            "ccr_at_fpr10": float(np.interp(0.10, fpr, ccr))}


def _prf(pred_pos: np.ndarray, true_pos: np.ndarray) -> Dict[str, float]:
    """Precision / recall / F1 for one binary class given boolean masks."""
    tp = int((pred_pos & true_pos).sum())
    fp = int((pred_pos & ~true_pos).sum())
    fn = int((~pred_pos & true_pos).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec == prec and rec == rec and (prec + rec) else float("nan"))
    return {"precision": prec, "recall": rec, "f1": f1}


def detection_pointwise(s: np.ndarray, known: np.ndarray) -> Dict[str, float]:
    """Thresholded known-vs-unknown metrics at the gate boundary ``s >= 0.5``.

    Reports precision/recall/F1 for **both** classes plus their macro average
    (the headline -- unweighted, so the unknown majority cannot flatter it),
    alongside plain and balanced accuracy.
    """
    accepted = s >= 0.5                      # predicted known
    rejected = ~accepted                      # predicted unknown
    unk = ~known
    nk, nu = int(known.sum()), int(unk.sum())
    if nk == 0 or nu == 0:
        nan = float("nan")
        return {"f1_unknown": nan, "f1_known": nan, "macro_f1": nan,
                "precision_unknown": nan, "recall_unknown": nan,
                "precision_known": nan, "recall_known": nan,
                "accuracy": nan, "balanced_accuracy": nan, "threshold": 0.5}
    u = _prf(rejected, unk)                   # unknown is the positive class
    k = _prf(accepted, known)                 # known is the positive class
    acc = float(int((accepted & known).sum()) + int((rejected & unk).sum())) / (nk + nu)
    return {"f1_unknown": u["f1"], "f1_known": k["f1"],
            "macro_f1": float(np.mean([u["f1"], k["f1"]])),
            "precision_unknown": u["precision"], "recall_unknown": u["recall"],
            "precision_known": k["precision"], "recall_known": k["recall"],
            "accuracy": acc,
            "balanced_accuracy": 0.5 * (k["recall"] + u["recall"]),
            "threshold": 0.5}


def unknown_model_f1(s: np.ndarray, known: np.ndarray,
                     model_names: np.ndarray) -> Dict[str, float]:
    """Per-unknown-model F1, macro-averaged across the individual TTS models.

    ``detection_pointwise``'s ``f1_unknown`` pools every unknown clip into one
    class, so a handful of easy (high-volume) unknown models can flatter it.
    This instead scores each unseen model against the pooled knowns
    separately -- model ``m``'s clips are the positive class, all known clips
    are the negative class, and every *other* unknown model's clips are
    excluded from that pairing -- then macro-averages F1 across models so
    each one counts equally regardless of its clip count.
    """
    accepted = s >= 0.5
    rejected = ~accepted
    unk = ~known
    models = sorted(set(model_names[unk]) - {""})
    per_model: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []
    for m in models:
        pos = unk & (model_names == m)
        mask = known | pos
        prf = _prf(rejected[mask], pos[mask])
        per_model[m] = prf
        if prf["f1"] == prf["f1"]:
            f1s.append(prf["f1"])
    return {"macro_f1_unknown": float(np.mean(f1s)) if f1s else float("nan"),
            "n_unknown_models": len(models),
            "per_model_f1": {m: v["f1"] for m, v in per_model.items()}}


def balanced_open_set_acc(pred: np.ndarray, labels: np.ndarray, s: np.ndarray,
                          known: np.ndarray, n_classes: int = N_CLASSES) -> Dict[str, float]:
    """Open-set accuracy weighing knowns and unknowns 50/50.

    A known clip counts as correct only if it is **accepted AND attributed to
    its own class** (so this scores the gate and Stage 2 jointly); an unknown
    counts as correct if it is rejected. Per-class recall is macro-averaged over
    the known classes, then averaged 50/50 with the unknown recall -- neither
    the 24-way imbalance nor the known/unknown imbalance can skew it.
    """
    accepted = s >= 0.5
    unk = ~known
    per_class: Dict[str, float] = {}
    recalls: List[float] = []
    for c in range(n_classes):
        m = known & (labels == c)
        if not m.sum():
            continue
        r = float((accepted & (pred == c))[m].mean())
        per_class[str(c)] = r
        recalls.append(r)
    known_macro = float(np.mean(recalls)) if recalls else float("nan")
    unk_recall = float((~accepted)[unk].mean()) if unk.sum() else float("nan")
    return {"balanced_open_set_acc": 0.5 * (known_macro + unk_recall),
            "known_macro_recall": known_macro,
            "unknown_recall": unk_recall,
            "n_classes_seen": len(recalls),
            "per_class_recall": per_class}


def closed_set(pred: np.ndarray, labels: np.ndarray, s: np.ndarray,
               known: np.ndarray) -> Dict[str, float]:
    """Top-1 on eval-known: unconditioned and conditioned on gate acceptance."""
    k = known
    if k.sum() == 0:
        return {"top1": float("nan"), "top1_accepted": float("nan"),
                "accept_rate": float("nan")}
    correct = pred == labels
    accepted = s >= 0.5
    top1 = float(correct[k].mean())
    ka = k & accepted
    top1_acc = float(correct[ka].mean()) if ka.sum() else float("nan")
    return {"top1": top1, "top1_accepted": top1_acc,
            "accept_rate": float(accepted[k].mean()),
            "unknown_reject_rate": float((~accepted[~k]).mean()) if (~k).sum() else float("nan")}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _eval_model_names(protocol_dir, feat_root, layer) -> Dict[str, str]:
    """rel -> model_name for the eval split (to identify overlap / stats)."""
    add_repo_to_path()
    from protocols_mlaad import load_split
    clips = load_split(protocol_dir, "eval", feat_root=feat_root, layer=layer)
    return {c.rel: c.model_name for c in clips}


def _subsplit_rels(protocol_dir, feat_root, layer, subsplit) -> set:
    add_repo_to_path()
    from protocols_mlaad import load_split
    clips = load_split(protocol_dir, "eval", subsplit=subsplit,
                       feat_root=feat_root, layer=layer)
    return {c.rel for c in clips}


def run(
    *,
    artifacts_dir: str,
    feat_root: str,
    layer: int,
    protocol_dir: str = PROTO_DIR_DEFAULT,
) -> Dict:
    art = Path(artifacts_dir)
    bundle = load_gate(art)
    stats = load_stats(art)
    is_busemann = bundle["head"] == "busemann"

    s, labels, known = score_split(art, "eval", bundle)
    cache = load_cache(art, "eval")
    rels = [str(r) for r in cache["rels"]]
    pred = cache["logits"].argmax(1)

    results: Dict = {"head": bundle["head"], "layer": layer,
                     "n_eval": len(s)}

    # Detection: 43-cat (all) and 41-cat (drop dev-overlap unknowns)
    results["detection_43"] = detection(s, known)
    rel2model = _eval_model_names(protocol_dir, feat_root, layer)
    model_names = np.array([rel2model.get(r, "") for r in rels])
    is_overlap = np.isin(model_names, list(OVERLAP_MODELS))
    keep = known | ~is_overlap          # keep all knowns + non-overlap unknowns
    results["detection_41"] = detection(s[keep], known[keep])
    results["unknown_model_f1"] = unknown_model_f1(s, known, model_names)

    # Joint + closed-set
    correct = pred == labels
    results["oscr"] = oscr(s, correct, known)
    results["closed_set"] = closed_set(pred, labels, s, known)
    results["open_set_acc"] = balanced_open_set_acc(pred, labels, s, known)

    # Memorization: dev-unknown vs eval-unknown separability
    s_dk, _, k_dk = score_split(art, "dev_known", bundle)
    s_du, _, k_du = score_split(art, "dev_unknown", bundle)
    dev_s = np.concatenate([s_dk, s_du])
    dev_known = np.concatenate([k_dk, k_du])
    results["memorization"] = {
        "dev_auroc": _auroc(dev_s, dev_known),
        "eval_auroc": results["detection_43"]["auroc"],
        "gap": _auroc(dev_s, dev_known) - results["detection_43"]["auroc"],
    }

    # Per-quadrant detection AUROC
    rel_set = set(rels)
    idx = {r: i for i, r in enumerate(rels)}
    quad = {}
    for sub in SUBSPLITS:
        try:
            qrels = _subsplit_rels(protocol_dir, feat_root, layer, sub) & rel_set
        except FileNotFoundError:
            continue
        if not qrels:
            continue
        qi = np.array([idx[r] for r in qrels])
        quad[sub] = {**detection(s[qi], known[qi])}
    results["quadrants"] = quad

    # Single-descriptor ablation (each column alone vs learned gate)
    X, names = build_descriptors(cache, stats, is_busemann)
    ablation = {names[j]: _auroc(X[:, j], known) for j in range(len(names))}
    ablation["learned_gate"] = results["detection_43"]["auroc"]
    results["ablation"] = ablation

    _save_and_print(art, results)
    return results


def _save_and_print(art: Path, results: Dict) -> None:
    out = Path("results")
    out.mkdir(exist_ok=True)
    fname = out / f"{results['head']}_layer{results['layer']:02d}.json"
    fname.write_text(json.dumps(results, indent=2), encoding="utf-8")

    d43, d41 = results["detection_43"], results["detection_41"]
    cs, mem = results["closed_set"], results["memorization"]
    dp = d43
    logger.info("=== %s layer %d (n=%d) ===", results["head"],
                results["layer"], results["n_eval"])
    osa = results["open_set_acc"]
    logger.info("Detection  AUROC 43-cat %.4f | 41-cat %.4f | AUPR(unk) %.4f",
                d43["auroc"], d41["auroc"], d43["aupr_unknown"])
    logger.info("           EER 43-cat %.4f | 41-cat %.4f",
                d43["eer"], d41["eer"])
    logger.info("@s>=0.5    macro-F1 %.4f (known %.4f, unk %.4f) | "
                "acc %.4f | bal-acc %.4f", dp["macro_f1"], dp["f1_known"],
                dp["f1_unknown"], dp["accuracy"], dp["balanced_accuracy"])
    umf1 = results["unknown_model_f1"]
    logger.info("Per-model  macro-F1(unknown) %.4f over %d unseen models",
                umf1["macro_f1_unknown"], umf1["n_unknown_models"])
    logger.info("Open-set   bal-OSA %.4f (known-macro %.4f, unk-recall %.4f)",
                osa["balanced_open_set_acc"], osa["known_macro_recall"],
                osa["unknown_recall"])
    logger.info("OSCR       AU %.4f | CCR@FPR0.1 %.4f",
                results["oscr"]["au_oscr"], results["oscr"]["ccr_at_fpr10"])
    logger.info("Closed-set top1 %.4f | top1|accepted %.4f | accept %.4f | "
                "unk-reject %.4f", cs["top1"], cs["top1_accepted"],
                cs["accept_rate"], cs["unknown_reject_rate"])
    logger.info("Memorization dev %.4f vs eval %.4f (gap %.4f)",
                mem["dev_auroc"], mem["eval_auroc"], mem["gap"])
    best_abl = sorted(results["ablation"].items(), key=lambda kv: -kv[1])[:3]
    logger.info("Top descriptors: %s",
                ", ".join(f"{k} {v:.3f}" for k, v in best_abl))
    logger.info("Saved -> %s", fname)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Held-out evaluation of the OSR cascade.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--artifacts-dir", required=True)
    p.add_argument("--feat-root", required=True)
    p.add_argument("--layer", type=int, required=True, choices=[1, 2, 5])
    p.add_argument("--protocol-dir", default=PROTO_DIR_DEFAULT)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    add_repo_to_path()
    configure_logging(verbose=args.verbose)
    run(artifacts_dir=args.artifacts_dir, feat_root=args.feat_root,
        layer=args.layer, protocol_dir=args.protocol_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
