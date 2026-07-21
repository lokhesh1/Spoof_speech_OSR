#!/usr/bin/env python3
"""Evaluate the LFCC + GMM + one-class-SVM baseline on dev / eval.

Per clip: LFCC -> K-dim GMM log-likelihood vector ``C`` ->

    gate   : OneClassSVM on scaled C  (inlier = known, outlier = unknown)
    class  : argmax(C)  when gated known, else UNKNOWN (-1)

Prints open-set metrics for the whole split, and with ``--fine`` also for each
of the 4 language/model subsplits.

Example::

    python eval.py \\
        --mlaad-root /path/to/Mlaad_v5/mlaad_v5 \\
        --artifacts artifacts/model.joblib \\
        --split eval --cache-dir cache/lfcc --fine
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from common import (
    LFCCConfig, build_lfcc_transform, lfcc_cached, score_vector,
    load_artifacts, compute_metrics, format_metrics, add_repo_to_path,
    UNKNOWN_LABEL,
)

logger = logging.getLogger("gmm_baseline.eval")

DEFAULT_PROTOCOL_DIR = Path(__file__).resolve().parent.parent / "mlaad4sourcetracing"


def evaluate_split(
    *,
    protocol_dir: Path,
    mlaad_root: Path,
    split: str,
    subsplit: Optional[str],
    art: dict,
    cfg: LFCCConfig,
    cache_dir: Optional[Path],
    transform,
    limit: Optional[int],
    shuffle: bool = False,
    seed: int = 0,
    log_every: int = 1000,
) -> Optional[dict]:
    add_repo_to_path()
    from protocols_mlaad import load_split

    gmms = art["gmms"]
    scaler = art["scaler"]
    ocsvm = art["ocsvm"]
    label_map = art["label_map"]
    classes = art["classes"]

    clips = load_split(protocol_dir, split, subsplit=subsplit,
                       mlaad_root=mlaad_root, label_map=label_map)
    if shuffle:
        # the protocol is ordered (knowns first), so a head slice is single-group
        idx = np.random.default_rng(seed).permutation(len(clips))
        clips = [clips[i] for i in idx]
    if limit:
        clips = clips[:limit]
    if not clips:
        logger.warning("No clips for %s/%s", split, subsplit)
        return None

    yt_label, yt_known, yp_label, yp_known, yp_argmax = [], [], [], [], []
    scores: List[float] = []
    rows: List[dict] = []
    t0 = time.time()
    for i, c in enumerate(clips):
        feat = lfcc_cached(c.rel, c.path, transform, cfg, cache_dir)
        if feat.shape[0] == 0:
            logger.warning("Empty LFCC (skipped): %s", c.rel)
            continue
        C = score_vector(feat, gmms)
        Cs = scaler.transform(C[None, :])
        is_inlier = int(ocsvm.predict(Cs)[0]) == 1
        # signed distance to the OCSVM boundary: higher = more inlier = more known
        df = float(ocsvm.decision_function(Cs)[0])
        argmax = int(np.argmax(C))
        pred_label = argmax if is_inlier else UNKNOWN_LABEL

        yt_label.append(c.label)
        yt_known.append(c.is_known)
        yp_label.append(pred_label)
        yp_known.append(is_inlier)
        yp_argmax.append(argmax)
        scores.append(df)
        rows.append({
            "rel": c.rel,
            "language": c.language,
            "true_model": c.model_name,
            "true_label": c.label,
            "true_is_known": int(c.is_known),
            "pred_label": pred_label,
            "pred_model": classes[argmax] if is_inlier else "unknown",
            "pred_is_known": int(is_inlier),
            "top_gmm": classes[argmax],
            "top_loglik": float(C[argmax]),
            "ocsvm_score": df,
        })

        if (i + 1) % log_every == 0 or (i + 1) == len(clips):
            logger.info("scored %d/%d (%.1f clip/s)", i + 1, len(clips),
                        (i + 1) / (time.time() - t0))

    return {"metrics": compute_metrics(yt_label, yt_known, yp_label, yp_known,
                                       yp_argmax, y_score_known=scores,
                                       n_classes=len(classes)),
            "rows": rows}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate the LFCC+GMM+OCSVM source-tracing baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mlaad-root", required=True,
                   help="MLAAD root containing fake/<lang>/<model>/ (raw .wav).")
    p.add_argument("--artifacts", default="artifacts/model.joblib",
                   help="Trained joblib bundle from train.py.")
    p.add_argument("--protocol-dir", default=str(DEFAULT_PROTOCOL_DIR))
    p.add_argument("--split", choices=("dev", "eval"), default="eval")
    p.add_argument("--subsplit", default=None,
                   help="Evaluate a single fine/ subsplit instead of the whole split.")
    p.add_argument("--fine", action="store_true",
                   help="Also report each of the 4 language/model subsplits.")
    p.add_argument("--cache-dir", default=None,
                   help="Reuse/write the same LFCC .npy cache as train.py.")
    p.add_argument("--out-json", default=None,
                   help="Write metrics (overall + subsplits) to this JSON file.")
    p.add_argument("--out-csv", default=None,
                   help="Write per-clip predictions to this CSV file.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle before --limit; the protocol is ordered, so a "
                        "plain head slice yields known-only clips.")
    p.add_argument("--seed", type=int, default=0, help="Seed for --shuffle.")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    add_repo_to_path()
    from protocols_mlaad import SUBSPLITS

    art = load_artifacts(args.artifacts)
    cfg = LFCCConfig(**{k: art["config"][k] for k in
                        ("sample_rate", "frame_seconds", "overlap",
                         "n_fft", "n_lfcc", "n_filter")})
    transform = build_lfcc_transform(cfg)

    common_kw = dict(
        protocol_dir=Path(args.protocol_dir), mlaad_root=Path(args.mlaad_root),
        split=args.split, art=art, cfg=cfg,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        transform=transform, limit=args.limit, shuffle=args.shuffle,
        seed=args.seed, log_every=args.log_every,
    )

    targets: List[Optional[str]]
    if args.subsplit:
        targets = [args.subsplit]
    else:
        targets = [None] + (list(SUBSPLITS) if args.fine else [])

    print(f"\n=== {args.split} ===")
    metrics_by_target: dict = {}
    csv_rows: List[dict] = []
    for sub in targets:
        res = evaluate_split(subsplit=sub, **common_kw)
        if res is None:
            continue
        name = sub or "overall"
        metrics_by_target[name] = res["metrics"]
        print(format_metrics(name, res["metrics"]))
        # per-clip CSV: take the broadest target (overall, else the single subsplit)
        if not csv_rows:
            csv_rows = res["rows"]

    if args.out_json:
        import json
        out = {
            "artifacts": str(args.artifacts),
            "split": args.split,
            "config": art["config"],
            "ocsvm_params": art.get("ocsvm_params"),
            "metrics": metrics_by_target,
        }
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nMetrics JSON -> {path}")

    if args.out_csv and csv_rows:
        import csv as _csv
        path = Path(args.out_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"Per-clip CSV ({len(csv_rows)} rows) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
