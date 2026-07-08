#!/usr/bin/env python3
"""Cache Wav2Vec2-Base layer-5 frames for the MLAAD source-tracing protocol.

Only the ~57k clips referenced by ``train/dev/eval.csv`` are extracted (not the
full 154k MLAAD tree), and the output layout matches ``extract_mlaad.py`` so
``protocols_mlaad.load_split(feat_root=..., layer=5)`` resolves them directly::

    <out_dir>/layer_05/fake/<lang>/<model_dir>/<stem>.npz   # features: (T, 768)

Run once before training/eval. Reuses ``extract_mlaad.load_audio`` (the
soundfile-based loader that sidesteps the TorchCodec issue) and
``model_loader.load_model`` (frozen Wav2Vec2 front-end).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from common import HierSpecConfig, add_repo_to_path, configure_logging

logger = logging.getLogger("hier_spec.extract_feats")


def protocol_rels(protocol_dir: Path) -> List[str]:
    """Union of ``rel`` paths across train/dev/eval (deduplicated, sorted)."""
    add_repo_to_path()
    from protocols_mlaad import SPLITS, _read_rows, _rel_from_csv, _csv_path

    rels: set[str] = set()
    for split in SPLITS:
        for row in _read_rows(_csv_path(protocol_dir, split, None)):
            rels.add(_rel_from_csv(row["path"]))
    return sorted(rels)


def extract(*, protocol_dir: Path, mlaad_root: Path, out_dir: Path,
            layer: int, hf_id: str, device: str, limit: Optional[int],
            overwrite: bool, log_every: int) -> None:
    add_repo_to_path()
    from extract_mlaad import load_audio
    from model_loader import load_model

    rels = protocol_rels(protocol_dir)
    if limit:
        rels = rels[:limit]
    n = len(rels)
    logger.info("Protocol clips: %d | hf_id=%s | layer=%d", n, hf_id, layer)

    model = load_model(hf_id, device=device)
    layer_dir = out_dir / f"layer_{layer:02d}"

    done = skipped = 0
    failed: List[str] = []
    t0 = time.time()
    for idx, rel in enumerate(rels):
        out_path = layer_dir / Path(rel).with_suffix(".npz")
        if not overwrite and out_path.exists():
            skipped += 1
        else:
            try:
                waveform = load_audio(mlaad_root / rel)
                layer_feats = model.forward_layers(waveform)
                if layer not in layer_feats:
                    raise KeyError(f"layer {layer} not produced by {hf_id}")
                feat = layer_feats[layer].numpy().astype(np.float32, copy=False)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(out_path, features=feat, path=np.asarray(rel),
                         layer=np.asarray(layer), hf_id=np.asarray(hf_id))
                done += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Extraction failed for %s: %s", rel, exc)
                failed.append(rel)
                continue

        if (idx + 1) % log_every == 0 or (idx + 1) == n:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
            eta = (n - (idx + 1)) / rate if rate > 0 else float("nan")
            logger.info("%d/%d  %.1f clip/s  ETA %.1f min  done=%d skip=%d fail=%d",
                        idx + 1, n, rate, eta / 60, done, skipped, len(failed))

    logger.info("Done: %d extracted, %d skipped, %d failed.", done, skipped, len(failed))
    if failed:
        fail_log = out_dir / "failed.txt"
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        fail_log.write_text("\n".join(failed) + "\n", encoding="utf-8")
        logger.info("Failed clip paths -> %s", fail_log)


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = HierSpecConfig()
    p = argparse.ArgumentParser(
        description="Cache Wav2Vec2 layer features for the MLAAD protocol clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mlaad-root", required=True, help="MLAAD v5 root (has fake/).")
    p.add_argument("--protocol-dir", default="../mlaad4sourcetracing")
    p.add_argument("--out-dir", default="../feats_mlaad")
    p.add_argument("--hf-id", default=cfg.hf_id)
    p.add_argument("--layer", type=int, default=cfg.layer)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    configure_logging(verbose=args.verbose)
    extract(protocol_dir=Path(args.protocol_dir), mlaad_root=Path(args.mlaad_root),
            out_dir=Path(args.out_dir), layer=args.layer, hf_id=args.hf_id,
            device=args.device, limit=args.limit, overwrite=args.overwrite,
            log_every=args.log_every)
    return 0


if __name__ == "__main__":
    sys.exit(main())
