#!/usr/bin/env python3
"""Per-clip feature extraction over the MLAAD v5 dataset.

Loads any Hugging Face model via ``--hf-id``, extracts one or more transformer
layers in a single forward pass, applies the chosen pooling, and saves one
``.npz`` per clip mirroring the MLAAD ``fake/`` tree.

Output layout::

    <out_dir>/
        layer_03/fake/<lang>/<model_dir>/<stem>.npz
        layer_06/fake/<lang>/<model_dir>/<stem>.npz
        ...

Each ``.npz`` contains:

    features   : (D,) for mean/max/mean_std pooling, or (T, D) for "none"
                 (float32, or float16 when --fp16 is set)
    path       : relative clip path (str)
    model_name : TTS model directory name (str)
    language   : language code (str)
    layer      : layer index (int)
    pooling    : pooling method used (str)
    hf_id      : Hugging Face model ID (str)

Examples
--------
Smoke-test on 20 clips (whisper-base, layer 3, mean pooling)::

    python extract_mlaad.py \\
        --hf-id openai/whisper-base \\
        --mlaad-root /data/mlaad5 \\
        --layers 3 --pooling mean --limit 20 --device cpu

Extract layers 1, 3, and 6 with full sequence output in one pass::

    python extract_mlaad.py \\
        --hf-id superb/hubert-large-superb-er \\
        --mlaad-root /data/mlaad5 \\
        --layers 1 3 6 --pooling none
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from model_loader import TARGET_SAMPLE_RATE, configure_logging, load_model

logger = logging.getLogger("ser.extract_mlaad")

MLAAD_ROOT_DEFAULT = Path("./data/mlaad5")
POOLING_CHOICES = ["mean", "max", "mean_std", "none"]


# --------------------------------------------------------------------------- #
# Audio loading
# --------------------------------------------------------------------------- #
def load_audio(path: Path) -> "torch.Tensor":
    """Load any audio file as a 1-D 16 kHz mono float32 tensor."""
    import torch

    try:
        import torchaudio
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:          # stereo -> mono
            wav = wav.mean(0, keepdim=True)
        if sr != TARGET_SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SAMPLE_RATE)
        return wav.squeeze(0)
    except Exception:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        wav = torch.from_numpy(data).unsqueeze(0)
        if sr != TARGET_SAMPLE_RATE:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, TARGET_SAMPLE_RATE)
        return wav.squeeze(0)


# --------------------------------------------------------------------------- #
# MLAAD directory scan
# --------------------------------------------------------------------------- #
def scan_mlaad(root: Path) -> List[Tuple[str, str, str]]:
    """Walk ``root/fake/<lang>/<model>/`` and collect all ``.wav`` clips.

    Returns a list of ``(rel_path, model_name, language)`` triples, where
    ``rel_path`` is relative to ``root`` (e.g. ``fake/en/tts.../foo.wav``).
    """
    fake_dir = root / "fake"
    if not fake_dir.is_dir():
        raise FileNotFoundError(
            f"Expected a 'fake/' subdirectory under MLAAD root: {root}"
        )
    recs: List[Tuple[str, str, str]] = []
    for lang_dir in sorted(fake_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        language = lang_dir.name
        for model_dir in sorted(lang_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            for wav_file in sorted(model_dir.glob("*.wav")):
                recs.append((
                    str(wav_file.relative_to(root)),
                    model_name,
                    language,
                ))
    return recs


# --------------------------------------------------------------------------- #
# Protocol filter
# --------------------------------------------------------------------------- #
def protocol_rels(protocol_dir: Path) -> Set[str]:
    """Union of clip rel-paths listed in ``{train,dev,eval}.csv``.

    Reads the ``path`` column, normalizes ``./fake/...`` -> ``fake/...`` (to
    match :func:`scan_mlaad`), and dedups across splits (so eval.csv's repeated
    rows collapse to one). Only these clips are extracted in protocol mode.
    """
    rels: Set[str] = set()
    for split in ("train", "dev", "eval"):
        csv_path = protocol_dir / f"{split}.csv"
        if not csv_path.is_file():
            logger.warning("Protocol CSV missing, skipping: %s", csv_path)
            continue
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                v = row["path"]
                rels.add(v.lstrip("./") if v.startswith("./") else v)
    return rels


# --------------------------------------------------------------------------- #
# Pooling
# --------------------------------------------------------------------------- #
def pool(features: np.ndarray, method: str) -> np.ndarray:
    """Reduce ``(T, D)`` to a fixed-size vector (or keep ``(T, D)`` for "none").

    - ``"mean"``     -> ``(D,)``
    - ``"max"``      -> ``(D,)``
    - ``"mean_std"`` -> ``(2D,)``   concat of mean and std
    - ``"none"``     -> ``(T, D)``  full sequence, no reduction
    """
    if method == "none":
        return features
    if method == "mean":
        return features.mean(axis=0)
    if method == "max":
        return features.max(axis=0)
    if method == "mean_std":
        return np.concatenate([features.mean(axis=0), features.std(axis=0)])
    raise ValueError(f"Unknown pooling '{method}'. Choose from: {POOLING_CHOICES}")


# --------------------------------------------------------------------------- #
# Main extraction loop
# --------------------------------------------------------------------------- #
def extract(
    *,
    hf_id: str,
    root: Path,
    out_dir: Path,
    layers: List[int],
    pooling: str,
    device: str,
    limit: Optional[int],
    overwrite: bool,
    log_every: int,
    fp16: bool = False,
    protocol_dir: Optional[Path] = None,
) -> None:
    recs = scan_mlaad(root)
    if protocol_dir is not None:
        allowed = protocol_rels(protocol_dir)
        before = len(recs)
        recs = [r for r in recs if r[0] in allowed]
        found = {r[0] for r in recs}
        logger.info(
            "Protocol filter: %d/%d protocol clips found on disk "
            "(%d listed but missing); skipping %d non-protocol clips.",
            len(recs), len(allowed), len(allowed - found), before - len(recs),
        )
    if limit:
        recs = recs[:limit]
    n = len(recs)
    logger.info(
        "MLAAD pool: %d clips | hf_id=%s | layers=%s | pooling=%s | dtype=%s",
        n, hf_id, layers, pooling, "float16" if fp16 else "float32",
    )

    model = load_model(hf_id, device=device)

    done = skipped = 0
    failed: List[str] = []
    t0 = time.time()

    for idx, (rel, model_name, language) in enumerate(recs):
        out_paths = {
            layer: out_dir / f"layer_{layer:02d}" / Path(rel).with_suffix(".npz")
            for layer in layers
        }

        if not overwrite and all(p.exists() for p in out_paths.values()):
            skipped += 1
        else:
            abs_path = root / rel
            try:
                waveform = load_audio(abs_path)
                layer_feats = model.forward_layers(waveform)
            except Exception as exc:
                logger.warning("Extraction failed for %s: %s", rel, exc)
                failed.append(rel)
                continue

            available = sorted(layer_feats)
            any_saved = False
            for layer in layers:
                if layer not in layer_feats:
                    logger.warning(
                        "%s: layer %d not available (model has layers %d..%d); skipping layer.",
                        rel, layer, available[0], available[-1],
                    )
                    continue
                if not overwrite and out_paths[layer].exists():
                    continue
                feat = pool(
                    layer_feats[layer].numpy().astype(np.float32, copy=False),
                    pooling,
                )
                if fp16:
                    # pool in fp32 (mean/std stay accurate), store in fp16
                    feat = feat.astype(np.float16)
                out_paths[layer].parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    out_paths[layer],
                    features=feat,
                    path=np.asarray(rel),
                    model_name=np.asarray(model_name),
                    language=np.asarray(language),
                    layer=np.asarray(layer),
                    pooling=np.asarray(pooling),
                    hf_id=np.asarray(hf_id),
                )
                any_saved = True
            if any_saved:
                done += 1

        if (idx + 1) % log_every == 0 or (idx + 1) == n:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
            eta = (n - (idx + 1)) / rate if rate > 0 else float("nan")
            logger.info(
                "%d/%d  %.1f clip/s  ETA %.1f min  done=%d  skip=%d  fail=%d",
                idx + 1, n, rate, eta / 60, done, skipped, len(failed),
            )

    logger.info("Done: %d extracted, %d skipped, %d failed.", done, skipped, len(failed))
    if failed:
        fail_log = out_dir / "failed.txt"
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        fail_log.write_text("\n".join(failed) + "\n", encoding="utf-8")
        logger.info("Failed clip paths -> %s", fail_log)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract per-clip MLAAD v5 features from a Hugging Face model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--hf-id", required=True,
        help="Hugging Face model repo ID (e.g. 'openai/whisper-base').",
    )
    p.add_argument(
        "--mlaad-root", default=str(MLAAD_ROOT_DEFAULT),
        help="MLAAD v5 root containing fake/<lang>/<model>/.",
    )
    p.add_argument(
        "--out-dir", default="feats_mlaad",
        help="Output root; layer_<N>/ subfolders are created inside.",
    )
    p.add_argument(
        "--layers", type=int, nargs="+", required=True,
        help=(
            "One or more transformer layer indices to extract in a single pass. "
            "0 = input embedding, 1..N = transformer block outputs. "
            "Example: --layers 1 3 6 12"
        ),
    )
    p.add_argument(
        "--pooling", choices=POOLING_CHOICES, default="mean",
        help=(
            "Time-axis pooling applied to each (T, D) layer output. "
            "'none' keeps the full sequence."
        ),
    )
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N clips (smoke test).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract even if the .npz already exists.")
    p.add_argument("--fp16", action="store_true",
                   help="Store features as float16 (halves disk; forward stays fp32).")
    p.add_argument("--protocol-dir", default=None,
                   help="Restrict extraction to clips in <dir>/{train,dev,eval}.csv "
                        "(e.g. mlaad4sourcetracing); omit to extract the whole tree.")
    p.add_argument("--log-every", type=int, default=500,
                   help="Progress log interval (clips).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    extract(
        hf_id=args.hf_id,
        root=Path(args.mlaad_root),
        out_dir=Path(args.out_dir),
        layers=sorted(set(args.layers)),
        pooling=args.pooling,
        device=args.device,
        limit=args.limit,
        overwrite=args.overwrite,
        log_every=args.log_every,
        fp16=args.fp16,
        protocol_dir=Path(args.protocol_dir) if args.protocol_dir else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
