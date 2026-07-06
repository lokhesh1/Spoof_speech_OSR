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
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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
) -> None:
    recs = scan_mlaad(root)
    if limit:
        recs = recs[:limit]
    n = len(recs)
    logger.info(
        "MLAAD pool: %d clips | hf_id=%s | layers=%s | pooling=%s",
        n, hf_id, layers, pooling,
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
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
