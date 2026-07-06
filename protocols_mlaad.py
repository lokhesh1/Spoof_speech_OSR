#!/usr/bin/env python3
"""Open-set source-tracing protocol for MLAAD v5.

Reads the ``mlaad4sourcetracing/`` split CSVs and resolves each clip to either
the raw MLAAD audio (``.wav``) or the features produced by
:mod:`extract_mlaad` (``.npz``). The two trees are identical below the split
prefix, so a single resolver serves both:

    protocol CSV path : ./fake/<lang>/<model_dir>/<stem>.wav
    raw audio         : <mlaad_root>/fake/<lang>/<model_dir>/<stem>.wav
    features          : <feat_root>/layer_<NN>/fake/<lang>/<model_dir>/<stem>.npz

Workflow: features are extracted **first** (``extract_mlaad.py``), then this
module builds train/dev/eval splits over whatever exists on disk.

Open-set labels
---------------
The 24 models in ``train.csv`` define the *known* classes, indexed ``0..23`` in
sorted order. Any model in dev/eval that is absent from train is *unknown* and
gets label :data:`UNKNOWN_LABEL` (``-1``). ``Clip.is_known`` is derived from
true ``train.csv`` membership -- the right basis for assigning a class label --
so it generally matches, but is not bound to, the ``model_seen`` axis of the
``fine/`` subsplits. (One model, ``tts_models/en/ljspeech/vits--neon`` / 91
clips, sits in ``eval___lang_seen___model_not_seen.csv`` yet is in train; those
clips are correctly treated as known here.)

Usage
-----
After running ``extract_mlaad.py --hf-id ... --layers 5 --out-dir feats``::

    from protocols_mlaad import load_split, MlaadFeatureDataset

    label_map = None  # built from train.csv automatically
    train = load_split("mlaad4sourcetracing", "train",
                       feat_root="feats", layer=5)
    eval_uu = load_split("mlaad4sourcetracing", "eval",
                         subsplit="lang_not_seen__model_not_seen",
                         feat_root="feats", layer=5)

    ds = MlaadFeatureDataset(train)          # -> (feature_tensor, label)

Or resolve to raw audio instead (``mlaad_root`` instead of ``feat_root``)::

    clips = load_split("mlaad4sourcetracing", "train",
                       mlaad_root=".../Mlaad_v5/mlaad_v5")

Quick stats from the command line::

    python protocols_mlaad.py mlaad4sourcetracing --feat-root feats --layer 5
"""
from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("ser.protocols_mlaad")

SPLITS = ("train", "dev", "eval")

#: Valid ``fine/`` subsplit selectors (the ``<split>___`` prefix is added per call).
SUBSPLITS = (
    "lang_seen__model_seen",
    "lang_seen__model_not_seen",
    "lang_not_seen__model_seen",
    "lang_not_seen__model_not_seen",
)

#: Label assigned to any clip whose generating model was not seen in training.
UNKNOWN_LABEL = -1


# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Clip:
    """One resolved clip from a protocol split."""

    path: Path        # resolved absolute/relative path on disk (.wav or .npz)
    rel: str          # stable key relative to the tree root: fake/<lang>/<model>/<stem>.wav
    model_name: str   # e.g. "tts_models/en/ljspeech/tacotron2-DCA"
    language: str     # e.g. "en"
    label: int        # known class index 0..K-1, or UNKNOWN_LABEL
    is_known: bool    # True iff the model was seen in train.csv


# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #
def _csv_path(protocol_dir: Path, split: str, subsplit: Optional[str]) -> Path:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if subsplit is None:
        return protocol_dir / f"{split}.csv"
    if subsplit not in SUBSPLITS:
        raise ValueError(f"subsplit must be one of {SUBSPLITS}, got {subsplit!r}")
    # fine/<split>___lang_seen___model_seen.csv  (double underscore -> triple)
    fname = f"{split}___{subsplit.replace('__', '___')}.csv"
    return protocol_dir / "fine" / fname


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Protocol CSV not found: {csv_path}")
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _rel_from_csv(csv_path_value: str) -> str:
    """``./fake/en/.../x.wav`` -> ``fake/en/.../x.wav`` (strip leading ``./``)."""
    return csv_path_value.lstrip("./") if csv_path_value.startswith("./") \
        else csv_path_value.lstrip("/")


def _language_from_rel(rel: str) -> str:
    """``fake/<lang>/<model_dir>/<stem>.wav`` -> ``<lang>``."""
    parts = Path(rel).parts
    return parts[1] if len(parts) >= 2 else ""


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def build_label_map(protocol_dir: Path | str) -> Dict[str, int]:
    """Map each train model to a class index ``0..K-1`` (sorted, deterministic).

    The 24 models in ``train.csv`` are the known/closed-set classes. Models that
    appear only in dev/eval are open-set unknowns and are *not* in this map.
    """
    protocol_dir = Path(protocol_dir)
    rows = _read_rows(protocol_dir / "train.csv")
    models = sorted({r["model_name"] for r in rows})
    return {m: i for i, m in enumerate(models)}


# --------------------------------------------------------------------------- #
# Split loading
# --------------------------------------------------------------------------- #
def load_split(
    protocol_dir: Path | str,
    split: str,
    *,
    subsplit: Optional[str] = None,
    mlaad_root: Optional[Path | str] = None,
    feat_root: Optional[Path | str] = None,
    layer: Optional[int] = None,
    label_map: Optional[Dict[str, int]] = None,
    require_exists: bool = True,
    dedup: bool = True,
) -> List[Clip]:
    """Load a protocol split and resolve every clip to disk.

    Exactly one of ``mlaad_root`` (resolve to ``.wav`` audio) or ``feat_root``
    (resolve to ``.npz`` features, ``layer`` required) must be given.

    Parameters
    ----------
    protocol_dir:
        Path to ``mlaad4sourcetracing/``.
    split:
        ``"train"`` | ``"dev"`` | ``"eval"``.
    subsplit:
        Optional ``fine/`` selector, one of :data:`SUBSPLITS`. ``None`` uses the
        full ``<split>.csv``.
    mlaad_root:
        Root containing ``fake/`` (e.g. ``.../Mlaad_v5/mlaad_v5``). Resolves WAV.
    feat_root:
        ``--out-dir`` passed to ``extract_mlaad.py``. Resolves NPZ under
        ``layer_<NN>/``; requires ``layer``.
    layer:
        Layer index to read when ``feat_root`` is used.
    label_map:
        Reuse a prebuilt known-model map; built from ``train.csv`` if ``None``.
    require_exists:
        Drop clips whose resolved file is missing (logs the count). Set ``False``
        to keep every protocol row regardless of what is on disk.
    dedup:
        Drop rows whose ``rel`` path was already seen (default). ``eval.csv``
        carries 109 verified-redundant duplicate rows (same file, model and
        transcript); deduping keeps one Clip per physical clip. Set ``False`` to
        return the CSV rows verbatim.
    """
    protocol_dir = Path(protocol_dir)

    if (mlaad_root is None) == (feat_root is None):
        raise ValueError("Provide exactly one of mlaad_root or feat_root.")
    if feat_root is not None and layer is None:
        raise ValueError("layer is required when resolving features (feat_root).")

    if label_map is None:
        label_map = build_label_map(protocol_dir)

    if mlaad_root is not None:
        base = Path(mlaad_root)
        def resolve(rel: str) -> Path:
            return base / rel
    else:
        base = Path(feat_root) / f"layer_{layer:02d}"
        def resolve(rel: str) -> Path:
            return base / Path(rel).with_suffix(".npz")

    rows = _read_rows(_csv_path(protocol_dir, split, subsplit))

    clips: List[Clip] = []
    missing = 0
    duplicate = 0
    seen: set[str] = set()
    for r in rows:
        rel = _rel_from_csv(r["path"])
        model_name = r["model_name"]
        if dedup:
            if rel in seen:
                duplicate += 1
                continue
            seen.add(rel)
        resolved = resolve(rel)
        if require_exists and not resolved.exists():
            missing += 1
            continue
        is_known = model_name in label_map
        clips.append(Clip(
            path=resolved,
            rel=rel,
            model_name=model_name,
            language=_language_from_rel(rel),
            label=label_map[model_name] if is_known else UNKNOWN_LABEL,
            is_known=is_known,
        ))

    tag = f"{split}" + (f"/{subsplit}" if subsplit else "")
    if duplicate:
        logger.info("%s: dropped %d duplicate rows (dedup=True).", tag, duplicate)
    if missing:
        logger.warning("%s: %d/%d clips missing on disk and dropped.",
                       tag, missing, len(rows))
    logger.info("%s: %d clips (%d known, %d unknown).", tag, len(clips),
                sum(c.is_known for c in clips), sum(not c.is_known for c in clips))
    return clips


# --------------------------------------------------------------------------- #
# Torch dataset over extracted features
# --------------------------------------------------------------------------- #
try:  # torch is optional for pure-protocol (stats-only) use
    from torch.utils.data import Dataset as _DatasetBase
except Exception:  # pragma: no cover
    _DatasetBase = object


class MlaadFeatureDataset(_DatasetBase):
    """Map-style dataset yielding ``(features, label)`` from ``.npz`` clips.

    Requires clips resolved with ``feat_root`` (so ``path`` points at a ``.npz``).
    ``features`` is a 1-D tensor for pooled extraction or 2-D ``(T, D)`` for
    ``--pooling none`` (use :func:`collate_pad` to batch variable lengths).
    Subclasses ``torch.utils.data.Dataset`` when torch is available.
    """

    def __init__(self, clips: List[Clip], return_meta: bool = False):
        self.clips = clips
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int):
        import numpy as np
        import torch

        c = self.clips[i]
        with np.load(c.path, allow_pickle=False) as data:
            feat = torch.from_numpy(data["features"].astype("float32"))
        if self.return_meta:
            return feat, c.label, {"rel": c.rel, "model_name": c.model_name,
                                   "language": c.language, "is_known": c.is_known}
        return feat, c.label


def collate_pad(batch):
    """Collate variable-length ``(T, D)`` features by right-padding to max T.

    Returns ``(feats (B, T_max, D), lengths (B,), labels (B,))``. Use only with
    ``--pooling none`` extractions; pooled (1-D) features stack with the default
    collate.
    """
    import torch

    feats = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    t_max = int(lengths.max())
    d = feats[0].shape[1]
    out = torch.zeros(len(feats), t_max, d, dtype=feats[0].dtype)
    for i, f in enumerate(feats):
        out[i, : f.shape[0]] = f
    return out, lengths, labels


# --------------------------------------------------------------------------- #
# CLI sanity check
# --------------------------------------------------------------------------- #
def _summarize(clips: List[Clip]) -> str:
    langs = sorted({c.language for c in clips})
    models = sorted({c.model_name for c in clips})
    known = sum(c.is_known for c in clips)
    return (f"{len(clips):6d} clips | {len(models):3d} models "
            f"({known} known-clips, {len(clips) - known} unknown-clips) | "
            f"{len(langs):2d} langs")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Print MLAAD source-tracing split statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("protocol_dir", help="Path to mlaad4sourcetracing/.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mlaad-root", help="Resolve raw .wav under this MLAAD root.")
    g.add_argument("--feat-root", help="Resolve .npz features under this root.")
    p.add_argument("--layer", type=int, help="Layer index (required with --feat-root).")
    p.add_argument("--no-require-exists", action="store_true",
                   help="Count protocol rows even if files are missing on disk.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    label_map = build_label_map(args.protocol_dir)
    print(f"Known classes (from train.csv): {len(label_map)}")

    common = dict(mlaad_root=args.mlaad_root, feat_root=args.feat_root,
                  layer=args.layer, label_map=label_map,
                  require_exists=not args.no_require_exists)
    for split in SPLITS:
        clips = load_split(args.protocol_dir, split, **common)
        print(f"\n[{split}]  {_summarize(clips)}")
        if split != "train":
            for sub in SUBSPLITS:
                sc = load_split(args.protocol_dir, split, subsplit=sub, **common)
                print(f"    {sub:32s} {_summarize(sc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
