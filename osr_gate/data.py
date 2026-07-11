#!/usr/bin/env python3
"""Data loading for the open-set gate, layered over :mod:`protocols_mlaad`.

``protocols_mlaad`` already resolves splits to ``.npz`` clips
(:func:`load_split`), yields ``(features, label)`` (:class:`MlaadFeatureDataset`)
and right-pads variable-length ``(T, D)`` batches (:func:`collate_pad`). This
module adds only the training-specific pieces on top:

    * :func:`random_crop` / :class:`CroppedDataset` -- fixed-length random crops
      for training augmentation (full sequence at eval);
    * :func:`collate_mask` -- ``collate_pad`` plus a boolean frame mask
      (True = real frame) which :class:`model.AttentiveStatsPooling` consumes;
    * :func:`class_balanced_sampler` -- inverse-frequency sampling over the 24
      known classes (principle P1);
    * :func:`train_loader` / :func:`eval_loader` and the known/unknown filters
      that wire everything into ``DataLoader`` objects;
    * :func:`sliding_windows` -- window slicer reused by chunk-and-average eval.

Frames are model-agnostic: crop length is given in frames. XLS-R and
wav2vec2-base both hop 20 ms, so ``seconds * 50`` frames (:func:`sec_to_frames`);
4 s = 200 frames.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Make ``protocols_mlaad`` (repo root) importable however data.py is loaded.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from protocols_mlaad import (
    Clip,
    MlaadFeatureDataset,
    collate_pad,
    load_split,
)

logger = logging.getLogger("ser.osr_gate.data")

#: Feature frame rate for 20 ms-hop encoders (wav2vec2 / XLS-R).
FRAMES_PER_SEC = 50

#: Default training crop length (4 s).
DEFAULT_CROP_FRAMES = 4 * FRAMES_PER_SEC


def sec_to_frames(seconds: float, frames_per_sec: int = FRAMES_PER_SEC) -> int:
    """Convert a duration in seconds to a number of feature frames."""
    return int(round(seconds * frames_per_sec))


# --------------------------------------------------------------------------- #
# Known / unknown filters
# --------------------------------------------------------------------------- #
def filter_known(clips: List[Clip]) -> List[Clip]:
    """Keep only clips whose model was seen in train (labels ``0..K-1``)."""
    return [c for c in clips if c.is_known]


def filter_unknown(clips: List[Clip]) -> List[Clip]:
    """Keep only unknown clips (label ``UNKNOWN_LABEL``)."""
    return [c for c in clips if not c.is_known]


# --------------------------------------------------------------------------- #
# Cropping
# --------------------------------------------------------------------------- #
def random_crop(feat: torch.Tensor, win: int,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Return a random ``win``-frame slice of ``feat`` ``(T, D)``.

    Clips shorter than ``win`` are returned unchanged (they get padded and
    masked in the collate). Longer clips are cropped at a uniformly random
    start.
    """
    t = feat.shape[0]
    if t <= win:
        return feat
    if generator is not None:
        start = int(torch.randint(0, t - win + 1, (1,), generator=generator))
    else:
        start = int(torch.randint(0, t - win + 1, (1,)))
    return feat[start:start + win]


class CroppedDataset(Dataset):
    """Wrap :class:`MlaadFeatureDataset`, applying a random crop per item.

    ``crop_frames=None`` disables cropping (full-sequence, for eval).
    """

    def __init__(self, clips: List[Clip], crop_frames: Optional[int]):
        self.inner = MlaadFeatureDataset(clips, return_meta=False)
        self.crop_frames = crop_frames

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i: int):
        feat, label = self.inner[i]
        if self.crop_frames is not None:
            feat = random_crop(feat, self.crop_frames)
        return feat, label


# --------------------------------------------------------------------------- #
# Collate with boolean mask
# --------------------------------------------------------------------------- #
def lengths_to_mask(lengths: torch.Tensor, t_max: int) -> torch.Tensor:
    """``(B,)`` real-frame counts -> ``(B, t_max)`` bool mask (True = real)."""
    return torch.arange(t_max, device=lengths.device)[None, :] < lengths[:, None]


def collate_mask(batch):
    """Like :func:`collate_pad` but returns a boolean mask instead of lengths.

    Returns ``(feats (B, T_max, D), mask (B, T_max) bool, labels (B,))``.
    """
    feats, lengths, labels = collate_pad(batch)
    mask = lengths_to_mask(lengths, feats.shape[1])
    return feats, mask, labels


# --------------------------------------------------------------------------- #
# Class-balanced sampling
# --------------------------------------------------------------------------- #
def class_balanced_sampler(clips: List[Clip]) -> WeightedRandomSampler:
    """Inverse-frequency sampler over known-class labels (with replacement)."""
    labels = [c.label for c in clips]
    counts: dict[int, int] = {}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1
    weights = torch.tensor([1.0 / counts[y] for y in labels], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


# --------------------------------------------------------------------------- #
# Loader builders
# --------------------------------------------------------------------------- #
def train_loader(
    clips: List[Clip],
    batch_size: int = 32,
    crop_frames: int = DEFAULT_CROP_FRAMES,
    num_workers: int = 4,
    balanced: bool = True,
    pin_memory: bool = False,
) -> DataLoader:
    """Known-only training loader: random crops, class-balanced sampling, mask.

    ``clips`` should already be known-only (see :func:`filter_known`); a warning
    is emitted if any unknown clips slip through, since they carry label -1.
    ``pin_memory`` should be enabled only when moving batches to CUDA (it forces
    a CUDA allocation); ``train.py`` turns it on for GPU runs.
    """
    if any(not c.is_known for c in clips):
        logger.warning("train_loader received %d unknown clips; filter first.",
                        sum(not c.is_known for c in clips))
    ds = CroppedDataset(clips, crop_frames=crop_frames)
    sampler = class_balanced_sampler(clips) if balanced else None
    return DataLoader(
        ds, batch_size=batch_size, sampler=sampler, shuffle=sampler is None,
        num_workers=num_workers, collate_fn=collate_mask, drop_last=True,
        pin_memory=pin_memory,
    )


def eval_loader(
    clips: List[Clip],
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> DataLoader:
    """Full-sequence loader (no crop, no sampler); order preserved."""
    ds = CroppedDataset(clips, crop_frames=None)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_mask, pin_memory=pin_memory,
    )


# --------------------------------------------------------------------------- #
# Sliding windows (for chunk-and-average eval, used by eval.py)
# --------------------------------------------------------------------------- #
def sliding_windows(feat: torch.Tensor, win: int,
                    hop: Optional[int] = None) -> List[torch.Tensor]:
    """Slice ``feat`` ``(T, D)`` into ``win``-frame windows (hop defaults win//2).

    Clips shorter than ``win`` yield a single (uncropped) window. The last
    window is anchored to the end so the tail is always covered.
    """
    t = feat.shape[0]
    if t <= win:
        return [feat]
    hop = hop or win // 2
    starts = list(range(0, t - win + 1, hop))
    if starts[-1] != t - win:
        starts.append(t - win)
    return [feat[s:s + win] for s in starts]


# --------------------------------------------------------------------------- #
# Convenience: load the OSR splits in one call
# --------------------------------------------------------------------------- #
def load_osr_splits(protocol_dir: str, feat_root: str, layer: int):
    """Return ``(train_known, dev_known, dev_unknown, eval_all)`` clip lists.

    ``train_known`` trains the head; ``dev_known`` (+1) and ``dev_unknown`` (-1)
    calibrate the gate; ``eval_all`` (known + held-out unknown) is the test set.
    A single ``label_map`` from train.csv is shared across all splits.
    """
    from protocols_mlaad import build_label_map

    lm = build_label_map(protocol_dir)
    kw = dict(feat_root=feat_root, layer=layer, label_map=lm)
    train = filter_known(load_split(protocol_dir, "train", **kw))
    dev = load_split(protocol_dir, "dev", **kw)
    eval_all = load_split(protocol_dir, "eval", **kw)
    return train, filter_known(dev), filter_unknown(dev), eval_all
