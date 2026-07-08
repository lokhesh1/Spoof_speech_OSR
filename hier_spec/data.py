#!/usr/bin/env python3
"""Datasets over cached Wav2Vec2 features for Hier-Spec.

Wraps ``protocols_mlaad.load_split(feat_root=...)`` clips into fixed-window
feature tensors plus the hierarchical labels the model needs:

    (feat (max_frames, feat_dim), arch_label, model_local_label, flat_label)

``arch_label`` indexes the 13 architectures, ``model_local_label`` is the
within-architecture model index for multi-model archs (``-1`` for singletons),
and ``flat_label`` is the 24-class known-model index (``-1`` for open-set
unknowns). Training uses inverse-frequency sample weights per the spec.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from common import fit_frames
import taxonomy as tax

try:
    from torch.utils.data import Dataset as _DatasetBase
except Exception:  # pragma: no cover
    _DatasetBase = object


class HierSpecDataset(_DatasetBase):
    def __init__(self, clips, *, max_frames: int, train: bool,
                 arch_label_map: Dict[str, int],
                 model_label_maps: Dict[str, Dict[str, int]],
                 return_meta: bool = False, seed: int = 0):
        self.clips = clips
        self.max_frames = max_frames
        self.train = train
        self.arch_label_map = arch_label_map
        self.model_label_maps = model_label_maps
        self.return_meta = return_meta
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.clips)

    def _labels_for(self, model_name: str) -> Tuple[int, int]:
        """(arch_label, model_local_label) for a known-model name."""
        arch = tax.ARCH_OF_KNOWN[model_name]
        arch_label = self.arch_label_map[arch]
        if arch in self.model_label_maps:
            model_local = self.model_label_maps[arch][model_name]
        else:
            model_local = -1                       # singleton -> no Stage-2 head
        return arch_label, model_local

    def __getitem__(self, i: int):
        import torch

        c = self.clips[i]
        with np.load(c.path, allow_pickle=False) as data:
            feat = data["features"].astype("float32")
        feat = fit_frames(feat, self.max_frames, train=self.train, rng=self._rng)
        feat = torch.from_numpy(np.ascontiguousarray(feat))

        if c.is_known:
            arch_label, model_local = self._labels_for(c.model_name)
        else:
            arch_label, model_local = -1, -1
        flat_label = c.label                       # 0..23 or -1

        if self.return_meta:
            meta = {"rel": c.rel, "model_name": c.model_name,
                    "language": c.language, "is_known": c.is_known}
            return feat, arch_label, model_local, flat_label, meta
        return feat, arch_label, model_local, flat_label


def collate(batch):
    """Stack fixed-window features and label vectors (default collate is fine,
    but this keeps meta out of the tensors when ``return_meta`` is used)."""
    import torch

    has_meta = len(batch[0]) == 5
    feats = torch.stack([b[0] for b in batch], dim=0)
    arch = torch.tensor([b[1] for b in batch], dtype=torch.long)
    model_local = torch.tensor([b[2] for b in batch], dtype=torch.long)
    flat = torch.tensor([b[3] for b in batch], dtype=torch.long)
    if has_meta:
        return feats, arch, model_local, flat, [b[4] for b in batch]
    return feats, arch, model_local, flat


def inverse_frequency_weights(clips) -> np.ndarray:
    """Per-clip weight ``w_c = 0.5 * (1/n_c + 1/N_s(c))`` (spec section 5).

    ``n_c`` is the count of the clip's model; ``N_s(c)`` the count of its
    architecture. Defined for known (train) clips only.
    """
    model_counts = Counter(c.model_name for c in clips)
    arch_counts: Counter = Counter()
    for c in clips:
        arch_counts[tax.ARCH_OF_KNOWN[c.model_name]] += 1
    w = np.empty(len(clips), dtype=np.float64)
    for i, c in enumerate(clips):
        n_c = model_counts[c.model_name]
        n_s = arch_counts[tax.ARCH_OF_KNOWN[c.model_name]]
        w[i] = 0.5 * (1.0 / n_c + 1.0 / n_s)
    return w
