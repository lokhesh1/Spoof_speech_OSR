#!/usr/bin/env python3
"""ICMM class-pair weight matrices: manual (architecture families) and auto.

Manual weights mirror the ASVspoof A04/A06 3.0-weight strategy: within-family
TTS pairs get weight 3.0 (the overlap-prone boundary), cross-family pairs
default to 1.0.

Auto weights are computed from learned centroid distances after the warm phase:
pairs with smaller angular gaps receive higher ICMM pressure.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _get_family(model_name: str) -> str:
    m = model_name.lower()
    if "tacotron2" in m:
        return "tacotron2"
    if "xtts" in m:
        return "xtts"
    if "vits" in m and "xtts" not in m and "vixtts" not in m:
        return "vits"
    if "bark" in m:
        return "bark"
    if "glow" in m:
        return "glow_tts"
    if "fast_pitch" in m:
        return "fast_pitch"
    if "speedy" in m:
        return "speedy_speech"
    if "melo" in m:
        return "melo"
    if "mars" in m:
        return "mars5"
    if "metavoice" in m:
        return "metavoice"
    if "mms-tts" in m or "mms_tts" in m:
        return "mms_tts"
    if "vixtts" in m:
        return "vixtts"
    if "griffin" in m:
        return "griffin_lim"
    return model_name


def manual_pair_weights(label_map: Dict[str, int]) -> torch.Tensor:
    """Build (K, K) pair-weight matrix from architecture family membership.

    Same-family pairs get weight 3.0 (the highest-risk overlap boundary).
    Cross-family pairs default to 1.0. Diagonal is 0.
    """
    K = len(label_map)
    idx_to_model = {v: k for k, v in label_map.items()}
    families = {i: _get_family(idx_to_model[i]) for i in range(K)}

    W = torch.ones(K, K)
    for i in range(K):
        for j in range(i + 1, K):
            if families[i] == families[j]:
                W[i, j] = 3.0
                W[j, i] = 3.0
        W[i, i] = 0.0
    return W


def auto_pair_weights(centroids: torch.Tensor) -> torch.Tensor:
    """Build (K, K) pair-weight matrix from centroid angular proximity.

    Pairs with smaller angular separation receive higher weight so that
    ICMM focuses its margin-carving budget on the tightest boundaries.
    Weights are normalised so the off-diagonal mean equals 1.0.
    """
    K = centroids.shape[0]
    c = F.normalize(centroids, dim=1)
    sim = c @ c.T
    dist = 1.0 - sim
    W = 1.0 / (dist + 0.01)
    W.fill_diagonal_(0)

    mask = ~torch.eye(K, dtype=torch.bool, device=centroids.device)
    mean_w = W[mask].mean()
    if mean_w > 0:
        W = W / mean_w
    W.fill_diagonal_(0)
    return W
