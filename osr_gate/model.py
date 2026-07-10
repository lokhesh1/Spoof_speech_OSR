#!/usr/bin/env python3
"""24-way heads over cached XLS-R feature sequences.

Input to every head is a frame sequence ``(B, T, D)`` (D = 1024 for
wav2vec2-xls-r-300m) plus a boolean padding ``mask`` ``(B, T)`` (True = real
frame). A shared trunk -- attentive statistics pooling + BatchNorm -- turns the
sequence into a fixed ``(B, 2D)`` utterance vector; two heads sit on top:

    * :class:`BusemannHead` -- projects to a ``d``-dim Poincare ball point ``z``
      (trained with :class:`hyperbolic.PenalizedBusemannLoss`); this is the
      open-set model.
    * :class:`EuclideanHead` -- plain linear classifier on an L2-normalized
      embedding (trained with cross-entropy); the Euclidean control demanded by
      principle P3.

Both share :class:`PooledTrunk`, so the two runs differ only in geometry, not in
pooling capacity. Geometry (exp-map, projection) always runs in fp32; features
may arrive in fp16.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic import expmap0, project

logger = logging.getLogger("ser.osr_gate.model")

_STD_EPS = 1e-5


# --------------------------------------------------------------------------- #
# Pooling trunk
# --------------------------------------------------------------------------- #
class AttentiveStatsPooling(nn.Module):
    """Attentive statistics pooling (Okabe et al., 2018) with padding mask.

    Scalar attention per frame ``a_t = softmax_t(v^T tanh(W h_t + b))`` over the
    valid frames, then concatenated weighted mean and weighted std:
    ``(B, T, D) -> (B, 2D)``.
    """

    def __init__(self, dim: int, attn_hidden: int = 128):
        super().__init__()
        self.w = nn.Linear(dim, attn_hidden)
        self.v = nn.Linear(attn_hidden, 1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scores = self.v(torch.tanh(self.w(x))).squeeze(-1)      # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        a = torch.softmax(scores, dim=1).unsqueeze(-1)          # (B, T, 1)
        mean = (a * x).sum(dim=1)                               # (B, D)
        var = (a * x.pow(2)).sum(dim=1) - mean.pow(2)
        std = var.clamp_min(_STD_EPS).sqrt()
        return torch.cat([mean, std], dim=1)                    # (B, 2D)


class PooledTrunk(nn.Module):
    """ASP + BatchNorm shared by both heads. ``(B, T, D) -> (B, 2D)``."""

    def __init__(self, in_dim: int, attn_hidden: int = 128):
        super().__init__()
        self.pool = AttentiveStatsPooling(in_dim, attn_hidden)
        self.bn = nn.BatchNorm1d(2 * in_dim)
        self.out_dim = 2 * in_dim

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.bn(self.pool(x, mask))


# --------------------------------------------------------------------------- #
# Hyperbolic (open-set) head
# --------------------------------------------------------------------------- #
class BusemannHead(nn.Module):
    """Trunk -> linear -> exp-map into the Poincare ball.

    ``forward`` returns the ball point ``z`` of shape ``(B, ball_dim)`` with
    ``||z|| < 1``. Classification logits and the training loss come from
    :class:`hyperbolic.PenalizedBusemannLoss` applied to ``z`` (kept separate so
    the fixed prototypes live in one place).
    """

    def __init__(self, in_dim: int, ball_dim: int = 16, attn_hidden: int = 128):
        super().__init__()
        self.trunk = PooledTrunk(in_dim, attn_hidden)
        self.proj = nn.Linear(self.trunk.out_dim, ball_dim)
        self.ball_dim = ball_dim

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        v = self.proj(self.trunk(x, mask)).float()      # fp32 for geometry
        return project(expmap0(v))


# --------------------------------------------------------------------------- #
# Euclidean control head
# --------------------------------------------------------------------------- #
class EuclideanHead(nn.Module):
    """Trunk -> L2-normalized embedding -> linear classifier (cross-entropy).

    ``forward`` returns ``(embedding, logits)``: the embedding feeds the
    Euclidean prototypes/descriptors, the logits feed cross-entropy.
    """

    def __init__(
        self, in_dim: int, n_classes: int, embed_dim: int = 256,
        attn_hidden: int = 128,
    ):
        super().__init__()
        self.trunk = PooledTrunk(in_dim, attn_hidden)
        self.embed = nn.Linear(self.trunk.out_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        e = F.normalize(self.embed(self.trunk(x, mask)), dim=1)
        return e, self.classifier(e)
