#!/usr/bin/env python3
"""NovelModel: frame projection -> TAPool -> encoder -> fine head.

Architecture from the C-ICMM specification, adapted for MLAADv5 source
tracing (coarse head dropped; fine head only).

Input: ``(B, T, in_dim)`` XLS-R-300M frame features + ``(B, T)`` padding mask.
Output: ``(embedding, logits)`` where ``embedding`` is L2-normalized on the
unit hypersphere and ``logits`` feed cross-entropy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TAPool(nn.Module):
    """Temporal Attention Pooling via multi-head cross-attention.

    A learnable query attends over the frame sequence, producing a single
    utterance-level vector ``(B, d_model)``.
    """

    def __init__(self, d_model: int = 512, n_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        key_padding_mask = ~mask if mask is not None else None
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return out.squeeze(1)


class NovelModel(nn.Module):
    """Full C-ICMM encoder: frame projection, TAPool, encoder, fine head.

    Returns ``(embedding, logits)`` where ``embedding`` is on the unit
    hypersphere (L2-normalized).
    """

    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 512,
        embed_dim: int = 256,
        n_heads: int = 4,
        n_classes: int = 24,
    ):
        super().__init__()
        self.frame_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.tapool = TAPool(hidden_dim, n_heads)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )
        self.fine_proj = nn.Linear(hidden_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

        self.embed_dim = embed_dim
        self.n_classes = n_classes

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.frame_proj(x)
        h = self.tapool(h, mask)
        h = self.encoder(h)
        e = F.normalize(self.fine_proj(h), dim=1)
        logits = self.classifier(e)
        return e, logits
