#!/usr/bin/env python3
"""Hier-Spec model: AASIST backend + ArcFace architecture/model heads.

Embedding path (frozen Wav2Vec2 already applied during feature extraction):

    (B, T, 768) cached frames -> AasistBackend -> L2-normalised (B, 160) x
        -> ArcMarginHead (architecture, 13 classes)
        -> ModuleDict of ArcMarginHead model heads, one per multi-model arch
           {XTTS:2, Tacotron:5, VITS:6, Bark:2}

Training is teacher-forced: the model loss for a sample is computed by the head
of its **ground-truth** architecture (singletons contribute only the arch loss),
so model-head gradients are never polluted by arch-routing mistakes. The joint
objective is ``L = 0.5 * (L_arch + L_model)``.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aasist import AasistBackend


class ArcMarginHead(nn.Module):
    """Bias-free linear head with ArcFace additive angular margin.

    Features and weights are L2-normalised, so the logits are ``s * cos(theta)``.
    With a target ``label`` the margin ``m`` is added to the target angle for the
    training loss; ``label=None`` returns the plain (marginless) cosine logits
    used for prediction (argmax) and is margin-free.
    """

    def __init__(self, in_dim: int, n_classes: int, s: float = 30.0,
                 m: float = 0.3):
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_dim))
        nn.init.xavier_normal_(self.weight)
        self._cos_m = math.cos(m)
        self._sin_m = math.sin(m)
        self._th = math.cos(math.pi - m)      # decision boundary for monotonicity
        self._mm = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor,
                label: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(x), F.normalize(self.weight)).clamp(-1, 1)
        if label is None:
            return cosine * self.s
        sine = torch.sqrt((1.0 - cosine ** 2).clamp_min(1e-9))
        phi = cosine * self._cos_m - sine * self._sin_m
        # keep the function monotonic outside the valid angular range
        phi = torch.where(cosine > self._th, phi, cosine - self._mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s


class HierSpecModel(nn.Module):
    def __init__(self, arch_label_map: Dict[str, int],
                 model_label_maps: Dict[str, Dict[str, int]],
                 multi_model_archs: Tuple[str, ...],
                 feat_dim: int = 768, s: float = 30.0, m: float = 0.3):
        super().__init__()
        self.backend = AasistBackend(feat_dim=feat_dim)
        embed_dim = self.backend.out_dim
        self.embed_dim = embed_dim

        self.arch_label_map = dict(arch_label_map)
        self.idx_to_arch = {i: a for a, i in self.arch_label_map.items()}
        self.multi_model_archs = tuple(multi_model_archs)

        self.arch_head = ArcMarginHead(embed_dim, len(arch_label_map), s=s, m=m)
        self.model_heads = nn.ModuleDict({
            arch: ArcMarginHead(embed_dim, len(model_label_maps[arch]), s=s, m=m)
            for arch in self.multi_model_archs
        })
        # arch index -> arch name, restricted to multi-model archs
        self._multi_arch_idx = {self.arch_label_map[a]: a
                                for a in self.multi_model_archs}

    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, T, feat_dim) -> L2-normalised (B, embed_dim) embedding."""
        h = self.backend(feats)
        return F.normalize(h, dim=1)

    def forward(self, feats: torch.Tensor, arch_labels: torch.Tensor,
                model_local_labels: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced joint forward.

        ``model_local_labels`` is the within-architecture model index for
        multi-model-arch samples and ``-1`` for singleton-arch samples.
        Returns ``(L_arch, L_model, arch_logits_marginless)``.
        """
        x = self.embed(feats)
        arch_logits_margin = self.arch_head(x, arch_labels)
        l_arch = F.cross_entropy(arch_logits_margin, arch_labels)

        # model-level loss over multi-model-arch samples, routed by GT arch
        total_loss = feats.new_zeros(())
        total_n = 0
        for arch_idx, arch in self._multi_arch_idx.items():
            mask = arch_labels == arch_idx
            n = int(mask.sum())
            if n == 0:
                continue
            logits = self.model_heads[arch](x[mask], model_local_labels[mask])
            total_loss = total_loss + F.cross_entropy(
                logits, model_local_labels[mask], reduction="sum")
            total_n += n
        l_model = total_loss / total_n if total_n > 0 else feats.new_zeros(())

        with torch.no_grad():
            arch_logits = self.arch_head(x)      # marginless, for monitoring
        return l_arch, l_model, arch_logits

    @torch.no_grad()
    def predict_logits(self, feats: torch.Tensor
                       ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor],
                                  torch.Tensor]:
        """Marginless arch logits, per-arch model logits, and the embedding."""
        x = self.embed(feats)
        arch_logits = self.arch_head(x)
        model_logits = {arch: head(x) for arch, head in self.model_heads.items()}
        return arch_logits, model_logits, x
