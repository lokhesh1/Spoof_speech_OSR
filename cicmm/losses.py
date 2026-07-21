#!/usr/bin/env python3
"""SupCon, ICMM, and centroid tracking for C-ICMM training.

Three losses with *disjoint sample sets* (Principle of Loss-Set Separation):
    * CE + SupCon operate on real training embeddings.
    * ICMM operates on synthetic interpolated embeddings only.

The composite loss is ``w_ce * L_CE + w_sc * L_SC + lambda_icmm * L_ICMM``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., 2020).

    All same-class samples in the batch are mutual positives; all
    different-class samples are negatives.  Operates on L2-normalized
    features.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=device)

        sim = features @ features.T / self.temperature

        eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = eq.float()
        pos_mask.fill_diagonal_(0)

        n_pos = pos_mask.sum(dim=1)
        valid = n_pos > 0
        if not valid.any():
            return torch.tensor(0.0, device=device)

        logits_max = sim.detach().max(dim=1, keepdim=True).values
        sim = sim - logits_max

        self_mask = torch.eye(B, device=device, dtype=torch.bool)
        exp_sim = sim.exp().masked_fill(self_mask, 0)
        log_denom = exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8).log()

        log_prob = sim - log_denom
        mean_log_prob = (log_prob * pos_mask).sum(dim=1) / n_pos.clamp(min=1)
        return -mean_log_prob[valid].mean()


class CentroidTracker:
    """Per-class centroids on the hypersphere.

    ``mode="ema"``: momentum-blended running history.
    ``mode="batch"``: centroid is recomputed from scratch every batch a
    class appears in -- no history blending, no momentum.
    """

    def __init__(self, n_classes: int, embed_dim: int,
                 momentum: float = 0.99, mode: str = "ema",
                 device: str | torch.device = "cpu"):
        self.centroids = torch.zeros(n_classes, embed_dim, device=device)
        self.initialized = torch.zeros(n_classes, dtype=torch.bool, device=device)
        self.momentum = momentum
        self.mode = mode

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        for k in range(self.centroids.shape[0]):
            mask = labels == k
            if not mask.any():
                continue
            batch_mean = F.normalize(embeddings[mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
            if self.mode == "batch" or not self.initialized[k]:
                self.centroids[k] = batch_mean
            else:
                self.centroids[k] = F.normalize(
                    (self.momentum * self.centroids[k]
                     + (1 - self.momentum) * batch_mean).unsqueeze(0),
                    dim=1,
                ).squeeze(0)
            self.initialized[k] = True

    def get(self) -> torch.Tensor:
        return self.centroids.detach()

    def all_initialized(self) -> bool:
        return self.initialized.all().item()


def generate_synthetic(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    omega_range: tuple[float, float],
    n_synthetic: int,
    pair_weight_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Generate synthetic unknown embeddings via uniform pair interpolation.

    Class pairs are drawn uniformly -- selection is not weighted. The
    per-pair ICMM weight is returned alongside so the *loss* (not the
    sampling) can be scaled by ``pair_weight_matrix``.

    Returns ``(synthetic, weights)`` -- L2-normalized ``(n_synthetic, D)``
    and ``(n_synthetic,)`` -- or ``(None, None)`` if the batch has fewer
    than two distinct classes.
    """
    device = embeddings.device
    unique_labels = labels.unique().tolist()
    if len(unique_labels) < 2:
        return None, None

    class_indices = {int(l): (labels == l).nonzero(as_tuple=True)[0] for l in unique_labels}

    pair_classes = [(li, lj) for i, li in enumerate(unique_labels)
                    for lj in unique_labels[i + 1:]]
    if not pair_classes:
        return None, None

    chosen = torch.randint(len(pair_classes), (n_synthetic,), device=device)

    omega_lo, omega_hi = omega_range
    omegas = torch.empty(n_synthetic, 1, device=device).uniform_(omega_lo, omega_hi)

    idx_a, idx_b, w = [], [], []
    for c in chosen.tolist():
        li, lj = pair_classes[c]
        ia = class_indices[li]
        ib = class_indices[lj]
        idx_a.append(ia[torch.randint(len(ia), (1,), device=device)])
        idx_b.append(ib[torch.randint(len(ib), (1,), device=device)])
        w.append(pair_weight_matrix[li, lj])

    zA = embeddings[torch.cat(idx_a)]
    zB = embeddings[torch.cat(idx_b)]
    synthetic = F.normalize(omegas * zA + (1 - omegas) * zB, dim=1)
    weights = torch.stack(w).to(device)
    return synthetic, weights


def icmm_loss(
    synthetic: torch.Tensor,
    centroids: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Push synthetic samples away from all class centroids.

    Minimises the maximum cosine similarity between each synthetic
    embedding and any centroid (the nearest-class angular margin). If
    ``weights`` is given (per-sample ICMM pair weight), each sample's
    contribution is scaled by it -- a weighted mean instead of the plain
    mean -- so class-pair weighting acts on the loss, not on which pairs
    get sampled.
    """
    sim = synthetic @ centroids.T
    per_sample = sim.max(dim=1).values
    if weights is not None:
        return (per_sample * weights).sum() / weights.sum().clamp(min=1e-8)
    return per_sample.mean()
