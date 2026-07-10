#!/usr/bin/env python3
"""Hyperbolic (Poincare-ball) geometry for the open-set gate.

The 24-way head projects a pooled embedding into a ``d``-dimensional Poincare
ball and is trained with the **penalized Busemann loss** toward 24 *fixed*
ideal prototypes on the ball boundary (Ghadimi Atigh et al., "Hyperbolic
Busemann Learning with Ideal Prototypes", NeurIPS 2021).

geoopt supplies the tested manifold primitives (exp/log map, projection,
distance); this module adds only what geoopt lacks:

    * :func:`busemann` -- the Busemann function toward boundary points,
    * :class:`PenalizedBusemannLoss` -- the training loss (and, with
      ``label=None``, the margin-free gate logits used for argmax/descriptors),
    * :func:`ideal_prototypes` -- 24 maximally-separated unit vectors, computed
      once by Riesz-energy repulsion on the sphere and then frozen.

Curvature is fixed at ``c = 1`` (unit ball). All geometry runs in fp32; callers
that hold the embedding in fp16 should upcast before entering the ball.

Busemann function (ideal point ``p`` with ``||p|| = 1``)::

    B_p(z) = log( ||p - z||^2 / (1 - ||z||^2) )

so ``B_p(0) = 0`` for every prototype, ``B_p -> -inf`` as ``z -> p`` along a
geodesic, and ``B_p -> +inf`` as ``z`` approaches the boundary elsewhere.

Penalized loss for a sample ``z`` with label ``y``::

    L = B_{p_y}(z) - phi * log(1 - ||z||^2)

The ``-phi log(1 - ||z||^2)`` term is +inf at the boundary, so it stops the
minimizer collapsing onto the ideal point. Along the ray ``z = t p_y`` the
minimizer sits at radius ``t* = 1/phi``; hence ``phi`` slightly above 1 places
knowns near (but not on) the boundary and preserves radius-as-certainty. Very
large ``phi`` pulls every known toward the origin -- do not do that.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import geoopt

logger = logging.getLogger("ser.osr_gate.hyperbolic")

#: Fixed curvature of the Poincare ball (unit ball).
CURVATURE = 1.0

#: Shared manifold object; all wrappers below delegate to it.
BALL = geoopt.PoincareBall(c=CURVATURE)

#: Floor for ``1 - ||z||^2`` and squared distances, to keep logs finite.
_EPS = 1e-6

__all__ = [
    "BALL",
    "CURVATURE",
    "project",
    "expmap0",
    "logmap0",
    "poincare_radius",
    "busemann",
    "busemann_logits",
    "ideal_prototypes",
    "PenalizedBusemannLoss",
]


# --------------------------------------------------------------------------- #
# Thin geoopt wrappers (so the rest of the package imports one module)
# --------------------------------------------------------------------------- #
def project(z: torch.Tensor) -> torch.Tensor:
    """Clamp ``z`` strictly inside the ball (``||z|| <= 1 - 1e-5``)."""
    return BALL.projx(z)


def expmap0(v: torch.Tensor) -> torch.Tensor:
    """Exponential map at the origin: tangent vector ``v`` -> ball point."""
    return BALL.expmap0(v)


def logmap0(z: torch.Tensor) -> torch.Tensor:
    """Logarithmic map at the origin: ball point ``z`` -> tangent vector."""
    return BALL.logmap0(z)


def poincare_radius(z: torch.Tensor) -> torch.Tensor:
    """Hyperbolic distance from the origin, ``2 * artanh(||z||)``.

    This is gate descriptor #7 (radius-as-certainty). Returns shape ``(B,)``
    for input ``(B, d)``.
    """
    return BALL.dist0(z)


# --------------------------------------------------------------------------- #
# Busemann function (not provided by geoopt)
# --------------------------------------------------------------------------- #
def busemann(z: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Busemann function ``B_p(z)`` for every (sample, prototype) pair.

    Parameters
    ----------
    z
        Ball points, shape ``(B, d)`` with ``||z|| < 1``.
    prototypes
        Ideal boundary points, shape ``(K, d)`` with unit norm.

    Returns
    -------
    torch.Tensor
        Shape ``(B, K)``; ``[i, k] = log(||p_k - z_i||^2 / (1 - ||z_i||^2))``.
    """
    z2 = z.pow(2).sum(-1, keepdim=True)                 # (B, 1)
    one_minus = (1.0 - z2).clamp_min(_EPS)              # (B, 1)
    diff2 = torch.cdist(z, prototypes).pow(2)           # (B, K), ||p - z||^2
    return torch.log(diff2.clamp_min(_EPS)) - torch.log(one_minus)


def busemann_logits(z: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Margin-free class logits ``z_k = -B_{p_k}(z)`` for argmax / descriptors."""
    return -busemann(z, prototypes)


# --------------------------------------------------------------------------- #
# Fixed ideal prototypes
# --------------------------------------------------------------------------- #
def ideal_prototypes(
    n_classes: int,
    dim: int,
    seed: int = 0,
    steps: int = 1000,
    lr: float = 0.05,
) -> torch.Tensor:
    """Return ``n_classes`` maximally-separated unit vectors in ``R^dim``.

    Points are spread on the unit sphere by minimizing the Riesz ``s=2`` energy
    ``sum_{i != j} 1 / ||p_i - p_j||^2`` (a Thomson-style repulsion), then
    L2-normalized. Deterministic given ``seed`` (CPU generator). These are the
    frozen boundary prototypes; no Riemannian optimization is needed.
    """
    if dim < 2:
        raise ValueError("prototype dim must be >= 2")
    g = torch.Generator().manual_seed(seed)
    raw = nn.Parameter(torch.randn(n_classes, dim, generator=g))
    opt = torch.optim.Adam([raw], lr=lr)
    off_diag = ~torch.eye(n_classes, dtype=torch.bool)
    for _ in range(steps):
        opt.zero_grad()
        p = raw / raw.norm(dim=1, keepdim=True)
        d2 = torch.cdist(p, p).pow(2)[off_diag].clamp_min(_EPS)
        energy = (1.0 / d2).sum()
        energy.backward()
        opt.step()
    with torch.no_grad():
        p = raw / raw.norm(dim=1, keepdim=True)
    return p.detach()


# --------------------------------------------------------------------------- #
# Penalized Busemann loss / gate head
# --------------------------------------------------------------------------- #
class PenalizedBusemannLoss(nn.Module):
    """Penalized Busemann training loss with fixed ideal prototypes.

    ``forward(z, label)`` returns the scalar mean loss
    ``mean_i [ B_{p_{y_i}}(z_i) - phi * log(1 - ||z_i||^2) ]``.

    ``forward(z, label=None)`` returns the margin-free gate logits
    ``-B_{p_k}(z)`` of shape ``(B, K)`` (argmax = predicted class; the softmax
    of these feeds gate descriptors #1-#4, and their max is descriptor #5).

    Parameters
    ----------
    prototypes
        ``(K, d)`` unit vectors from :func:`ideal_prototypes`, stored as a
        buffer (moves with ``.to(device)``, saved in ``state_dict``, not
        trained).
    phi
        Boundary penalty strength. Along-ray minimizer radius is ``1/phi``;
        use a value slightly above 1 (default 1.1) to keep knowns near the
        boundary. Must be > 0.
    """

    def __init__(self, prototypes: torch.Tensor, phi: float = 1.1):
        super().__init__()
        if phi <= 0:
            raise ValueError("phi must be > 0")
        self.phi = float(phi)
        self.register_buffer("prototypes", prototypes.clone())

    def forward(
        self, z: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b = busemann(z, self.prototypes)                    # (B, K)
        if label is None:
            return -b
        b_y = b.gather(1, label.view(-1, 1)).squeeze(1)     # (B,)
        z2 = z.pow(2).sum(-1)                               # (B,)
        one_minus = (1.0 - z2).clamp_min(_EPS)
        loss = b_y - self.phi * torch.log(one_minus)
        return loss.mean()

    def extra_repr(self) -> str:
        k, d = tuple(self.prototypes.shape)
        return f"n_classes={k}, dim={d}, phi={self.phi}"
