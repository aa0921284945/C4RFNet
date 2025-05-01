# -*- coding: utf-8 -*-
"""Utility losses for point-cloud reconstruction.

* Loss_global  : squared-L2 bidirectional Chamfer distance (Eq. 20)
* Loss_local   : K-NN averaged distance (Eq. 23)
* total_loss   : Loss_global  + λ · Loss_local (Eq. 24)

The implementation keeps mask/variable-length support and falls back to a
naïve PyTorch version when PyTorch3D is unavailable (slower but portable).
"""
from __future__ import annotations

import torch
from typing import Optional, Tuple

# -----------------------------------------------------------------------------
# 1.  Try importing the fast CUDA implementation from PyTorch3D
# -----------------------------------------------------------------------------
try:
    from pytorch3d.loss import chamfer_distance as chamfer_distance_p3d  # type: ignore

    _HAS_P3D = True
except ImportError:  # pragma: no cover — keep runtime dependency optional
    _HAS_P3D = False
    print("[ChamferDistance] ⚠️  PyTorch3D not found; falling back to naïve CPU/GPU implementation (slower).")

# -----------------------------------------------------------------------------
# 2.  Helper: (B, P) boolean mask → (B,) int64 lengths  (same device)
# -----------------------------------------------------------------------------

def _to_lengths(mask: torch.Tensor) -> torch.Tensor:
    """Convert a validity **mask** to **lengths** tensor.

    Parameters
    ----------
    mask : (B, P) bool
      *True*  means valid point, *False* means padded/invalid.

    Returns
    -------
    lengths : (B,) int64
      At least *1* to avoid downstream divide-by-zero.
    """
    lengths = mask.sum(dim=1).long()
    return torch.clamp(lengths, min=1)  # keep on the same device

# -----------------------------------------------------------------------------
# 3.  Naïve Chamfer fallback (small clouds, research-friendly, autograd-safe)
# -----------------------------------------------------------------------------

def _chamfer_naive(pred: torch.Tensor,
                   gt: torch.Tensor,
                   lengths1: Optional[torch.Tensor] = None,
                   lengths2: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Straightforward O(N²) Chamfer loss, mirrors PyTorch3D semantics.

    **Squared** L2 distances, bidirectional averaging, then batch mean.
    Intended as a safe fallback when PyTorch3D is not available.
    """
    b, n1, _ = pred.shape
    _, n2, _ = gt.shape

    dists = torch.cdist(pred, gt, p=2) ** 2  # (B,N1,N2)

    # Build lengths if missing (assume dense clouds)
    if lengths1 is None:
        lengths1 = torch.full((b,), n1, dtype=torch.long, device=pred.device)
    if lengths2 is None:
        lengths2 = torch.full((b,), n2, dtype=torch.long, device=pred.device)

    losses = []
    for i in range(b):
        li1 = lengths1[i].item()
        li2 = lengths2[i].item()

        # pred → gt then gt → pred
        d1 = dists[i, :li1, :li2].min(dim=2).values.mean()
        d2 = dists[i, :li1, :li2].min(dim=1).values.mean()
        losses.append(d1 + d2)

    return torch.stack(losses).mean()

# -----------------------------------------------------------------------------
# 4.  Global Chamfer distance (Eq. 20)
# -----------------------------------------------------------------------------

def chamfer_distance(pred: torch.Tensor,
                     gt: torch.Tensor,
                     mask: Optional[torch.Tensor] = None,
                     *,
                     lengths1: Optional[torch.Tensor] = None,
                     lengths2: Optional[torch.Tensor] = None,
                     batch_reduction: str = "mean",
                     point_reduction: str = "mean") -> torch.Tensor:
    """Squared-L2 Chamfer distance identical to Eq. (20).

    If *mask* is supplied it is assumed **shared** by *pred* and *gt*.
    For differing valid point counts please pass *lengths1/lengths2* instead.
    """
    # derive lengths from mask if provided
    if mask is not None:
        lengths1 = _to_lengths(mask)
        lengths2 = lengths1

    if _HAS_P3D:
        loss, _ = chamfer_distance_p3d(
            pred, gt, lengths1, lengths2,
            batch_reduction=batch_reduction,
            point_reduction=point_reduction,
        )
        return loss

    # ── fallback ────────────────────────────────────────────────────────────
    return _chamfer_naive(pred, gt, lengths1, lengths2)

# -----------------------------------------------------------------------------
# 5.  Local neighbourhood K-NN loss (Eq. 23)
# -----------------------------------------------------------------------------

def local_knn_loss(pred: torch.Tensor,
                   gt: torch.Tensor,
                   *,
                   K: int = 20,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Average distance to *K* nearest neighbours (bidirectional).

    Handles optional **mask** by turning invalid points into NaNs so they are
    ignored during *topk* selection.
    """
    if mask is not None:
        nan = float("nan")
        pred = pred.masked_fill(~mask.unsqueeze(-1), nan)
        gt   = gt.masked_fill(~mask.unsqueeze(-1),   nan)

    dist = torch.cdist(pred, gt, p=2)  # (B, N_pred, N_gt)
    dist = torch.nan_to_num(dist, nan=float("inf"))  # ignore invalid pairs

    k = max(1, min(K, dist.shape[-1] - 1))  # ensure k ≤ N−1

    loss_pred2gt = dist.topk(k=k, largest=False).values.mean()
    loss_gt2pred = dist.transpose(1, 2).topk(k=k, largest=False).values.mean()

    return loss_pred2gt + loss_gt2pred

# -----------------------------------------------------------------------------
# 6.  Combined loss  (Eq. 24)
# -----------------------------------------------------------------------------

def total_chamfer_loss(pred: torch.Tensor,
                       gt: torch.Tensor,
                       mask: Optional[torch.Tensor] = None,
                       *,
                       local_k: int = 6,
                       lambda_local: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return *total*, *global*, *local* losses.

    total = global + λ · local
    """
    global_loss = chamfer_distance(pred, gt, mask)
    local_loss = local_knn_loss(pred, gt, K=local_k, mask=mask)
    total_loss = global_loss + lambda_local * local_loss
    return total_loss, global_loss, local_loss
