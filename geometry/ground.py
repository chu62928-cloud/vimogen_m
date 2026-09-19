"""Frozen ground estimation shared by every guidance method."""

from __future__ import annotations

import torch


GROUND_VERSION = "m1_m7_ground_quantile_v1"


def estimate_ground_height(
    marker_positions: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    up_axis: int = 2,
    quantile: float = 0.02,
) -> torch.Tensor:
    """Estimate one ground height per sequence from M0 foot markers.

    ``marker_positions`` is ``[B,T,M,3]``.  Only M0 and the shared valid mask
    may be passed here; callers freeze the result before running any method.
    """

    if marker_positions.ndim != 4 or marker_positions.shape[-1] != 3:
        raise ValueError("marker_positions must have shape [B,T,M,3]")
    if valid_mask.dtype is not torch.bool or valid_mask.shape != marker_positions.shape[:2]:
        raise ValueError("valid_mask must be bool[B,T]")
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be 0, 1, or 2")
    if not 0.0 <= float(quantile) <= 0.5:
        raise ValueError("ground quantile must lie in [0,0.5]")
    heights = marker_positions[..., up_axis]
    rows = []
    for index in range(marker_positions.shape[0]):
        values = heights[index][valid_mask[index]].reshape(-1)
        if not values.numel() or not torch.isfinite(values).all():
            raise ValueError("valid marker heights must be finite and non-empty")
        rows.append(torch.quantile(values, float(quantile)))
    return torch.stack(rows)
