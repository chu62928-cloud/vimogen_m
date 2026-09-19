"""Compatibility endpoint for the recovered historical M1 protocol.

The relative-root-forward v1 protocol does not import this module.  It is
kept solely so the server-traced M1 implementation remains importable and its
optional ``velocity_authoritative_v2`` mode can still finalise an endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .consistent_finalizer import finalize_consistent_motion_tensor


@dataclass(frozen=True)
class ReconciledEndpoint:
    motion: torch.Tensor
    valid_mask: torch.Tensor


def reconcile_guided_endpoint(
    baseline_norm: torch.Tensor,
    guided_norm: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    output_dtype: torch.dtype | None = torch.float32,
) -> ReconciledEndpoint:
    """Apply the historical pose-authoritative endpoint finalizer."""

    del baseline_norm  # retained in the public signature for trace compatibility
    result = finalize_consistent_motion_tensor(
        guided_norm,
        valid_mask=valid_mask,
        mean=mean,
        std=std,
        input_standardized=True,
        output_standardized=True,
        output_dtype=output_dtype,
    )
    return ReconciledEndpoint(result.motion, result.valid_mask)


__all__ = ["ReconciledEndpoint", "reconcile_guided_endpoint"]
