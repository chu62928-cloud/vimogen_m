"""Residual-adaptive root-forward guidance (v1.1).

This module deliberately subclasses the frozen v1 implementation.  The
authority boundary, endpoint projection, candidate reconstruction, loss
checks, backtracking, and finalization stay shared; only the scalar proposal
amplitude changes from a fixed normalized gradient to a geometric residual.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch

from sampling.relative_root_forward_guidance import (
    RESIDUAL_ADAPTIVE_PROTOCOL_NAME,
    RelativeRootForwardConfig,
    RelativeRootForwardGuidance,
)


PROTOCOL_NAME = RESIDUAL_ADAPTIVE_PROTOCOL_NAME


def signed_root_forward_residual_deg(
    current_forward: torch.Tensor,
    target_forward: torch.Tensor,
    frozen_right_axis: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed one-dimensional residual and its valid-frame mask.

    The vectors are projected onto the plane normal to the frozen right axis.
    Positive values mean a positive left multiplication around that axis is
    required to move the current forward vector to the target vector.
    """

    if current_forward.shape != target_forward.shape or current_forward.shape != frozen_right_axis.shape:
        raise ValueError("forward vectors and frozen axis must have identical shapes")
    if current_forward.ndim != 3 or current_forward.shape[-1] != 3:
        raise ValueError("forward vectors must have shape [B,T,3]")
    if valid_mask.shape != current_forward.shape[:2] or valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be bool[B,T]")

    axis_norm = torch.linalg.vector_norm(frozen_right_axis, dim=-1, keepdim=True)
    axis = frozen_right_axis / axis_norm.clamp_min(eps)
    current_proj = current_forward - (current_forward * axis).sum(-1, keepdim=True) * axis
    target_proj = target_forward - (target_forward * axis).sum(-1, keepdim=True) * axis
    current_norm = torch.linalg.vector_norm(current_proj, dim=-1)
    target_norm = torch.linalg.vector_norm(target_proj, dim=-1)
    frame_valid = (
        torch.isfinite(axis).all(-1)
        & torch.isfinite(current_proj).all(-1)
        & torch.isfinite(target_proj).all(-1)
        & (axis_norm[..., 0] > eps)
        & (current_norm > eps)
        & (target_norm > eps)
    )
    current_unit = current_proj / current_norm.unsqueeze(-1).clamp_min(eps)
    target_unit = target_proj / target_norm.unsqueeze(-1).clamp_min(eps)
    sine = (axis * torch.cross(current_unit, target_unit, dim=-1)).sum(-1)
    cosine = (current_unit * target_unit).sum(-1).clamp(-1.0, 1.0)
    residual_deg = torch.atan2(sine, cosine) * (180.0 / math.pi)
    return residual_deg, frame_valid & valid_mask


@dataclass(frozen=True)
class ResidualAdaptiveRootForwardConfig(RelativeRootForwardConfig):
    """Configuration for the independent residual-adaptive protocol."""

    protocol: str = PROTOCOL_NAME

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None
    ) -> "ResidualAdaptiveRootForwardConfig":
        values = dict(values or {})
        values.setdefault("protocol", PROTOCOL_NAME)
        return super().from_mapping(values)  # type: ignore[return-value]

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.protocol != PROTOCOL_NAME:
            raise ValueError(f"protocol must be {PROTOCOL_NAME}")


class ResidualAdaptiveRootForwardGuidance(RelativeRootForwardGuidance):
    """v1 guidance with residual-proportional scalar root proposals."""

    PROTOCOL = PROTOCOL_NAME

    def _make_proposal(
        self,
        *,
        base_phys: torch.Tensor,
        f_base: torch.Tensor,
        target_f: torch.Tensor,
        grad: torch.Tensor,
        grad_rms: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del base_phys, grad, grad_rms
        axis = self.targets.r0.to(device=f_base.device, dtype=f_base.dtype)
        residual_deg, residual_valid = signed_root_forward_residual_deg(
            f_base,
            target_f,
            axis,
            frame_mask,
            eps=self.config.eps,
        )
        if not bool(residual_valid[frame_mask].all()):
            return torch.zeros_like(residual_deg), {
                "proposal_rejected": True,
                "rejected_reason": "degenerate_tangent_projection",
                "invalid_residual_frames": int((frame_mask & ~residual_valid).sum().detach().cpu()),
            }
        cfg = self.config
        raw_proposal = cfg.guidance_strength * cfg.residual_gain * residual_deg
        proposal = raw_proposal.clamp(-cfg.max_step_deg, cfg.max_step_deg)
        proposal = proposal * frame_mask.to(proposal.dtype)
        abs_raw = raw_proposal.abs()[frame_mask]
        abs_proposal = proposal.abs()[frame_mask]
        diagnostics = {
            "signed_residual_mean_deg": float(residual_deg[frame_mask].mean().detach().cpu()),
            "signed_residual_rms_deg": float(
                torch.sqrt(residual_deg[frame_mask].square().mean()).detach().cpu()
            ),
            "signed_residual_p95_deg": float(
                torch.quantile(residual_deg[frame_mask].abs(), 0.95).detach().cpu()
            ),
            "proposal_mean_deg": float(proposal[frame_mask].mean().detach().cpu()),
            "proposal_rms_deg": float(torch.sqrt(abs_proposal.square().mean()).detach().cpu()),
            "proposal_max_deg": float(abs_proposal.max().detach().cpu()),
            "proposal_clipped_fraction": float(
                (abs_raw > cfg.max_step_deg + cfg.eps).to(torch.float32).mean().detach().cpu()
            ),
        }
        return proposal, diagnostics

    def protocol_record(self) -> dict[str, Any]:
        record = super().protocol_record()
        record["protocol"] = PROTOCOL_NAME
        record["proposal_rule"] = (
            "clamp(guidance_strength * residual_gain * signed_tangent_residual_deg, "
            "-max_step_deg, max_step_deg)"
        )
        record["config"] = asdict(self.config)
        return record


__all__ = [
    "PROTOCOL_NAME",
    "ResidualAdaptiveRootForwardConfig",
    "ResidualAdaptiveRootForwardGuidance",
    "signed_root_forward_residual_deg",
]
