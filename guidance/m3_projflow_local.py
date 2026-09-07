"""M3 local endpoint projection for the scalar C0 pelvis constraint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from geometry.authoritative_motion import authoritative_motion
from geometry.pelvis_angle import pelvis_angle_curve_deg, wrap_angle_deg
from guidance.base import ConstraintPack, GuidanceRequest, slice_batch_stat, slice_request
from guidance.sampling_common import (
    authoritative_normalized,
    masked_rms,
    normalized_from_physical,
    predicted_clean,
    velocity_from_clean,
)
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe, encode_rot6d
from motion_rep.sagittal_pelvis_angle import apply_person_right_axis_rotation


METHOD_NAME = "M3_PROJFLOW_LOCAL"
PROTOCOL_NAME = "vimogen_m3_projflow_local_c0_v1"


@dataclass(frozen=True)
class M3Config:
    sigma_min: float = 0.0662879
    sigma_max: float = 0.65
    damping: float = 1.0e-6
    max_step_deg: float = 2.0
    eps: float = 1.0e-8

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "M3Config":
        cfg = cls(**dict(values or {}))
        if cfg.damping < 0 or cfg.max_step_deg <= 0:
            raise ValueError("M3 damping must be nonnegative and max step positive")
        if not 0 <= cfg.sigma_min <= cfg.sigma_max <= 1:
            raise ValueError("M3 sigma window must lie in [0,1]")
        return cfg


class M3ProjFlowLocalHook:
    name = METHOD_NAME
    protocol = PROTOCOL_NAME

    def __init__(
        self,
        request: GuidanceRequest,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
        config: M3Config | Mapping[str, Any] | None = None,
    ) -> None:
        if request.constraint_pack is not ConstraintPack.C0:
            raise NotImplementedError("M3 v1 implements C0 only")
        self.request = request
        self.mean = mean.detach()
        self.std = std.detach()
        self.config = config if isinstance(config, M3Config) else M3Config.from_mapping(config)
        self.step_records: list[dict[str, Any]] = []

    def slice(self, index: int) -> "M3ProjFlowLocalHook":
        batch = self.request.baseline_motion.shape[0]
        return type(self)(
            slice_request(self.request, index),
            mean=slice_batch_stat(self.mean, index, batch),
            std=slice_batch_stat(self.std, index, batch),
            config=self.config,
        )

    def correct_velocity(
        self,
        *,
        x_sigma: torch.Tensor,
        velocity: torch.Tensor,
        sigma: torch.Tensor | float,
        valid_mask: torch.Tensor,
        return_trace: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sigma_value = float(torch.as_tensor(sigma).detach().cpu())
        cfg = self.config
        record: dict[str, Any] = {"protocol": PROTOCOL_NAME, "active": False, "sigma": sigma_value}
        if sigma_value < cfg.sigma_min or sigma_value > cfg.sigma_max or sigma_value <= cfg.eps:
            self.step_records.append(record)
            return velocity, record
        with torch.no_grad(), torch.amp.autocast(device_type=x_sigma.device.type, enabled=False):
            clean = predicted_clean(x_sigma, velocity, sigma)
            physical, _ = authoritative_normalized(clean, valid_mask, self.mean, self.std)
            actual = pelvis_angle_curve_deg(physical)
            target = self.request.shared_evidence.target_angle_curve_deg.to(actual.device)
            residual = wrap_angle_deg(target - actual)
            # In C0, d(phi)/d(theta_right)=1. The damped scalar normal
            # equation is the exact local linear projection.
            increment_deg = (residual / (1.0 + cfg.damping)).clamp(
                -cfg.max_step_deg, cfg.max_step_deg
            )
            increment_deg = torch.where(valid_mask, increment_deg, torch.zeros_like(increment_deg))
            root = decode_rot6d_safe(physical[..., MOTION_LAYOUT.root_rotation])
            edited = physical.clone()
            edited[..., MOTION_LAYOUT.root_rotation] = encode_rot6d(
                apply_person_right_axis_rotation(root, increment_deg)
            )
            projected = authoritative_motion(edited, valid_mask=valid_mask).motion
            projected_norm = normalized_from_physical(projected, self.mean, self.std)
            corrected = velocity_from_clean(x_sigma, projected_norm, sigma)
            corrected = torch.where(valid_mask.unsqueeze(-1), corrected, velocity.float())
        record.update(
            {
                "active": True,
                "angle_residual_mae_deg": float(residual[valid_mask].abs().mean().cpu()),
                "projected_increment_rms_deg": float(masked_rms(increment_deg, valid_mask).cpu()),
                "projected_increment_max_deg": float(increment_deg[valid_mask].abs().max().cpu()),
                "jacobian_rank": int(valid_mask.sum().item()),
                "solver_iterations": 1,
                "nonfinite_count": int((~torch.isfinite(corrected)).sum().cpu()),
            }
        )
        if return_trace:
            record["trace"] = {
                "velocity_model": velocity.detach().float().clone(),
                "v_cfg": velocity.detach().float().clone(),
                "x0_hat": clean.detach().clone(),
                "x0_guided": projected_norm.detach().clone(),
                "x0_reconciled": projected_norm.detach().clone(),
            }
        self.step_records.append({key: value for key, value in record.items() if key != "trace"})
        return corrected.to(velocity.dtype), record
