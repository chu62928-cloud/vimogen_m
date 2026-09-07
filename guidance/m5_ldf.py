"""M5: Lagrangian Dual Flow with an explicit per-frame dual state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from geometry.pelvis_angle import angle_error_deg
from guidance.base import ConstraintPack, GuidanceRequest, slice_batch_stat, slice_request
from guidance.sampling_common import authoritative_normalized, clip_rms, predicted_clean


METHOD_NAME = "M5_LAGRANGIAN_DUAL_FLOW"
PROTOCOL_NAME = "vimogen_m5_lagrangian_dual_flow_c0_v1"


@dataclass(frozen=True)
class M5Config:
    primal_gain: float = 0.05
    penalty: float = 0.1
    dual_gain: float = 1.0
    max_dual_norm: float = 50.0
    gradient_clip_rms: float = 1.0
    sigma_min: float = 0.02
    sigma_max: float = 0.65
    eps: float = 1.0e-8

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "M5Config":
        cfg = cls(**dict(values or {}))
        if min(cfg.primal_gain, cfg.penalty, cfg.dual_gain) < 0:
            raise ValueError("M5 gains must be nonnegative")
        if cfg.max_dual_norm <= 0 or cfg.gradient_clip_rms <= 0:
            raise ValueError("M5 norm limits must be positive")
        if not 0 <= cfg.sigma_min <= cfg.sigma_max <= 1:
            raise ValueError("M5 sigma window must lie in [0,1]")
        return cfg


class M5LagrangianDualFlowHook:
    name = METHOD_NAME
    protocol = PROTOCOL_NAME
    requires_sigma_next = True

    def __init__(
        self, request: GuidanceRequest, *, mean: torch.Tensor, std: torch.Tensor,
        config: M5Config | Mapping[str, Any] | None = None,
    ) -> None:
        if request.constraint_pack is not ConstraintPack.C0:
            raise NotImplementedError("M5 v1 implements equality-only C0")
        self.request = request
        self.mean = mean.detach()
        self.std = std.detach()
        self.config = config if isinstance(config, M5Config) else M5Config.from_mapping(config)
        self.dual = torch.zeros_like(request.shared_evidence.target_angle_curve_deg, dtype=torch.float32)
        self.step_records: list[dict[str, Any]] = []

    def slice(self, index: int) -> "M5LagrangianDualFlowHook":
        batch = self.request.baseline_motion.shape[0]
        result = type(self)(
            slice_request(self.request, index),
            mean=slice_batch_stat(self.mean, index, batch),
            std=slice_batch_stat(self.std, index, batch), config=self.config,
        )
        result.dual = self.dual[index : index + 1].detach().clone()
        return result

    def correct_velocity(
        self, *, x_sigma: torch.Tensor, velocity: torch.Tensor,
        sigma: torch.Tensor | float, sigma_next: torch.Tensor | float,
        valid_mask: torch.Tensor, return_trace: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sigma_value = float(torch.as_tensor(sigma).detach().cpu())
        next_value = float(torch.as_tensor(sigma_next).detach().cpu())
        cfg = self.config
        record: dict[str, Any] = {"protocol": PROTOCOL_NAME, "active": False, "sigma": sigma_value}
        if sigma_value < cfg.sigma_min or sigma_value > cfg.sigma_max:
            self.step_records.append(record)
            return velocity, record
        with torch.enable_grad(), torch.amp.autocast(device_type=x_sigma.device.type, enabled=False):
            state = x_sigma.detach().float().requires_grad_(True)
            clean = predicted_clean(state, velocity.detach(), sigma)
            physical, _ = authoritative_normalized(clean, valid_mask, self.mean, self.std)
            residual = angle_error_deg(
                physical, self.request.shared_evidence.target_angle_curve_deg.to(physical.device)
            )
            dual = self.dual.to(device=state.device)
            multiplier = dual + cfg.penalty * residual.detach()
            lagrangian = (multiplier[valid_mask] * residual[valid_mask]).mean()
            gradient = torch.autograd.grad(lagrangian, state)[0]
            gradient = torch.nan_to_num(gradient) * valid_mask.unsqueeze(-1)
            gradient, gradient_rms = clip_rms(
                gradient, valid_mask, cfg.gradient_clip_rms, cfg.eps
            )
            # Sigma decreases during the sampler, so adding +grad to the flow
            # field yields a negative-gradient state displacement.
            corrected = velocity.float() + cfg.primal_gain * gradient
            corrected = torch.where(valid_mask.unsqueeze(-1), corrected, velocity.float())
            dt = abs(next_value - sigma_value)
            updated_dual = dual + cfg.dual_gain * dt * residual.detach()
            flat = updated_dual[valid_mask]
            norm = torch.linalg.vector_norm(flat)
            scale = torch.clamp(
                torch.as_tensor(cfg.max_dual_norm, device=norm.device) / norm.clamp_min(cfg.eps),
                max=1.0,
            )
            self.dual = torch.where(
                valid_mask, updated_dual * scale, torch.zeros_like(updated_dual)
            ).detach().cpu()
        record.update(
            {
                "active": True,
                "angle_residual_mae_deg": float(residual[valid_mask].abs().mean().detach().cpu()),
                "gradient_rms": float(gradient_rms.detach().cpu()),
                "dual_norm": float(torch.linalg.vector_norm(self.dual).cpu()),
                "dual_clipped": bool(float(scale.detach().cpu()) < 1.0),
                "backward_count": 1,
                "explicit_projection_used": False,
                "pseudoinverse_used": False,
                "nonfinite_count": int((~torch.isfinite(corrected)).sum().detach().cpu()),
            }
        )
        if return_trace:
            record["trace"] = {
                "velocity_model": velocity.detach().float().clone(),
                "v_cfg": velocity.detach().float().clone(),
                "x0_hat": clean.detach().clone(),
                "x0_guided": clean.detach().clone(),
                "x0_reconciled": clean.detach().clone(),
            }
        self.step_records.append({k: v for k, v in record.items() if k != "trace"})
        return corrected.to(velocity.dtype), record
