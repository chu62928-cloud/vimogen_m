"""M1 flow-energy guidance with a frozen per-frame pelvis target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from guidance.base import ConstraintPack, GuidanceRequest, slice_batch_stat, slice_request
from guidance.sampling_common import (
    authoritative_normalized,
    clip_rms,
    masked_angle_loss,
    masked_rms,
    predicted_clean,
)


METHOD_NAME = "M1_FLOW_ENERGY_LOSS_GUIDANCE"
PROTOCOL_NAME = "vimogen_m1_flow_energy_loss_guidance_c0_v1"


@dataclass(frozen=True)
class M1Config:
    guidance_scale: float = 0.05
    gradient_clip_rms: float = 1.0
    sigma_min: float = 0.0662879
    sigma_max: float = 0.65
    line_search_steps: int = 12
    eps: float = 1.0e-8

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "M1Config":
        cfg = cls(**dict(values or {}))
        if cfg.guidance_scale < 0 or cfg.gradient_clip_rms <= 0:
            raise ValueError("M1 scale must be nonnegative and clip positive")
        if not 0 <= cfg.sigma_min <= cfg.sigma_max <= 1:
            raise ValueError("M1 sigma window must lie in [0,1]")
        if cfg.line_search_steps < 1:
            raise ValueError("M1 line_search_steps must be positive")
        return cfg


class M1LossGuidanceHook:
    name = METHOD_NAME
    protocol = PROTOCOL_NAME
    requires_sigma_next = True

    def __init__(
        self,
        request: GuidanceRequest,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
        config: M1Config | Mapping[str, Any] | None = None,
    ) -> None:
        if request.constraint_pack is not ConstraintPack.C0:
            raise NotImplementedError("M1 v1 implements C0 only")
        self.request = request
        self.mean = mean.detach()
        self.std = std.detach()
        self.config = config if isinstance(config, M1Config) else M1Config.from_mapping(config)
        self.step_records: list[dict[str, Any]] = []

    def slice(self, index: int) -> "M1LossGuidanceHook":
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
        sigma_next: torch.Tensor | float,
        valid_mask: torch.Tensor,
        return_trace: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sigma_value = float(torch.as_tensor(sigma).detach().cpu())
        next_value = float(torch.as_tensor(sigma_next).detach().cpu())
        cfg = self.config
        record: dict[str, Any] = {
            "protocol": PROTOCOL_NAME,
            "active": False,
            "sigma": sigma_value,
            "sigma_next": next_value,
        }
        if (
            cfg.guidance_scale <= cfg.eps
            or sigma_value < cfg.sigma_min
            or sigma_value > cfg.sigma_max
            or abs(next_value - sigma_value) <= cfg.eps
        ):
            self.step_records.append(record)
            return velocity, record
        with torch.enable_grad(), torch.amp.autocast(
            device_type=x_sigma.device.type, enabled=False
        ):
            state = x_sigma.detach().float().requires_grad_(True)
            clean = predicted_clean(state, velocity.detach(), sigma)
            physical, _ = authoritative_normalized(clean, valid_mask, self.mean, self.std)
            loss, residual = masked_angle_loss(
                physical,
                self.request.shared_evidence.target_angle_curve_deg,
                valid_mask,
            )
            gradient = torch.autograd.grad(loss, state)[0]
            gradient = torch.nan_to_num(gradient) * valid_mask.unsqueeze(-1)
            clipped, gradient_rms = clip_rms(
                gradient, valid_mask, cfg.gradient_clip_rms, cfg.eps
            )
            state_delta = -cfg.guidance_scale * clipped
            accepted_scale = 1.0
            rejected_steps = 0
            with torch.no_grad():
                for _ in range(cfg.line_search_steps):
                    trial_clean = predicted_clean(
                        state.detach() + accepted_scale * state_delta.detach(),
                        velocity.detach(),
                        sigma,
                    )
                    trial_physical, _ = authoritative_normalized(
                        trial_clean, valid_mask, self.mean, self.std
                    )
                    trial_loss, _ = masked_angle_loss(
                        trial_physical,
                        self.request.shared_evidence.target_angle_curve_deg,
                        valid_mask,
                    )
                    if trial_loss <= loss.detach() + cfg.eps:
                        break
                    accepted_scale *= 0.5
                    rejected_steps += 1
                else:
                    accepted_scale = 0.0
                state_delta = state_delta * accepted_scale
            corrected = velocity.float() + state_delta / (next_value - sigma_value)
            corrected = torch.where(valid_mask.unsqueeze(-1), corrected, velocity.float())
        record.update(
            {
                "active": True,
                "loss": float(loss.detach().cpu()),
                "angle_residual_mae_deg": float(residual[valid_mask].abs().mean().detach().cpu()),
                "gradient_rms": float(gradient_rms.detach().cpu()),
                "state_update_rms": float(masked_rms(state_delta, valid_mask).detach().cpu()),
                "accepted_scale": float(accepted_scale),
                "rejected_steps": int(rejected_steps),
                "backward_count": 1,
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
        self.step_records.append({key: value for key, value in record.items() if key != "trace"})
        return corrected.to(velocity.dtype), record
