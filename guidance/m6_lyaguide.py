"""M6 LyaGuide pseudo-projection with the same C0 energy as M1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from guidance.base import ConstraintPack, GuidanceRequest, slice_batch_stat, slice_request
from guidance.sampling_common import (
    authoritative_normalized,
    clip_rms,
    masked_angle_loss,
    predicted_clean,
)


METHOD_NAME = "M6_LYAGUIDE"
PROTOCOL_NAME = "vimogen_m6_lyaguide_c0_v1"


@dataclass(frozen=True)
class M6Config:
    candidate_scale: float = 0.05
    delta: float = 0.1
    gradient_clip_rms: float = 1.0
    sigma_min: float = 0.0662879
    sigma_max: float = 0.65
    eps: float = 1.0e-8

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "M6Config":
        cfg = cls(**dict(values or {}))
        if min(cfg.candidate_scale, cfg.delta) < 0 or cfg.gradient_clip_rms <= 0:
            raise ValueError("M6 scales must be nonnegative and clip positive")
        if not 0 <= cfg.sigma_min <= cfg.sigma_max <= 1:
            raise ValueError("M6 sigma window must lie in [0,1]")
        return cfg


class M6LyaGuideHook:
    name = METHOD_NAME
    protocol = PROTOCOL_NAME

    def __init__(
        self,
        request: GuidanceRequest,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
        config: M6Config | Mapping[str, Any] | None = None,
    ) -> None:
        if request.constraint_pack is not ConstraintPack.C0:
            raise NotImplementedError("M6 v1 implements C0 only")
        self.request = request
        self.mean = mean.detach()
        self.std = std.detach()
        self.config = config if isinstance(config, M6Config) else M6Config.from_mapping(config)
        self.step_records: list[dict[str, Any]] = []

    def slice(self, index: int) -> "M6LyaGuideHook":
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
        if sigma_value < cfg.sigma_min or sigma_value > cfg.sigma_max:
            self.step_records.append(record)
            return velocity, record
        with torch.enable_grad(), torch.amp.autocast(device_type=x_sigma.device.type, enabled=False):
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
            gradient, gradient_rms = clip_rms(
                gradient, valid_mask, cfg.gradient_clip_rms, cfg.eps
            )
            # FlowSampler integrates from larger to smaller sigma.  The
            # Lyapunov derivative is therefore written in the descending-
            # sigma direction: a positive state gradient gives a positive
            # velocity correction because ``sigma_next - sigma < 0``.
            candidate = cfg.candidate_scale * gradient
            flat_mask = valid_mask.unsqueeze(-1).expand_as(gradient)
            g = gradient[flat_mask]
            u = velocity.float()[flat_mask]
            c = candidate[flat_mask]
            before = -(g * (u + c)).sum() + cfg.delta * loss
            coefficient = torch.relu(before) / (g.square().sum() + cfg.eps)
            projected = candidate + coefficient * gradient
            corrected = velocity.float() + projected
            corrected = torch.where(valid_mask.unsqueeze(-1), corrected, velocity.float())
            after = -(g * (u + c + coefficient * g)).sum() + cfg.delta * loss
        record.update(
            {
                "active": True,
                "loss": float(loss.detach().cpu()),
                "angle_residual_mae_deg": float(residual[valid_mask].abs().mean().detach().cpu()),
                "gradient_rms": float(gradient_rms.detach().cpu()),
                "pseudo_projection_coefficient": float(coefficient.detach().cpu()),
                "lyapunov_condition_before": float(before.detach().cpu()),
                "lyapunov_condition_after": float(after.detach().cpu()),
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
