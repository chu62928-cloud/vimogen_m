"""M3-v2 sparse local projection with endpoint trust protection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from guidance.base import GuidanceRequest, slice_batch_stat, slice_request
from guidance.m3_projflow_local import M3Config, M3ProjFlowLocalHook
from guidance.sampling_common import clip_rms, velocity_from_clean


METHOD_NAME = "M3_PROJFLOW_LOCAL_V2"
PROTOCOL_NAME = "vimogen_m3_projflow_local_c0_v2"


@dataclass(frozen=True)
class M3V2Config:
    sigma_min: float = 0.10
    sigma_max: float = 0.45
    damping: float = 1.0e-6
    max_step_deg: float = 1.0
    eps: float = 1.0e-8
    projection_stride: int = 2
    max_projections: int = 4
    max_endpoint_delta_rms: float = 0.05

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "M3V2Config":
        cfg = cls(**dict(values or {}))
        M3Config(
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            damping=cfg.damping,
            max_step_deg=cfg.max_step_deg,
            eps=cfg.eps,
        )
        if cfg.projection_stride < 1 or cfg.max_projections < 1:
            raise ValueError("M3-v2 projection cadence must be positive")
        if cfg.max_endpoint_delta_rms <= 0:
            raise ValueError("M3-v2 endpoint trust radius must be positive")
        return cfg

    def base(self) -> M3Config:
        return M3Config(
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            damping=self.damping,
            max_step_deg=self.max_step_deg,
            eps=self.eps,
        )


class M3ProjFlowLocalHookV2(M3ProjFlowLocalHook):
    name = METHOD_NAME
    protocol = PROTOCOL_NAME

    def __init__(
        self,
        request: GuidanceRequest,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
        config: M3V2Config | Mapping[str, Any] | None = None,
    ) -> None:
        self.v2_config = (
            config if isinstance(config, M3V2Config) else M3V2Config.from_mapping(config)
        )
        self._eligible_calls = 0
        self._projection_count = 0
        super().__init__(
            request,
            mean=mean,
            std=std,
            config=self.v2_config.base(),
        )

    def slice(self, index: int) -> "M3ProjFlowLocalHookV2":
        batch = self.request.baseline_motion.shape[0]
        return type(self)(
            slice_request(self.request, index),
            mean=slice_batch_stat(self.mean, index, batch),
            std=slice_batch_stat(self.std, index, batch),
            config=self.v2_config,
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
        if float(self.request.target_dose_deg) == 0.0:
            record = {
                "protocol": self.protocol,
                "active": False,
                "sigma": sigma_value,
                "reason": "ZERO_DOSE_STRICT_BYPASS",
                "projection_count": self._projection_count,
            }
            self.step_records.append(record)
            return velocity, record
        in_window = self.config.sigma_min <= sigma_value <= self.config.sigma_max
        if in_window:
            call_index = self._eligible_calls
            self._eligible_calls += 1
            if call_index % self.v2_config.projection_stride != 0:
                record = {
                    "protocol": self.protocol,
                    "active": False,
                    "sigma": sigma_value,
                    "reason": "SPARSE_CADENCE_SKIP",
                    "projection_count": self._projection_count,
                }
                self.step_records.append(record)
                return velocity, record
            if self._projection_count >= self.v2_config.max_projections:
                record = {
                    "protocol": self.protocol,
                    "active": False,
                    "sigma": sigma_value,
                    "reason": "PROJECTION_BUDGET_EXHAUSTED",
                    "projection_count": self._projection_count,
                }
                self.step_records.append(record)
                return velocity, record

        corrected, record = super().correct_velocity(
            x_sigma=x_sigma,
            velocity=velocity,
            sigma=sigma,
            valid_mask=valid_mask,
            return_trace=True,
        )
        if record.get("active"):
            trace = record["trace"]
            delta = trace["x0_guided"] - trace["x0_hat"]
            clipped, original_rms = clip_rms(
                delta, valid_mask, self.v2_config.max_endpoint_delta_rms
            )
            guided = trace["x0_hat"] + clipped
            corrected = velocity_from_clean(x_sigma, guided, sigma)
            corrected = torch.where(valid_mask.unsqueeze(-1), corrected, velocity.float())
            self._projection_count += 1
            record.update(
                {
                    "protocol": self.protocol,
                    "projection_count": self._projection_count,
                    "endpoint_delta_rms_before_clip": float(original_rms.cpu()),
                    "endpoint_delta_rms_after_clip": float(
                        torch.sqrt(clipped[valid_mask].square().mean()).cpu()
                    ),
                    "endpoint_trust_region_hit": bool(
                        original_rms > self.v2_config.max_endpoint_delta_rms
                    ),
                }
            )
            trace["x0_guided"] = guided.detach().clone()
            trace["x0_reconciled"] = guided.detach().clone()
            self.step_records[-1] = {
                key: value for key, value in record.items() if key != "trace"
            }
        if not return_trace:
            record.pop("trace", None)
        return corrected.to(velocity.dtype), record


__all__ = ["METHOD_NAME", "PROTOCOL_NAME", "M3V2Config", "M3ProjFlowLocalHookV2"]
