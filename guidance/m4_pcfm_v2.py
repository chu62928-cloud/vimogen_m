"""M4-v2 terminal-only control with an exact zero-dose bypass."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from guidance.base import GuidanceRequest
from guidance.m4_pcfm import M4Config, M4PCFMHook


METHOD_NAME = "M4_PCFM_TERMINAL_ONLY_V2"
PROTOCOL_NAME = "vimogen_m4_pcfm_terminal_only_c0_v2"


class M4PCFMHookV2(M4PCFMHook):
    """Retain terminal GN while disabling forward-shoot propagation by default."""

    name = METHOD_NAME
    protocol = PROTOCOL_NAME

    def __init__(
        self,
        request: GuidanceRequest,
        *,
        runtime: Any,
        mean: torch.Tensor,
        std: torch.Tensor,
        config: M4Config | Mapping[str, Any] | None = None,
    ) -> None:
        values = dict(config or {}) if not isinstance(config, M4Config) else config
        if not isinstance(values, M4Config):
            values.setdefault("shooting_sigmas", ())
            values.setdefault("propagation_gain", 0.5)
        super().__init__(
            request,
            runtime=runtime,
            mean=mean,
            std=std,
            config=values,
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
        if float(self.request.target_dose_deg) == 0.0:
            record = {
                "protocol": self.protocol,
                "sigma": float(torch.as_tensor(sigma).detach().cpu()),
                "active": False,
                "reason": "ZERO_DOSE_STRICT_BYPASS",
            }
            self.step_records.append(record)
            return velocity, record
        return super().correct_velocity(
            x_sigma=x_sigma,
            velocity=velocity,
            sigma=sigma,
            valid_mask=valid_mask,
            return_trace=return_trace,
        )

    def finalize_output(
        self, official_norm: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict[str, float]]]:
        if float(self.request.target_dose_deg) == 0.0:
            return official_norm, []
        return super().finalize_output(official_norm, valid_mask)


__all__ = ["METHOD_NAME", "PROTOCOL_NAME", "M4PCFMHookV2"]
