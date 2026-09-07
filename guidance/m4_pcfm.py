"""M4: PCFM-style forward shooting and terminal Gauss--Newton correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from geometry.authoritative_motion import authoritative_motion
from geometry.pelvis_angle import pelvis_angle_curve_deg, wrap_angle_deg
from guidance.base import ConstraintPack, GuidanceRequest, slice_batch_stat, slice_request
from guidance.sampling_common import align_stat, masked_rms, normalized_from_physical, velocity_from_clean
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe, encode_rot6d
from motion_rep.sagittal_pelvis_angle import apply_person_right_axis_rotation


METHOD_NAME = "M4_PCFM"
PROTOCOL_NAME = "vimogen_m4_pcfm_c0_v1"


@dataclass(frozen=True)
class M4Config:
    shooting_sigmas: tuple[float, ...] = (0.55, 0.35, 0.15)
    gn_iterations: int = 3
    damping: float = 1.0e-5
    trust_radius_deg: float = 4.0
    propagation_gain: float = 1.0
    terminal_tolerance_deg: float = 1.0e-4
    eps: float = 1.0e-8

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "M4Config":
        data = dict(values or {})
        if "shooting_sigmas" in data:
            data["shooting_sigmas"] = tuple(float(x) for x in data["shooting_sigmas"])
        cfg = cls(**data)
        if cfg.gn_iterations < 1 or cfg.damping < 0 or cfg.trust_radius_deg <= 0:
            raise ValueError("M4 GN settings are invalid")
        if not 0 < cfg.propagation_gain <= 1:
            raise ValueError("M4 propagation_gain must lie in (0,1]")
        if any(not 0 <= sigma <= 1 for sigma in cfg.shooting_sigmas):
            raise ValueError("M4 shooting sigmas must lie in [0,1]")
        return cfg


def _terminal_gn(
    motion: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    config: M4Config,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    candidate = authoritative_motion(motion, valid_mask=valid_mask).motion
    records: list[dict[str, float]] = []
    for iteration in range(config.gn_iterations):
        actual = pelvis_angle_curve_deg(candidate)
        residual = wrap_angle_deg(target.to(actual.device) - actual)
        # C0 has the analytic local Jacobian d(phi)/d(theta_right)=1.
        step = (residual / (1.0 + config.damping)).clamp(
            -config.trust_radius_deg, config.trust_radius_deg
        )
        step = torch.where(valid_mask, step, torch.zeros_like(step))
        root = decode_rot6d_safe(candidate[..., MOTION_LAYOUT.root_rotation])
        candidate = candidate.clone()
        candidate[..., MOTION_LAYOUT.root_rotation] = encode_rot6d(
            apply_person_right_axis_rotation(root, step)
        )
        candidate = authoritative_motion(candidate, valid_mask=valid_mask).motion
        records.append(
            {
                "iteration": float(iteration),
                "residual_mae_deg": float(residual[valid_mask].abs().mean().cpu()),
                "step_rms_deg": float(masked_rms(step, valid_mask).cpu()),
            }
        )
        if float(residual[valid_mask].abs().max().cpu()) <= config.terminal_tolerance_deg:
            break
    return candidate, records


class M4PCFMHook:
    """The runtime must expose ``forward_shoot`` for a complete remaining rollout."""

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
        if request.constraint_pack is not ConstraintPack.C0:
            raise NotImplementedError("M4 v1 implements C0 only")
        if not hasattr(runtime, "forward_shoot"):
            raise TypeError("M4 runtime must expose forward_shoot")
        self.request = request
        self.runtime = runtime
        self.mean = mean.detach()
        self.std = std.detach()
        self.config = config if isinstance(config, M4Config) else M4Config.from_mapping(config)
        self.step_records: list[dict[str, Any]] = []

    def slice(self, index: int) -> "M4PCFMHook":
        batch = self.request.baseline_motion.shape[0]
        runtime = self.runtime.slice(index) if hasattr(self.runtime, "slice") else self.runtime
        return type(self)(
            slice_request(self.request, index), runtime=runtime,
            mean=slice_batch_stat(self.mean, index, batch),
            std=slice_batch_stat(self.std, index, batch), config=self.config,
        )

    def correct_velocity(
        self, *, x_sigma: torch.Tensor, velocity: torch.Tensor,
        sigma: torch.Tensor | float, valid_mask: torch.Tensor,
        return_trace: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sigma_value = float(torch.as_tensor(sigma).detach().cpu())
        tolerance = max(1.0e-6, 0.5 / max(len(self.config.shooting_sigmas), 1))
        active = any(abs(sigma_value - point) <= tolerance for point in self.config.shooting_sigmas)
        record: dict[str, Any] = {"protocol": PROTOCOL_NAME, "sigma": sigma_value, "active": active}
        if not active or sigma_value <= self.config.eps:
            self.step_records.append(record)
            return velocity, record
        with torch.no_grad(), torch.amp.autocast(device_type=x_sigma.device.type, enabled=False):
            terminal = self.runtime.forward_shoot(
                x_sigma=x_sigma.detach(), sigma=sigma, request=self.request
            )
            if hasattr(terminal, "motion"):
                terminal = terminal.motion
            if not isinstance(terminal, torch.Tensor) or terminal.shape != x_sigma.shape:
                raise ValueError("M4 forward shooting returned an invalid terminal motion")
            # Runtime terminal states are standardized; the GN solve is physical.
            mean = self.mean.to(x_sigma.device)
            std = self.std.to(x_sigma.device)
            physical = terminal.float() * align_stat(std, terminal, "std") + align_stat(
                mean, terminal, "mean"
            )
            corrected_physical, gn_records = _terminal_gn(
                physical, self.request.shared_evidence.target_angle_curve_deg,
                valid_mask, self.config,
            )
            corrected_terminal = normalized_from_physical(corrected_physical, self.mean, self.std)
            propagated = terminal.float() + self.config.propagation_gain * (
                corrected_terminal - terminal.float()
            )
            corrected = velocity_from_clean(x_sigma, propagated, sigma)
            corrected = torch.where(valid_mask.unsqueeze(-1), corrected, velocity.float())
        record.update(
            {
                "full_rollout_count": 1,
                "solver_iterations": len(gn_records),
                "gn_records": gn_records,
                "terminal_update_rms": float(masked_rms(corrected_terminal - terminal, valid_mask).cpu()),
                "nonfinite_count": int((~torch.isfinite(corrected)).sum().cpu()),
            }
        )
        if return_trace:
            record["trace"] = {
                "velocity_model": velocity.detach().float().clone(),
                "v_cfg": velocity.detach().float().clone(),
                "x0_hat": terminal.detach().float().clone(),
                "x0_guided": propagated.detach().clone(),
                "x0_reconciled": propagated.detach().clone(),
            }
        self.step_records.append({k: v for k, v in record.items() if k != "trace"})
        return corrected.to(velocity.dtype), record

    def finalize_output(
        self, official_norm: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict[str, float]]]:
        mean = self.mean.to(official_norm.device)
        std = self.std.to(official_norm.device)
        physical = official_norm.float() * align_stat(
            std, official_norm, "std"
        ) + align_stat(mean, official_norm, "mean")
        projected, records = _terminal_gn(
            physical, self.request.shared_evidence.target_angle_curve_deg,
            valid_mask, self.config,
        )
        return normalized_from_physical(projected, self.mean, self.std), records
