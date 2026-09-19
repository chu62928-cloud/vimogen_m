"""Minimal M1 approximate clean-endpoint guidance.

The model forward remains under the sampler's ``no_grad`` context.  Only the
detached clean-endpoint estimate receives a short-lived autograd graph.  This
module deliberately exposes the three flow operations used by the sampler so
the M0 path can remain unchanged when guidance is absent or disabled.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

import torch

import motion_rep.m1_consistent as m1_consistency
from motion_rep.phase1 import decode_rot6d_safe
from motion_rep.pelvis_angle import pelvis_pitch_degrees


def _broadcast_sigma(sigma: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(sigma, dtype=reference.dtype, device=reference.device)
    while value.ndim < reference.ndim:
        value = value.unsqueeze(-1)
    return value


def predict_x0(
    x_sigma: torch.Tensor, velocity: torch.Tensor, sigma: torch.Tensor | float
) -> torch.Tensor:
    """Estimate the clean endpoint ``x0_hat = x_sigma - sigma * v``."""

    if x_sigma.shape != velocity.shape:
        raise ValueError("x_sigma and velocity must have identical shapes")
    return x_sigma - _broadcast_sigma(sigma, x_sigma) * velocity


def velocity_from_x0(
    x_sigma: torch.Tensor,
    x0: torch.Tensor,
    sigma: torch.Tensor | float,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Recompose a flow velocity from a corrected clean endpoint."""

    if x_sigma.shape != x0.shape:
        raise ValueError("x_sigma and x0 must have identical shapes")
    sigma_value = _broadcast_sigma(sigma, x_sigma)
    if torch.any(sigma_value.abs() < eps):
        raise ValueError("velocity_from_x0 is undefined at sigma=0")
    return (x_sigma - x0) / sigma_value


@dataclass(frozen=True)
class M1Config:
    enabled: bool = True
    lambda_scale: float = 0.5
    sigma_min: float = 0.15
    sigma_max: float = 0.75
    angle_weight: float = 1.0
    hold_weight: float = 0.1
    max_correction_rms: float = 0.05
    min_speed: float = 0.05 / 20.0
    # ``travel`` uses root-translation heading; ``canonical_y`` is an
    # explicit model-space +y heading for stationary/non-locomotion clips.
    heading_mode: str = "travel"
    # ``legacy`` preserves every historical M1 result.  The opt-in v2 mode
    # reconciles each guided clean endpoint through an explicit physical
    # T+1 pose stream before the flow velocity is recomposed.
    consistency_mode: str = "legacy"
    # Step-level tensors are an explicitly opt-in diagnostic.  Keeping this
    # disabled by default is important: the normal M0/M1 path must not clone
    # or move any extra tensors and must remain bitwise unchanged.
    trace_enabled: bool = False
    eps: float = 1e-6

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "M1Config":
        """Parse the public YAML/OmegaConf M1 section in one place."""

        values = values or {}
        defaults = cls()
        return cls(
            # Configuration absence has always meant "no M1 hook", even
            # though direct construction of M1Config is convenient as an
            # enabled strategy in unit tests and explicit callers.
            enabled=bool(values.get("enabled", False)),
            lambda_scale=float(values.get("lambda_scale", defaults.lambda_scale)),
            sigma_min=float(values.get("sigma_min", defaults.sigma_min)),
            sigma_max=float(values.get("sigma_max", defaults.sigma_max)),
            angle_weight=float(values.get("angle_weight", defaults.angle_weight)),
            hold_weight=float(values.get("hold_weight", defaults.hold_weight)),
            max_correction_rms=float(
                values.get("max_correction_rms", defaults.max_correction_rms)
            ),
            min_speed=float(values.get("min_speed", defaults.min_speed)),
            heading_mode=str(values.get("heading_mode", defaults.heading_mode)),
            consistency_mode=str(
                values.get("consistency_mode", defaults.consistency_mode)
            ),
            trace_enabled=bool(values.get("trace_enabled", defaults.trace_enabled)),
            eps=float(values.get("eps", defaults.eps)),
        )


class M1TraceRecorder:
    """Collect detached CPU snapshots for one optional sampler trajectory.

    The recorder is intentionally passive.  It never participates in the
    guidance computation and does no work when ``enabled`` is false.  Values
    are copied at the boundary so a later in-place scheduler update cannot
    mutate the diagnostic record.
    """

    REQUIRED_FIELDS = (
        "sigma", "timestep", "x_sigma", "v_cfg", "x0_hat", "x0_guided",
        "x0_reconciled", "v_corrected", "x_next", "next_model_x0",
    )

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self.records: list[dict[str, Any]] = []

    @staticmethod
    def _snapshot(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu").clone()
        return value

    def record_step(self, **payload: Any) -> None:
        if not self.enabled:
            return
        missing = [name for name in self.REQUIRED_FIELDS if name not in payload]
        if missing:
            raise ValueError(f"trace step missing fields: {missing}")
        self.records.append({
            name: self._snapshot(value) for name, value in payload.items()
        })


def _travel_heading(velocity: torch.Tensor, min_speed: float) -> tuple[torch.Tensor, torch.Tensor]:
    if velocity.ndim < 2 or velocity.shape[-1] != 3:
        raise ValueError("root translation velocity must have shape [...,T,3]")
    horizontal = velocity.clone()
    horizontal[..., 2] = 0
    speed = torch.linalg.vector_norm(horizontal, dim=-1)
    valid = speed >= min_speed
    heading = horizontal / speed.unsqueeze(-1).clamp_min(torch.finfo(velocity.dtype).eps)
    canonical = torch.zeros(3, dtype=velocity.dtype, device=velocity.device)
    canonical[1] = 1.0
    heading = torch.where(valid.unsqueeze(-1), heading, canonical)
    return heading, valid


class M1Guidance:
    """Full-space endpoint guidance toward a relative candidate-angle target."""

    def __init__(
        self,
        *,
        baseline_motion_norm: torch.Tensor,
        valid_mask: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        target_delta_deg: float,
        config: M1Config | None = None,
    ) -> None:
        if baseline_motion_norm.ndim != 3 or baseline_motion_norm.shape[-1] != 276:
            raise ValueError("baseline_motion_norm must have shape [B,T,276]")
        if valid_mask.shape != baseline_motion_norm.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool[B,T]")
        if mean.shape[-1] != 276 or std.shape[-1] != 276:
            raise ValueError("mean and std must have 276 channels")
        if torch.any(std <= 0) or not torch.isfinite(std).all():
            raise ValueError("std must be finite and positive")
        self.baseline_motion_norm = baseline_motion_norm.detach()
        self.valid_mask = valid_mask.detach()
        self.mean = mean.detach()
        self.std = std.detach()
        self.target_delta_deg = float(target_delta_deg)
        self.config = config or M1Config()
        self.trace_recorder = M1TraceRecorder(enabled=self.config.trace_enabled)
        if self.config.consistency_mode not in (
            "legacy",
            "velocity_authoritative_v2",
        ):
            raise ValueError(
                "consistency_mode must be 'legacy' or "
                "'velocity_authoritative_v2'"
            )
        self.last_diagnostics: dict[str, Any] = {
            "enabled": bool(self.config.enabled),
            "active": False,
            "consistency_mode": self.config.consistency_mode,
            "endpoint_reconciled": False,
        }

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def slice(self, index: int) -> "M1Guidance":
        """Select one sample for the sampler's batch-invariant mode."""

        if self.baseline_motion_norm.shape[0] == 1:
            return self
        mean = self.mean
        std = self.std
        # Batch-invariant sampling selects one motion while the statistics
        # may still be stored per sample as [B,276].  Keep those axes aligned;
        # global [276] statistics are shared and remain unchanged.
        if mean.ndim == 2 and mean.shape[0] == self.baseline_motion_norm.shape[0]:
            mean = mean[index:index + 1]
            std = std[index:index + 1]
        return M1Guidance(
            baseline_motion_norm=self.baseline_motion_norm[index:index + 1],
            valid_mask=self.valid_mask[index:index + 1],
            mean=mean,
            std=std,
            target_delta_deg=self.target_delta_deg,
            config=self.config,
        )

    @property
    def trace_enabled(self) -> bool:
        return bool(self.trace_recorder.enabled)

    def pop_trace_records(self) -> list[dict[str, Any]]:
        """Return and clear records accumulated by this guidance instance."""

        records = self.trace_recorder.records
        self.trace_recorder.records = []
        return records

    def _physical(self, motion_norm: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=motion_norm.device, dtype=motion_norm.dtype)
        std = self.std.to(device=motion_norm.device, dtype=motion_norm.dtype)
        # Per-sample statistics arrive as [B,276] while motion is [B,T,276].
        # Insert the frame axis explicitly; a [276] vector already broadcasts
        # correctly.  This keeps standardization boundaries unambiguous.
        if mean.ndim == motion_norm.ndim - 1:
            mean = mean.unsqueeze(-2)
            std = std.unsqueeze(-2)
        return motion_norm * std + mean

    def _candidate_angle(self, motion_phys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        root = decode_rot6d_safe(motion_phys[..., 258:264])
        if self.config.heading_mode == "travel":
            heading, speed_mask = _travel_heading(
                motion_phys[..., 273:276], self.config.min_speed
            )
        elif self.config.heading_mode == "canonical_y":
            heading = torch.zeros(
                (*motion_phys.shape[:-1], 3),
                dtype=motion_phys.dtype,
                device=motion_phys.device,
            )
            heading[..., 1] = 1.0
            speed_mask = torch.ones(
                motion_phys.shape[:-1], dtype=torch.bool, device=motion_phys.device
            )
        else:
            raise ValueError(
                "heading_mode must be 'travel' or 'canonical_y', got "
                f"{self.config.heading_mode!r}"
            )
        angle = pelvis_pitch_degrees(root, heading, local_forward_axis=2, up_axis=2)
        return angle, speed_mask

    def correct_velocity(
        self,
        *,
        x_sigma: torch.Tensor,
        velocity: torch.Tensor,
        sigma: torch.Tensor | float,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply one M1 endpoint correction and recompose the velocity."""

        if x_sigma.shape != velocity.shape or x_sigma.ndim != 3 or x_sigma.shape[-1] != 276:
            raise ValueError("x_sigma and velocity must have shape [B,T,276]")
        if valid_mask.shape != x_sigma.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool[B,T]")
        sigma_value = float(torch.as_tensor(sigma).detach().cpu().item())
        cfg = self.config
        diagnostics: dict[str, Any] = {
            "enabled": bool(cfg.enabled),
            "active": False,
            "sigma": sigma_value,
            "target_delta_deg": self.target_delta_deg,
            "consistency_mode": cfg.consistency_mode,
            "endpoint_reconciled": False,
        }
        if (
            not cfg.enabled
            or abs(self.target_delta_deg) <= cfg.eps
            or sigma_value <= cfg.eps
            or sigma_value < cfg.sigma_min
            or sigma_value > cfg.sigma_max
        ):
            self.last_diagnostics = diagnostics
            return velocity, diagnostics

        # The model forward is already no-grad.  Re-enable autograd only for
        # the detached endpoint and the small physical-space loss graph.
        with torch.enable_grad(), torch.amp.autocast(
            device_type=x_sigma.device.type, enabled=False
        ):
            x0_norm = predict_x0(x_sigma.float(), velocity.float(), sigma_value).detach()
            x0_norm.requires_grad_(True)
            baseline_norm = self.baseline_motion_norm.to(device=x0_norm.device, dtype=x0_norm.dtype)
            baseline_phys = self._physical(baseline_norm)
            x0_phys = self._physical(x0_norm)

            angle, speed_mask = self._candidate_angle(x0_phys)
            with torch.no_grad():
                baseline_angle, baseline_speed_mask = self._candidate_angle(baseline_phys)
            mask = valid_mask & speed_mask & baseline_speed_mask
            mask_f = mask.to(dtype=x0_phys.dtype).unsqueeze(-1)
            count = mask_f.sum().clamp_min(1.0)
            angle_residual = angle - (baseline_angle + self.target_delta_deg)
            angle_loss = (angle_residual.square().unsqueeze(-1) * mask_f).sum() / count
            hold_loss = ((x0_phys - baseline_phys).square() * mask_f).sum() / (
                count * x0_phys.shape[-1]
            )
            loss = cfg.angle_weight * angle_loss + cfg.hold_weight * hold_loss
            gradient = torch.autograd.grad(loss, x0_norm, allow_unused=False)[0]
            gradient = gradient * mask_f
            rms = torch.sqrt((gradient.square() * mask_f).sum() / (count * x0_norm.shape[-1])).clamp_min(cfg.eps)
            correction = cfg.lambda_scale * gradient / rms
            correction_rms = torch.sqrt(
                (correction.square() * mask_f).sum() / (count * x0_norm.shape[-1])
            )
            scale = torch.clamp(
                torch.as_tensor(cfg.max_correction_rms, dtype=correction.dtype, device=correction.device)
                / correction_rms.clamp_min(cfg.eps),
                max=1.0,
            )
            correction = correction * scale
            x0_guided = x0_norm - correction
            x0_guided_pre_reconcile = x0_guided.detach()
            if cfg.consistency_mode == "velocity_authoritative_v2":
                # Geometry is an endpoint projection, not part of the loss
                # graph.  Run it after autograd in FP32 and discard the short
                # graph before returning to the no-grad flow sampler.
                with torch.no_grad():
                    reconciled = m1_consistency.reconcile_guided_endpoint(
                        x0_norm.detach(),
                        x0_guided.detach(),
                        valid_mask=valid_mask,
                        mean=self.mean,
                        std=self.std,
                        output_dtype=torch.float32,
                    )
                    x0_guided = reconciled.motion
                diagnostics["endpoint_reconciled"] = True
            corrected_velocity = velocity_from_x0(
                x_sigma.float(), x0_guided, sigma_value
            ).to(dtype=velocity.dtype)
            if cfg.consistency_mode == "velocity_authoritative_v2":
                # Invalid latent rows are outside the motion objective and
                # must remain byte-for-byte equal to the model velocity.
                corrected_velocity = torch.where(
                    valid_mask.unsqueeze(-1), corrected_velocity, velocity
                )
            diagnostics.update(
                {
                    "active": True,
                    "angle_loss": float(angle_loss.detach().cpu().item()),
                    "hold_loss": float(hold_loss.detach().cpu().item()),
                    "gradient_rms": float(rms.detach().cpu().item()),
                    "correction_rms": float(torch.sqrt((correction.square() * mask_f).sum() / (count * x0_norm.shape[-1])).detach().cpu().item()),
                    "valid_frames": int(mask.sum().detach().cpu().item()),
                }
            )
            if self.trace_enabled:
                # Keep tensors out of the regular diagnostics dictionary.  It
                # is consumed only by FlowSampler when explicit tracing is on.
                diagnostics["_trace_payload"] = {
                    "x_sigma": x_sigma.detach(),
                    "v_cfg": velocity.detach(),
                    "x0_hat": x0_norm.detach(),
                    "x0_guided": x0_guided_pre_reconcile,
                    "x0_reconciled": x0_guided.detach(),
                    "v_corrected": corrected_velocity.detach(),
                }
        self.last_diagnostics = diagnostics
        return corrected_velocity, diagnostics


