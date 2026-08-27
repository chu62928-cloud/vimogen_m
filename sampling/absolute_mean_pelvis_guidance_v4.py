"""Anatomical-local pelvis guidance with soft anti-cheat constraints (v4).

This module is a new protocol.  v3 remains importable and unchanged.  The
guidance state is standardised, while every geometric loss is evaluated after
the v3 authoritative pose -> FK -> 276-D repacking boundary.  Only pelvis,
bilateral hips, and spine1/2/3 Rot6D channels receive guidance gradients.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from motion_rep.anatomical_pelvis import (
    PelvisCalibration,
    anatomical_pelvis_geometry,
    anti_cheat_penalty,
    load_pelvis_calibration,
    trunk_and_thigh_angles,
)
from motion_rep.consistency_v2 import load_smplx_neutral_22_skeleton
from motion_rep.consistency_v3 import reconcile_motion_tensor_v3
from motion_rep.phase1 import ACTIVE_ROTATION_SLICES, MOTION_LAYOUT, decode_rot6d_safe
from sampling.m1_guidance import predict_x0, velocity_from_x0
from sampling.absolute_mean_pelvis_guidance_v2 import (
    _align_statistic,
    _masked_mean,
)


PROTOCOL_NAME = "vimogen_absolute_mean_pelvis_v4_anatomical_local"


@dataclass(frozen=True)
class AbsoluteMeanPelvisConfigV4:
    enabled: bool = True
    guidance_strength: float = 1.0
    sigma_min: float = 0.25
    sigma_max: float = 0.65
    mean_weight: float = 1.0
    shape_weight: float = 0.1
    motion_weight: float = 0.1
    anti_cheat_weight: float = 1.0
    max_correction_rms: float = 0.05
    fusion_window: int = 9
    anchor_weight: float = 1.0
    terminal_enabled: bool = True
    terminal_max_deg: float = 1.0
    soft_limit_deg: float = 2.0
    p95_limit_deg: float = 3.0
    low_signal_deg: float = 0.5
    ratio_floor_deg: float = 0.25
    eps: float = 1e-6
    skeleton_path: str | None = None
    calibration_path: str | None = None

    def __post_init__(self) -> None:
        if self.guidance_strength < 0 or self.mean_weight < 0 or self.shape_weight < 0 or self.motion_weight < 0:
            raise ValueError("guidance strength and loss weights must be non-negative")
        if not 0 <= self.sigma_min <= self.sigma_max <= 1:
            raise ValueError("sigma window must satisfy 0 <= min <= max <= 1")
        if self.anti_cheat_weight != 1.0:
            raise ValueError("anti_cheat_weight is fixed at 1.0 by the v4 protocol")
        if self.max_correction_rms <= 0:
            raise ValueError("max_correction_rms must be positive")
        if self.fusion_window < 1 or self.fusion_window % 2 == 0:
            raise ValueError("fusion_window must be a positive odd integer")
        if not 0 <= self.anchor_weight <= 1:
            raise ValueError("anchor_weight must lie in [0,1]")
        if not 0 <= self.terminal_max_deg <= 1:
            raise ValueError("terminal_max_deg must lie in [0,1]")
        if self.soft_limit_deg < 0 or self.p95_limit_deg <= self.soft_limit_deg:
            raise ValueError("p95_limit_deg must be greater than soft_limit_deg")
        if self.low_signal_deg <= 0 or self.ratio_floor_deg <= 0:
            raise ValueError("low-signal and ratio floors must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "AbsoluteMeanPelvisConfigV4":
        values = values or {}
        defaults = cls()
        kwargs = asdict(defaults)
        for key in kwargs:
            if key in values:
                kwargs[key] = values[key]
        if kwargs["skeleton_path"] is not None:
            kwargs["skeleton_path"] = str(kwargs["skeleton_path"])
        if kwargs["calibration_path"] is not None:
            kwargs["calibration_path"] = str(kwargs["calibration_path"])
        return cls(**kwargs)


@dataclass(frozen=True)
class AbsoluteMeanFinalOutputsV4:
    g0: torch.Tensor
    g1: torch.Tensor
    g0_valid_mask: torch.Tensor
    g1_valid_mask: torch.Tensor
    terminal_records: tuple[dict[str, Any], ...]
    protocol: str = PROTOCOL_NAME


def _active_rotation_mask(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros(MOTION_LAYOUT.total_dim, dtype=dtype, device=device)
    for span in ACTIVE_ROTATION_SLICES.values():
        mask[span] = 1.0
    return mask


def anatomical_angle_curves(
    motion_phys: torch.Tensor,
    calibration: PelvisCalibration,
) -> dict[str, torch.Tensor]:
    """Compute all shared geometry curves from a physical packed motion."""

    root = decode_rot6d_safe(motion_phys[..., MOTION_LAYOUT.root_rotation])
    pelvis = anatomical_pelvis_geometry(root, calibration)
    joints = motion_phys[..., MOTION_LAYOUT.joints].reshape(*motion_phys.shape[:-1], 22, 3)
    segments = trunk_and_thigh_angles(joints, pelvis)
    return {
        "pelvis_deg": pelvis.angle_degrees,
        "trunk_deg": segments["trunk_deg"],
        "thigh_left_deg": segments["thigh_left_deg"],
        "thigh_right_deg": segments["thigh_right_deg"],
        "pelvis_valid": pelvis.valid,
        "trunk_valid": segments["trunk_valid"],
        "thigh_left_valid": segments["thigh_left_valid"],
        "thigh_right_valid": segments["thigh_right_valid"],
    }


class AbsoluteMeanPelvisGuidanceV4:
    """Full-FK guidance whose pelvis change is locally dominant by design."""

    def __init__(
        self,
        *,
        baseline_motion_norm: torch.Tensor,
        valid_mask: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        target_mean_deg: float,
        config: AbsoluteMeanPelvisConfigV4 | None = None,
        calibration: PelvisCalibration | None = None,
    ) -> None:
        if baseline_motion_norm.ndim != 3 or baseline_motion_norm.shape[-1] != 276:
            raise ValueError("baseline_motion_norm must have shape [B,T,276]")
        if valid_mask.shape != baseline_motion_norm.shape[:2] or valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must be bool[B,T]")
        if mean.shape[-1] != 276 or std.shape[-1] != 276:
            raise ValueError("mean and std must have 276 channels")
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("std must be finite and positive")
        self.baseline_motion_norm = baseline_motion_norm.detach()
        self.valid_mask = valid_mask.detach()
        self.mean = mean.detach()
        self.std = std.detach()
        self.target_mean_deg = float(target_mean_deg)
        self.config = config or AbsoluteMeanPelvisConfigV4()
        self.calibration = calibration
        if self.calibration is None and self.config.calibration_path is not None:
            self.calibration = load_pelvis_calibration(self.config.calibration_path)
        if self.calibration is None:
            raise ValueError("v4 requires a frozen PelvisCalibration or calibration_path")
        self.last_diagnostics: dict[str, Any] = {"protocol": PROTOCOL_NAME, "enabled": self.config.enabled, "active": False}

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def slice(self, index: int) -> "AbsoluteMeanPelvisGuidanceV4":
        if self.baseline_motion_norm.shape[0] == 1:
            return self
        mean, std = self.mean, self.std
        if mean.ndim == 2 and mean.shape[0] == self.baseline_motion_norm.shape[0]:
            mean, std = mean[index:index + 1], std[index:index + 1]
        return type(self)(
            baseline_motion_norm=self.baseline_motion_norm[index:index + 1],
            valid_mask=self.valid_mask[index:index + 1],
            mean=mean,
            std=std,
            target_mean_deg=self.target_mean_deg,
            config=self.config,
            calibration=self.calibration,
        )

    def protocol_record(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL_NAME,
            "target_semantics": "absolute_valid_frame_mean_anatomical_pelvis_tilt_deg",
            "angle_definition": "atan2(-dot(A-P,up), dot(A-P,heading))",
            "positive_direction": "anterior_side_down",
            "active_rotation_channels": {name: [span.start, span.stop] for name, span in ACTIVE_ROTATION_SLICES.items()},
            "anti_cheat": {"soft_limit_deg": self.config.soft_limit_deg, "p95_limit_deg": self.config.p95_limit_deg, "coefficient": 1.0},
            "calibration": self.calibration.to_mapping(),
            "config": asdict(self.config),
        }

    def _reconcile(self, motion_norm: torch.Tensor, *, output_standardized: bool):
        skeleton = None if self.config.skeleton_path is None else load_smplx_neutral_22_skeleton(self.config.skeleton_path)
        return reconcile_motion_tensor_v3(
            motion_norm,
            fusion_window=self.config.fusion_window,
            anchor_weight=self.config.anchor_weight,
            root_rotation_anchor_weight=self.config.anchor_weight,
            valid_mask=self.valid_mask.to(motion_norm.device),
            mean=self.mean,
            std=self.std,
            input_standardized=True,
            output_standardized=output_standardized,
            output_dtype=torch.float32,
            skeleton=skeleton,
        )

    def _standardize(self, motion_phys: torch.Tensor) -> torch.Tensor:
        return (motion_phys - _align_statistic(self.mean, motion_phys, "mean")) / _align_statistic(self.std, motion_phys, "std")

    def _objective(
        self,
        candidate: torch.Tensor,
        baseline: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.config
        candidate_angles = anatomical_angle_curves(candidate, self.calibration)
        baseline_angles = anatomical_angle_curves(baseline, self.calibration)
        count = mask.sum().clamp_min(1).to(candidate.dtype)
        pelvis = candidate_angles["pelvis_deg"]
        pelvis0 = baseline_angles["pelvis_deg"]
        mean_angle = _masked_mean(pelvis, mask)
        baseline_mean = _masked_mean(pelvis0, mask)
        mean_loss = (mean_angle - self.target_mean_deg).square().mean()
        centered = pelvis - mean_angle.unsqueeze(-1)
        centered0 = pelvis0 - baseline_mean.unsqueeze(-1)
        shape_loss = ((centered - centered0).square() * mask.to(candidate.dtype)).sum() / count
        candidate_norm = self._standardize(candidate)
        baseline_norm = self._standardize(baseline)
        motion_mask = mask.to(candidate.dtype).unsqueeze(-1)
        motion_loss = ((candidate_norm - baseline_norm).square() * motion_mask).sum() / (count * candidate.shape[-1])
        dtrunk = candidate_angles["trunk_deg"] - baseline_angles["trunk_deg"]
        dleft = candidate_angles["thigh_left_deg"] - baseline_angles["thigh_left_deg"]
        dright = candidate_angles["thigh_right_deg"] - baseline_angles["thigh_right_deg"]
        anti_loss = anti_cheat_penalty(dtrunk, dleft, dright, mask, soft_limit_deg=cfg.soft_limit_deg)
        total = cfg.mean_weight * mean_loss + cfg.shape_weight * shape_loss + cfg.motion_weight * motion_loss + anti_loss
        return total, {
            "mean_angle": mean_angle,
            "baseline_mean": baseline_mean,
            "mean_loss": mean_loss,
            "shape_loss": shape_loss,
            "motion_loss": motion_loss,
            "anti_cheat_loss": anti_loss,
            "delta_trunk": dtrunk,
            "delta_thigh_left": dleft,
            "delta_thigh_right": dright,
        }

    def correct_velocity(
        self,
        *,
        x_sigma: torch.Tensor,
        velocity: torch.Tensor,
        sigma: torch.Tensor | float,
        valid_mask: torch.Tensor,
        return_trace: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if x_sigma.shape != velocity.shape or x_sigma.ndim != 3 or x_sigma.shape[-1] != 276:
            raise ValueError("x_sigma and velocity must have shape [B,T,276]")
        if valid_mask.shape != x_sigma.shape[:2] or valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must be bool[B,T]")
        if not torch.equal(valid_mask, self.valid_mask.to(valid_mask.device)):
            raise ValueError("sampler valid_mask differs from the frozen guidance mask")
        sigma_value = float(torch.as_tensor(sigma).detach().cpu().item())
        cfg = self.config
        diagnostics: dict[str, Any] = {"protocol": PROTOCOL_NAME, "enabled": cfg.enabled, "active": False, "sigma": sigma_value, "target_mean_deg": self.target_mean_deg}
        inactive = not cfg.enabled or cfg.guidance_strength <= cfg.eps or sigma_value <= cfg.eps or sigma_value < cfg.sigma_min or sigma_value > cfg.sigma_max
        if inactive:
            if return_trace:
                x0_hat = predict_x0(x_sigma.float(), velocity.float(), sigma_value)
                diagnostics["trace"] = {"velocity_model": velocity.detach().float().clone(), "v_cfg": velocity.detach().float().clone(), "x0_hat": x0_hat.detach().clone(), "x0_guided": x0_hat.detach().clone(), "x0_reconciled": x0_hat.detach().clone()}
            self.last_diagnostics = diagnostics
            return velocity, diagnostics
        with torch.enable_grad(), torch.amp.autocast(device_type=x_sigma.device.type, enabled=False):
            x0_norm = predict_x0(x_sigma.float(), velocity.float(), sigma_value).detach().requires_grad_(True)
            candidate = self._reconcile(x0_norm, output_standardized=False)
            with torch.no_grad():
                baseline = self._reconcile(self.baseline_motion_norm.to(x0_norm.device, torch.float32), output_standardized=False)
            mask = valid_mask & candidate.valid_mask & baseline.valid_mask
            before_loss, before = self._objective(candidate.motion, baseline.motion, mask)
            gradient = torch.autograd.grad(before_loss, x0_norm, allow_unused=False)[0]
            channel_mask = _active_rotation_mask(x0_norm.device, x0_norm.dtype)
            mask_f = mask.to(x0_norm.dtype).unsqueeze(-1)
            gradient = torch.nan_to_num(gradient) * mask_f * channel_mask
            correction = cfg.guidance_strength * gradient
            count = mask.sum().clamp_min(1).to(x0_norm.dtype)
            correction_rms_before_cap = torch.sqrt((correction.square() * mask_f).sum() / (count * x0_norm.shape[-1]))
            cap_scale = torch.clamp(torch.as_tensor(cfg.max_correction_rms, dtype=correction.dtype, device=correction.device) / correction_rms_before_cap.clamp_min(cfg.eps), max=1.0)
            correction = correction * cap_scale
            accepted_scale = 1.0
            post = None
            after = before
            with torch.no_grad():
                for _ in range(11):
                    trial = x0_norm.detach() - accepted_scale * correction.detach()
                    trial_post = self._reconcile(trial, output_standardized=False)
                    trial_mask = mask & trial_post.valid_mask
                    trial_loss, trial_values = self._objective(trial_post.motion, baseline.motion, trial_mask)
                    if trial_loss <= before_loss + cfg.eps:
                        post = self._reconcile(trial, output_standardized=True)
                        after = trial_values
                        break
                    accepted_scale *= 0.5
                if post is None:
                    accepted_scale = 0.0
                    post = self._reconcile(x0_norm.detach(), output_standardized=True)
                x0_reconciled = post.motion
            corrected_velocity = velocity_from_x0(x_sigma.float(), x0_reconciled, sigma_value).to(velocity.dtype)
            corrected_velocity = torch.where(valid_mask.unsqueeze(-1), corrected_velocity, velocity)
            correction = correction * accepted_scale
            correction_rms = torch.sqrt((correction.square() * mask_f).sum() / (count * x0_norm.shape[-1]))
            diagnostics.update({
                "active": True,
                "mean_angle_deg": [float(v) for v in after["mean_angle"].detach().cpu()],
                "baseline_mean_deg": [float(v) for v in after["baseline_mean"].detach().cpu()],
                "mean_loss": float(after["mean_loss"].detach().cpu()),
                "shape_loss": float(after["shape_loss"].detach().cpu()),
                "motion_loss": float(after["motion_loss"].detach().cpu()),
                "anti_cheat_loss": float(after["anti_cheat_loss"].detach().cpu()),
                "gradient_rms": float(torch.sqrt((gradient.square() * mask_f).sum() / (count * x0_norm.shape[-1])).detach().cpu()),
                "correction_rms_before_cap": float(correction_rms_before_cap.detach().cpu()),
                "correction_rms": float(correction_rms.detach().cpu()),
                "backtracking_scale": float(accepted_scale),
                "active_rotation_channels": [name for name in ACTIVE_ROTATION_SLICES],
                "valid_frames": int(mask.sum().detach().cpu()),
            })
            if return_trace:
                diagnostics["trace"] = {"velocity_model": velocity.detach().float().clone(), "v_cfg": velocity.detach().float().clone(), "x0_hat": x0_norm.detach().float().clone(), "x0_guided": (x0_norm.detach() - correction).float().clone(), "x0_reconciled": x0_reconciled.detach().float().clone()}
        self.last_diagnostics = diagnostics
        return corrected_velocity, diagnostics

    @torch.no_grad()
    def finalize_outputs(self, official_norm: torch.Tensor) -> AbsoluteMeanFinalOutputsV4:
        """Return G0 and a conservative G1; no unconstrained root-only edit is used."""

        g0_result = self._reconcile(official_norm.float(), output_standardized=True)
        records = tuple({
            "target_mean_deg": self.target_mean_deg,
            "triggered": False,
            "applied_deg": 0.0,
            "mean_after_deg": None,
            "terminal_skipped": True,
            "reason": "v4 terminal refinement must use the same constrained optimiser; no rigid root edit applied",
        } for _ in range(g0_result.motion.shape[0]))
        return AbsoluteMeanFinalOutputsV4(
            g0=g0_result.motion,
            g1=g0_result.motion.clone(),
            g0_valid_mask=g0_result.valid_mask,
            g1_valid_mask=g0_result.valid_mask.clone(),
            terminal_records=records,
        )


__all__ = [
    "AbsoluteMeanFinalOutputsV4", "AbsoluteMeanPelvisConfigV4", "AbsoluteMeanPelvisGuidanceV4",
    "PROTOCOL_NAME", "anatomical_angle_curves",
]
