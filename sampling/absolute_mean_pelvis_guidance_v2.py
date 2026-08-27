"""Full-FK local-sagittal absolute pelvis guidance (protocol v2).

This module is deliberately separate from the frozen v1 implementation.  At
every active diffusion step, each candidate endpoint is passed through the
v2 root fusion -> differentiable FK -> complete 276-D repacking boundary.
G1 applies its terminal correction around the per-frame person's right axis,
so turning/yaw does not change the meaning of the signed tilt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch

from motion_rep.finalize import finalize_motion
from motion_rep.sagittal_pelvis_angle import (
    apply_person_right_axis_rotation,
    pelvis_sagittal_tilt_degrees,
)
from motion_rep.consistency_v2 import (
    differentiable_forward_kinematics,
    load_smplx_neutral_22_skeleton,
    reconcile_motion_tensor_v2,
)
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe
from motion_rep.unified_finalizer import recover_motion_stream
from sampling.m1_guidance import predict_x0, velocity_from_x0


PROTOCOL_NAME = "vimogen_absolute_mean_pelvis_v2_full_fk_sagittal"


@dataclass(frozen=True)
class AbsoluteMeanPelvisConfigV2:
    enabled: bool = True
    guidance_strength: float = 1.0
    sigma_min: float = 0.25
    sigma_max: float = 0.65
    mean_weight: float = 1.0
    shape_weight: float = 0.1
    motion_weight: float = 0.1
    max_correction_rms: float = 0.05
    fusion_window: int = 9
    anchor_weight: float = 1.0
    terminal_enabled: bool = True
    terminal_max_deg: float = 1.0
    eps: float = 1e-6
    skeleton_path: str | None = None

    def __post_init__(self) -> None:
        if self.guidance_strength < 0:
            raise ValueError("guidance_strength must be non-negative")
        if not 0 <= self.sigma_min <= self.sigma_max <= 1:
            raise ValueError("sigma window must satisfy 0 <= min <= max <= 1")
        if min(self.mean_weight, self.shape_weight, self.motion_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        if self.max_correction_rms <= 0:
            raise ValueError("max_correction_rms must be positive")
        if self.fusion_window < 1 or self.fusion_window % 2 == 0:
            raise ValueError("fusion_window must be a positive odd integer")
        if not 0 <= self.anchor_weight <= 1:
            raise ValueError("anchor_weight must lie in [0,1]")
        if not 0 <= self.terminal_max_deg <= 1.0:
            raise ValueError("terminal_max_deg must lie in [0,1]")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None
    ) -> "AbsoluteMeanPelvisConfigV2":
        values = values or {}
        defaults = cls()
        return cls(
            enabled=bool(values.get("enabled", False)),
            guidance_strength=float(
                values.get("guidance_strength", defaults.guidance_strength)
            ),
            sigma_min=float(values.get("sigma_min", defaults.sigma_min)),
            sigma_max=float(values.get("sigma_max", defaults.sigma_max)),
            mean_weight=float(values.get("mean_weight", defaults.mean_weight)),
            shape_weight=float(values.get("shape_weight", defaults.shape_weight)),
            motion_weight=float(values.get("motion_weight", defaults.motion_weight)),
            max_correction_rms=float(
                values.get("max_correction_rms", defaults.max_correction_rms)
            ),
            fusion_window=int(values.get("fusion_window", defaults.fusion_window)),
            anchor_weight=float(values.get("anchor_weight", defaults.anchor_weight)),
            terminal_enabled=bool(
                values.get("terminal_enabled", defaults.terminal_enabled)
            ),
            terminal_max_deg=float(
                values.get("terminal_max_deg", defaults.terminal_max_deg)
            ),
            eps=float(values.get("eps", defaults.eps)),
            skeleton_path=(None if values.get("skeleton_path", defaults.skeleton_path) is None else str(values.get("skeleton_path", defaults.skeleton_path))),
        )


@dataclass(frozen=True)
class AbsoluteMeanFinalOutputsV2:
    g0: torch.Tensor
    g1: torch.Tensor
    g0_valid_mask: torch.Tensor
    g1_valid_mask: torch.Tensor
    terminal_records: tuple[dict[str, Any], ...]
    protocol: str = PROTOCOL_NAME


def _align_statistic(value: torch.Tensor, motion: torch.Tensor, name: str) -> torch.Tensor:
    value = value.to(device=motion.device, dtype=motion.dtype)
    if value.ndim == 1 and value.shape == (MOTION_LAYOUT.total_dim,):
        return value
    if (
        value.ndim == 2
        and motion.ndim == 3
        and value.shape == (motion.shape[0], MOTION_LAYOUT.total_dim)
    ):
        return value.unsqueeze(1)
    if value.ndim == 2 and motion.ndim == 2 and value.shape == (1, MOTION_LAYOUT.total_dim):
        return value[0]
    raise ValueError(f"{name} must have shape [276] or one [B,276] row per sample")


def _canonical_heading(motion: torch.Tensor) -> torch.Tensor:
    heading = torch.zeros(
        (*motion.shape[:-1], 3), dtype=motion.dtype, device=motion.device
    )
    heading[..., 1] = 1.0
    return heading


def pelvis_angle_curve(motion_phys: torch.Tensor) -> torch.Tensor:
    """Return the v2 yaw-removed local-sagittal tilt curve in degrees."""

    root = decode_rot6d_safe(motion_phys[..., MOTION_LAYOUT.root_rotation])
    return pelvis_sagittal_tilt_degrees(root)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum(dim=-1).clamp_min(1).to(values.dtype)
    return (values * mask.to(values.dtype)).sum(dim=-1) / count


def _prefix_pose_mask(row_mask: torch.Tensor) -> torch.Tensor:
    """Expand a valid output-row prefix to the corresponding T+1 pose mask.

    ``finalize_motion`` consumes a pose mask and marks output row ``t`` valid
    only when both pose ``t`` and pose ``t+1`` are valid.  For a variable
    length motion with ``L`` valid output rows, the authoritative pose stream
    therefore has exactly ``L+1`` valid poses.  The old ``cat(mask[-1:])``
    convention is intentionally not used: it puts the hidden pose after the
    padded tail and can discard the final valid row.
    """

    if row_mask.ndim != 1 or row_mask.dtype is not torch.bool:
        raise ValueError("row_mask must be a one-dimensional bool tensor")
    if torch.any(row_mask[1:] & ~row_mask[:-1]):
        raise ValueError("row_mask must be a contiguous valid prefix")
    pose_mask = torch.zeros(
        row_mask.numel() + 1, dtype=torch.bool, device=row_mask.device
    )
    pose_mask[: int(row_mask.sum().item()) + 1] = True
    return pose_mask


class AbsoluteMeanPelvisGuidanceV2:
    """Guide absolute valid-frame mean after full-FK v2 consistency."""

    def __init__(
        self,
        *,
        baseline_motion_norm: torch.Tensor,
        valid_mask: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        target_mean_deg: float,
        config: AbsoluteMeanPelvisConfigV2 | None = None,
    ) -> None:
        if baseline_motion_norm.ndim != 3 or baseline_motion_norm.shape[-1] != 276:
            raise ValueError("baseline_motion_norm must have shape [B,T,276]")
        if valid_mask.shape != baseline_motion_norm.shape[:2] or valid_mask.dtype != torch.bool:
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
        self.config = config or AbsoluteMeanPelvisConfigV2()
        self.last_diagnostics: dict[str, Any] = {
            "protocol": PROTOCOL_NAME,
            "enabled": self.config.enabled,
            "active": False,
        }

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def slice(self, index: int) -> "AbsoluteMeanPelvisGuidanceV2":
        if self.baseline_motion_norm.shape[0] == 1:
            return self
        mean, std = self.mean, self.std
        if mean.ndim == 2 and mean.shape[0] == self.baseline_motion_norm.shape[0]:
            mean, std = mean[index:index + 1], std[index:index + 1]
        return AbsoluteMeanPelvisGuidanceV2(
            baseline_motion_norm=self.baseline_motion_norm[index:index + 1],
            valid_mask=self.valid_mask[index:index + 1],
            mean=mean,
            std=std,
            target_mean_deg=self.target_mean_deg,
            config=self.config,
        )

    def protocol_record(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL_NAME,
            "target_semantics": "absolute_valid_frame_mean_model_space_sagittal_tilt_deg",
            "target_mean_deg": self.target_mean_deg,
            "angle_authority": "root_rotation_yaw_removed_local_sagittal_plane",
            "post_update_boundary": "fused_root_then_differentiable_fk22_then_repack_all_276_channels",
            "config": asdict(self.config),
        }

    def _reconcile(
        self, motion_norm: torch.Tensor, *, output_standardized: bool
    ):
        skeleton = None if self.config.skeleton_path is None else load_smplx_neutral_22_skeleton(self.config.skeleton_path)
        return reconcile_motion_tensor_v2(
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
        mean = _align_statistic(self.mean, motion_phys, "mean")
        std = _align_statistic(self.std, motion_phys, "std")
        return (motion_phys - mean) / std

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
        if valid_mask.shape != x_sigma.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool[B,T]")
        if not torch.equal(valid_mask, self.valid_mask.to(valid_mask.device)):
            raise ValueError("sampler valid_mask differs from the frozen guidance mask")
        sigma_value = float(torch.as_tensor(sigma).detach().cpu().item())
        cfg = self.config
        diagnostics: dict[str, Any] = {
            "protocol": PROTOCOL_NAME,
            "enabled": cfg.enabled,
            "active": False,
            "sigma": sigma_value,
            "target_mean_deg": self.target_mean_deg,
        }
        inactive = (
            not cfg.enabled
            or cfg.guidance_strength <= cfg.eps
            or sigma_value <= cfg.eps
            or sigma_value < cfg.sigma_min
            or sigma_value > cfg.sigma_max
        )
        if inactive:
            if return_trace:
                x0_hat = predict_x0(x_sigma.float(), velocity.float(), sigma_value)
                diagnostics["trace"] = {
                    "velocity_model": velocity.detach().float().clone(),
                    "v_cfg": velocity.detach().float().clone(),
                    "x0_hat": x0_hat.detach().clone(),
                    "x0_guided": x0_hat.detach().clone(),
                    "x0_reconciled": x0_hat.detach().clone(),
                }
            self.last_diagnostics = diagnostics
            return velocity, diagnostics

        with torch.enable_grad(), torch.amp.autocast(
            device_type=x_sigma.device.type, enabled=False
        ):
            x0_norm = predict_x0(x_sigma.float(), velocity.float(), sigma_value).detach()
            x0_norm.requires_grad_(True)
            candidate = self._reconcile(x0_norm, output_standardized=False)
            with torch.no_grad():
                baseline = self._reconcile(
                    self.baseline_motion_norm.to(x0_norm.device, torch.float32),
                    output_standardized=False,
                )
            mask = valid_mask & candidate.valid_mask & baseline.valid_mask
            count = mask.sum().clamp_min(1).to(x0_norm.dtype)
            angle = pelvis_angle_curve(candidate.motion)
            baseline_angle = pelvis_angle_curve(baseline.motion)
            mean_angle = _masked_mean(angle, mask)
            baseline_mean = _masked_mean(baseline_angle, mask)
            mean_loss = (mean_angle - self.target_mean_deg).square().mean()
            centered = angle - mean_angle.unsqueeze(-1)
            baseline_centered = baseline_angle - baseline_mean.unsqueeze(-1)
            shape_loss = (
                (centered - baseline_centered).square() * mask.to(angle.dtype)
            ).sum() / count
            candidate_norm = self._standardize(candidate.motion)
            baseline_norm = self._standardize(baseline.motion)
            mask_f = mask.to(x0_norm.dtype).unsqueeze(-1)
            motion_loss = ((candidate_norm - baseline_norm).square() * mask_f).sum() / (
                count * x0_norm.shape[-1]
            )
            loss = (
                cfg.mean_weight * mean_loss
                + cfg.shape_weight * shape_loss
                + cfg.motion_weight * motion_loss
            )
            gradient = torch.autograd.grad(loss, x0_norm, allow_unused=False)[0]
            gradient = torch.nan_to_num(gradient) * mask_f
            correction = cfg.guidance_strength * gradient
            correction_rms_before_cap = torch.sqrt(
                (correction.square() * mask_f).sum()
                / (count * x0_norm.shape[-1])
            )
            cap_scale = torch.clamp(
                torch.as_tensor(
                    cfg.max_correction_rms,
                    dtype=correction.dtype,
                    device=correction.device,
                )
                / correction_rms_before_cap.clamp_min(cfg.eps),
                max=1.0,
            )
            correction = correction * cap_scale
            x0_guided = x0_norm - correction
            # The authoritative post-update boundary is deliberately outside
            # the short loss graph.  It unifies root rotation and recomputes
            # joints/root-rotation/root-translation velocities together.  A
            # deterministic backtracking guard prevents the broad 276-D RMS
            # cap from overshooting the scalar angle objective.
            with torch.no_grad():
                before_error = (mean_angle - self.target_mean_deg).abs().mean()
                accepted_scale = 1.0
                post = None
                for _ in range(11):
                    trial = x0_norm.detach() - accepted_scale * correction.detach()
                    trial_post_phys = self._reconcile(
                        trial, output_standardized=False
                    )
                    trial_mean = _masked_mean(
                        pelvis_angle_curve(trial_post_phys.motion), mask
                    )
                    if (
                        (trial_mean - self.target_mean_deg).abs().mean()
                        <= before_error + cfg.eps
                    ):
                        post = self._reconcile(trial, output_standardized=True)
                        break
                    accepted_scale *= 0.5
                if post is None:
                    accepted_scale = 0.0
                    post = self._reconcile(
                        x0_norm.detach(), output_standardized=True
                    )
                correction = correction * accepted_scale
                x0_guided = x0_norm.detach() - correction
                x0_reconciled = post.motion
            corrected_velocity = velocity_from_x0(
                x_sigma.float(), x0_reconciled, sigma_value
            ).to(velocity.dtype)
            corrected_velocity = torch.where(
                valid_mask.unsqueeze(-1), corrected_velocity, velocity
            )
            correction_rms = torch.sqrt(
                (correction.square() * mask_f).sum()
                / (count * x0_norm.shape[-1])
            )
            diagnostics.update(
                {
                    "active": True,
                    "mean_angle_deg": [float(v) for v in mean_angle.detach().cpu()],
                    "baseline_mean_deg": [float(v) for v in baseline_mean.detach().cpu()],
                    "mean_loss": float(mean_loss.detach().cpu()),
                    "shape_loss": float(shape_loss.detach().cpu()),
                    "motion_loss": float(motion_loss.detach().cpu()),
                    "gradient_rms": float(
                        torch.sqrt((gradient.square() * mask_f).sum() / (count * x0_norm.shape[-1])).detach().cpu()
                    ),
                    "correction_rms_before_cap": float(correction_rms_before_cap.detach().cpu()),
                    "correction_rms": float(correction_rms.detach().cpu()),
                    "cap_applied": bool((cap_scale < 1).detach().cpu()),
                    "backtracking_scale": float(accepted_scale),
                    "valid_frames": int(mask.sum().detach().cpu()),
                }
            )
            if return_trace:
                diagnostics["trace"] = {
                    "velocity_model": velocity.detach().float().clone(),
                    "v_cfg": velocity.detach().float().clone(),
                    "x0_hat": x0_norm.detach().float().clone(),
                    "x0_guided": x0_guided.detach().float().clone(),
                    "x0_reconciled": x0_reconciled.detach().float().clone(),
                }
        self.last_diagnostics = diagnostics
        return corrected_velocity, diagnostics

    def _terminal_single(
        self,
        g0_norm: torch.Tensor,
        mask: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        single = AbsoluteMeanPelvisGuidanceV2(
            baseline_motion_norm=g0_norm.unsqueeze(0),
            valid_mask=mask.unsqueeze(0),
            mean=mean,
            std=std,
            target_mean_deg=self.target_mean_deg,
            config=self.config,
        )
        physical_result = single._reconcile(g0_norm.unsqueeze(0), output_standardized=False)
        physical = physical_result.motion[0]
        out_mask = physical_result.valid_mask[0]
        angle = pelvis_angle_curve(physical)
        mean_before = float(_masked_mean(angle.unsqueeze(0), out_mask.unsqueeze(0))[0].detach().cpu())
        residual = self.target_mean_deg - mean_before
        record: dict[str, Any] = {
            "target_mean_deg": self.target_mean_deg,
            "mean_before_deg": mean_before,
            "residual_before_deg": residual,
            "eligible": abs(residual) <= self.config.terminal_max_deg + self.config.eps,
            "triggered": False,
            "failed_residual_over_limit": False,
            "applied_deg": 0.0,
            "mean_after_deg": mean_before,
        }
        if not self.config.terminal_enabled:
            record["eligible"] = False
            record["disabled"] = True
            return g0_norm, out_mask, record
        if abs(residual) > self.config.terminal_max_deg + self.config.eps:
            record["failed_residual_over_limit"] = True
            return g0_norm, out_mask, record

        stream = recover_motion_stream(physical)
        candidates = torch.linspace(
            -self.config.terminal_max_deg,
            self.config.terminal_max_deg,
            401,
            dtype=physical.dtype,
            device=physical.device,
        )
        candidate_count = candidates.numel()
        root_stream = stream.root_rotation
        root_candidates = root_stream.unsqueeze(0).expand(candidate_count, -1, -1, -1)
        candidate_degrees = candidates[:, None].expand(-1, root_stream.shape[0])
        roots = apply_person_right_axis_rotation(
            root_candidates, candidate_degrees, eps=self.config.eps
        )
        skeleton = (
            load_smplx_neutral_22_skeleton(self.config.skeleton_path)
            if self.config.skeleton_path is not None
            else load_smplx_neutral_22_skeleton()
        )
        body_stream = stream.body_rotations
        valid_length = int(out_mask.sum().item())
        if valid_length:
            # ``recover_motion_stream`` recovers a T+1 stream from all packed
            # rows.  For a padded input its default final body pose would come
            # from the padded tail, while the valid hidden pose is pose L.
            # Body-local rotations have no velocity channel, so hold the last
            # valid local pose through the whole invalid suffix.  Root and
            # translation pose L already come from the last valid velocity row
            # and remain authoritative below.
            body_stream = body_stream.clone()
            body_stream[valid_length:] = body_stream[valid_length - 1 : valid_length]
        body_candidates = body_stream.unsqueeze(0).expand(
            candidate_count, -1, -1, -1, -1
        )
        translation_candidates = stream.root_translation.unsqueeze(0).expand(
            candidate_count, -1, -1
        )
        # Evaluate every terminal candidate at the same complete v2 boundary
        # as an ordinary diffusion update: authoritative roots + FK22 + all
        # 276 channels repacked from the T+1 pose stream.
        candidate_fk = differentiable_forward_kinematics(
            body_candidates,
            roots,
            translation_candidates,
            skeleton=skeleton,
        )
        candidate_finalized = finalize_motion(
            body_candidates,
            candidate_fk.joints,
            roots,
            translation_candidates,
            valid_mask=_prefix_pose_mask(out_mask).unsqueeze(0).expand(
                candidate_count, -1
            ),
        )
        probe_angle = pelvis_sagittal_tilt_degrees(
            roots[..., :-1, :, :], eps=self.config.eps
        )
        probe_mean = _masked_mean(
            probe_angle, candidate_finalized.valid_mask
        )
        best = torch.argmin((probe_mean - self.target_mean_deg).abs())
        applied = candidates[best]
        finalized = type(candidate_finalized)(
            motion=candidate_finalized.motion[best],
            valid_mask=candidate_finalized.valid_mask[best],
        )
        mean_aligned = _align_statistic(mean, finalized.motion, "mean")
        std_aligned = _align_statistic(std, finalized.motion, "std")
        g1 = ((finalized.motion - mean_aligned) / std_aligned).masked_fill(
            ~finalized.valid_mask.unsqueeze(-1), 0
        )
        mean_after = float(probe_mean[best].detach().cpu())
        record.update(
            {
                "triggered": abs(float(applied.detach().cpu())) > self.config.eps,
                "applied_deg": float(applied.detach().cpu()),
                "mean_after_deg": mean_after,
            }
        )
        return g1, finalized.valid_mask, record

    @torch.no_grad()
    def finalize_outputs(self, official_norm: torch.Tensor) -> AbsoluteMeanFinalOutputsV2:
        """Produce G0 and the separately auditable optional G1 endpoint."""

        g0_result = self._reconcile(official_norm.float(), output_standardized=True)
        g0 = g0_result.motion
        g1_rows, masks, records = [], [], []
        for index in range(g0.shape[0]):
            mean = self.mean
            std = self.std
            if mean.ndim == 2 and mean.shape[0] == g0.shape[0]:
                mean, std = mean[index:index + 1], std[index:index + 1]
            g1, g1_mask, record = self._terminal_single(
                g0[index], self.valid_mask[index].to(g0.device), mean, std
            )
            g1_rows.append(g1)
            masks.append(g1_mask)
            records.append(record)
        return AbsoluteMeanFinalOutputsV2(
            g0=g0,
            g1=torch.stack(g1_rows),
            g0_valid_mask=g0_result.valid_mask,
            g1_valid_mask=torch.stack(masks),
            terminal_records=tuple(records),
        )


__all__ = [
    "PROTOCOL_NAME",
    "AbsoluteMeanFinalOutputsV2",
    "AbsoluteMeanPelvisConfigV2",
    "AbsoluteMeanPelvisGuidanceV2",
    "pelvis_angle_curve",
]
