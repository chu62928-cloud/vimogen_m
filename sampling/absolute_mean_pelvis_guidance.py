"""Absolute mean model-space pelvis-pitch guidance for ViMoGen.

Protocol ``vimogen_absolute_mean_pelvis_v1`` keeps the historical M0/M1
paths opt-in.  At every active diffusion step it decodes a detached clean
endpoint in physical space, reconciles direct and velocity-integrated root
rotation on SO(3), evaluates the loss on that authoritative rotation, then
reconciles the edited endpoint again so every redundant velocity channel is
recomputed from one pose stream.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch

from motion_rep.finalize import finalize_motion
from motion_rep.pelvis_angle import pelvis_pitch_degrees, pelvis_pitch_curve_v2, pelvis_pitch_degrees_v2
from motion_rep.consistency_v2 import (
    differentiable_forward_kinematics,
    load_smplx_neutral_22_skeleton,
    reconcile_motion_tensor_v2,
)
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe
from motion_rep.reconciliation import ReconciliationConfig, reconcile_motion_tensor
from motion_rep.rotation_transform import axis_angle_to_mat3x3
from motion_rep.unified_finalizer import recover_motion_stream
from sampling.m1_guidance import predict_x0, velocity_from_x0


PROTOCOL_NAME = "vimogen_absolute_mean_pelvis_v1"


@dataclass(frozen=True)
class AbsoluteMeanPelvisConfig:
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
    # ``legacy_v1`` preserves every historical result.  ``fk_v2`` enables
    # full FK -> all-276D repacking at each guidance and terminal boundary.
    consistency_mode: str = "legacy_v1"
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
        if self.consistency_mode not in {
            "legacy_v1", "v1", "fk_v2", "v2", "vimogen_276d_consistency_v2"
        }:
            raise ValueError("consistency_mode must be legacy_v1 or fk_v2")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None
    ) -> "AbsoluteMeanPelvisConfig":
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
            consistency_mode=str(values.get("consistency_mode", values.get("representation_version", defaults.consistency_mode))),
            skeleton_path=(None if values.get("skeleton_path", defaults.skeleton_path) is None else str(values.get("skeleton_path", defaults.skeleton_path))),
        )


@dataclass(frozen=True)
class AbsoluteMeanFinalOutputs:
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
    """Return the authoritative model-space pitch curve in degrees."""

    root = decode_rot6d_safe(motion_phys[..., MOTION_LAYOUT.root_rotation])
    return pelvis_pitch_degrees(root, _canonical_heading(motion_phys))


def pelvis_angle_curve_v2_from_motion(
    motion_phys: torch.Tensor,
    *,
    skeleton=None,
) -> torch.Tensor:
    """v2 local-sagittal angle curve from one packed physical motion."""

    return pelvis_pitch_curve_v2(motion_phys, skeleton=skeleton)


def local_sagittal_normal(
    joints: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return per-frame local right axis (the sagittal-plane normal)."""

    if joints.ndim < 3 or joints.shape[-2:] != (22, 3):
        raise ValueError("joints must have shape [...,22,3]")
    up = torch.tensor([0.0, 0.0, 1.0], dtype=joints.dtype, device=joints.device)
    lateral = joints[..., 2, :] - joints[..., 1, :]
    lateral = lateral - (lateral * up).sum(-1, keepdim=True) * up
    norm = torch.linalg.vector_norm(lateral, dim=-1, keepdim=True)
    fallback = torch.tensor([1.0, 0.0, 0.0], dtype=joints.dtype, device=joints.device)
    return torch.where(norm > eps, lateral / norm.clamp_min(eps), fallback)


def apply_local_sagittal_correction(
    root_rotation: torch.Tensor,
    joints: torch.Tensor,
    angle_deg: torch.Tensor | float,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Left-multiply roots by a signed correction around each local right axis."""

    if root_rotation.shape[-2:] != (3, 3) or joints.shape[:-2] != root_rotation.shape[:-2] or joints.shape[-2:] != (22, 3):
        raise ValueError("root_rotation [...,3,3] and joints [...,22,3] must share prefix shape")
    lateral = local_sagittal_normal(joints, eps=eps)
    amount = torch.as_tensor(angle_deg, dtype=root_rotation.dtype, device=root_rotation.device)
    while amount.ndim < lateral.ndim - 1:
        amount = amount.unsqueeze(-1)
    delta = axis_angle_to_mat3x3(lateral * torch.deg2rad(amount).unsqueeze(-1))
    return delta @ root_rotation


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum(dim=-1).clamp_min(1).to(values.dtype)
    return (values * mask.to(values.dtype)).sum(dim=-1) / count


class AbsoluteMeanPelvisGuidance:
    """Guide the absolute valid-frame mean of fused SO(3) pelvis pitch."""

    def __init__(
        self,
        *,
        baseline_motion_norm: torch.Tensor,
        valid_mask: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        target_mean_deg: float,
        config: AbsoluteMeanPelvisConfig | None = None,
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
        self.config = config or AbsoluteMeanPelvisConfig()
        self.last_diagnostics: dict[str, Any] = {
            "protocol": PROTOCOL_NAME,
            "enabled": self.config.enabled,
            "active": False,
        }

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def slice(self, index: int) -> "AbsoluteMeanPelvisGuidance":
        if self.baseline_motion_norm.shape[0] == 1:
            return self
        mean, std = self.mean, self.std
        if mean.ndim == 2 and mean.shape[0] == self.baseline_motion_norm.shape[0]:
            mean, std = mean[index:index + 1], std[index:index + 1]
        return AbsoluteMeanPelvisGuidance(
            baseline_motion_norm=self.baseline_motion_norm[index:index + 1],
            valid_mask=self.valid_mask[index:index + 1],
            mean=mean,
            std=std,
            target_mean_deg=self.target_mean_deg,
            config=self.config,
        )

    def protocol_record(self) -> dict[str, Any]:
        is_v2 = self.config.consistency_mode in {
            "fk_v2", "v2", "vimogen_276d_consistency_v2"
        }
        return {
            "protocol": "vimogen_absolute_mean_pelvis_v2" if is_v2 else PROTOCOL_NAME,
            "target_semantics": "absolute_valid_frame_mean_model_space_pelvis_pitch_deg",
            "target_mean_deg": self.target_mean_deg,
            "angle_authority": "fk22_local_sagittal_plane" if is_v2 else "fused_direct_root_and_rotation_velocity_integral_SO3",
            "post_update_boundary": "fk22_then_repack_all_276_channels" if is_v2 else "unified_root_rotation_and_all_velocities_recomputed",
            "config": asdict(self.config),
        }

    def _reconcile(
        self, motion_norm: torch.Tensor, *, output_standardized: bool
    ):
        if self.config.consistency_mode in {"fk_v2", "v2", "vimogen_276d_consistency_v2"}:
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
        cfg = ReconciliationConfig(
            correction_window=self.config.fusion_window,
            anchor_weight=self.config.anchor_weight,
            root_rotation_anchor_weight=self.config.anchor_weight,
        )
        return reconcile_motion_tensor(
            motion_norm,
            config=cfg,
            valid_mask=self.valid_mask.to(motion_norm.device),
            mean=self.mean,
            std=self.std,
            input_standardized=True,
            output_standardized=output_standardized,
            output_dtype=torch.float32,
        )

    @property
    def _uses_v2(self) -> bool:
        return self.config.consistency_mode in {
            "fk_v2", "v2", "vimogen_276d_consistency_v2"
        }

    def _angle_curve(self, motion: torch.Tensor) -> torch.Tensor:
        if self._uses_v2:
            skeleton = None if self.config.skeleton_path is None else load_smplx_neutral_22_skeleton(self.config.skeleton_path)
            return pelvis_angle_curve_v2_from_motion(motion, skeleton=skeleton)
        return pelvis_angle_curve(motion)

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
            angle = self._angle_curve(candidate.motion)
            baseline_angle = self._angle_curve(baseline.motion)
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
                        self._angle_curve(trial_post_phys.motion), mask
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
        single = AbsoluteMeanPelvisGuidance(
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
        angle = self._angle_curve(physical)
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
        if self._uses_v2:
            # v2's correction axis is the per-frame local right/lateral axis,
            # i.e. the normal of the local sagittal plane.  A fixed world x
            # axis is only correct at yaw=0 and silently changes the meaning
            # of G1 on turning motions.
            skeleton = None if self.config.skeleton_path is None else load_smplx_neutral_22_skeleton(self.config.skeleton_path)
            if skeleton is None:
                skeleton = load_smplx_neutral_22_skeleton()
            up = torch.tensor([0.0, 0.0, 1.0], dtype=physical.dtype, device=physical.device)
            lateral = local_sagittal_normal(stream.joint_positions, eps=self.config.eps)
            axis_angle = lateral.unsqueeze(0) * torch.deg2rad(candidates).view(-1, 1, 1)
            delta = axis_angle_to_mat3x3(axis_angle)
            roots = delta @ stream.root_rotation.unsqueeze(0)
            body = stream.body_rotations.unsqueeze(0).expand(candidates.numel(), -1, -1, -1, -1)
            translation = stream.root_translation.unsqueeze(0).expand(candidates.numel(), -1, -1)
            fk = differentiable_forward_kinematics(body, roots, translation, skeleton=skeleton)
            probe_angle = pelvis_pitch_degrees_v2(
                fk.joints[..., :-1, :, :],
                rest_offsets=skeleton.rest_offsets,
                eps=self.config.eps,
            )
        else:
            axis_angle = torch.zeros((candidates.numel(), 3), dtype=physical.dtype, device=physical.device)
            axis_angle[:, 0] = torch.deg2rad(candidates)
            delta = axis_angle_to_mat3x3(axis_angle)
            roots = delta[:, None] @ stream.root_rotation[None]
            probe_motion = physical.unsqueeze(0).expand(candidates.numel(), -1, -1)
            probe_heading = _canonical_heading(probe_motion)
            probe_angle = pelvis_pitch_degrees(roots[:, :-1], probe_heading)
        probe_mean = _masked_mean(
            probe_angle, out_mask.unsqueeze(0).expand(candidates.numel(), -1)
        )
        best = torch.argmin((probe_mean - self.target_mean_deg).abs())
        applied = candidates[best]
        edited_root = delta[best] @ stream.root_rotation
        if self._uses_v2:
            # Re-run FK for the chosen root and only then pack all 276
            # channels.  This is the terminal counterpart of the per-step
            # v2 projection and prevents stale J/dJ channels after G1.
            chosen_fk = differentiable_forward_kinematics(
                stream.body_rotations,
                edited_root,
                stream.root_translation,
                skeleton=skeleton,
            )
            pose_mask = torch.cat((mask, mask[-1:]))
            finalized = finalize_motion(
                stream.body_rotations,
                chosen_fk.joints,
                edited_root,
                stream.root_translation,
                valid_mask=pose_mask,
            )
        else:
            pose_mask = torch.cat((mask, mask[-1:]))
            finalized = finalize_motion(
                stream.body_rotations,
                stream.joint_positions,
                edited_root,
                stream.root_translation,
                valid_mask=pose_mask,
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
    def finalize_outputs(self, official_norm: torch.Tensor) -> AbsoluteMeanFinalOutputs:
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
        return AbsoluteMeanFinalOutputs(
            g0=g0,
            g1=torch.stack(g1_rows),
            g0_valid_mask=g0_result.valid_mask,
            g1_valid_mask=torch.stack(masks),
            terminal_records=tuple(records),
            protocol=("vimogen_absolute_mean_pelvis_v2" if self._uses_v2 else PROTOCOL_NAME),
        )


__all__ = [
    "PROTOCOL_NAME",
    "AbsoluteMeanFinalOutputs",
    "AbsoluteMeanPelvisConfig",
    "AbsoluteMeanPelvisGuidance",
    "apply_local_sagittal_correction",
    "local_sagittal_normal",
    "pelvis_angle_curve",
    "pelvis_angle_curve_v2_from_motion",
]
