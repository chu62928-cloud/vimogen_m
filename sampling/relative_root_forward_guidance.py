"""Relative root-forward v1 guidance with a direct-pose authority boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch

from motion_rep.consistency_v2 import default_smplx_neutral_22_skeleton, load_smplx_neutral_22_skeleton
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe
from motion_rep.rotation_transform import axis_angle_to_mat3x3
from motion_rep.pose_authority import (
    PROTOCOL_NAME,
    AuthorityProjection,
    RelativeRootForwardTargets,
    _authority_from_streams,
    _direct_streams,
    _prefix_source,
    _root_forward,
    authority_project,
    forward_vector_loss,
    prepare_targets,
    whole_body_audit,
)


RESIDUAL_ADAPTIVE_PROTOCOL_NAME = (
    "vimogen_relative_root_forward_v1_1_residual_adaptive"
)


def apply_root_forward_tangent(
    root_rotation: torch.Tensor,
    frozen_right_axis: torch.Tensor,
    correction_deg: torch.Tensor | float,
) -> torch.Tensor:
    """Left-multiply root rotations by a scalar correction around M0 ``r``."""

    angle = torch.as_tensor(correction_deg, dtype=root_rotation.dtype, device=root_rotation.device)
    while angle.ndim < root_rotation.ndim - 1:
        angle = angle.unsqueeze(-1)
    axis = frozen_right_axis.to(device=root_rotation.device, dtype=root_rotation.dtype)
    return axis_angle_to_mat3x3(axis * (angle * math.pi / 180.0)) @ root_rotation


def _broadcast_sigma(sigma: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(sigma, dtype=reference.dtype, device=reference.device)
    while value.ndim < reference.ndim:
        value = value.unsqueeze(-1)
    return value


def predict_x0(x_sigma: torch.Tensor, velocity: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
    if x_sigma.shape != velocity.shape:
        raise ValueError("x_sigma and velocity must have identical shapes")
    return x_sigma - _broadcast_sigma(sigma, x_sigma) * velocity


def velocity_from_x0(x_sigma: torch.Tensor, x0: torch.Tensor, sigma: torch.Tensor | float, *, eps: float = 1e-6) -> torch.Tensor:
    if x_sigma.shape != x0.shape:
        raise ValueError("x_sigma and x0 must have identical shapes")
    value = _broadcast_sigma(sigma, x_sigma)
    if torch.any(value.abs() < eps):
        raise ValueError("velocity_from_x0 is undefined at sigma=0")
    return (x_sigma - x0) / value


@dataclass(frozen=True)
class RelativeRootForwardConfig:
    protocol: str = PROTOCOL_NAME
    enabled: bool = True
    guidance_strength: float = 1.0
    sigma_min: float = 0.25
    sigma_max: float = 0.65
    motion_weight: float = 0.1
    base_step_deg: float = 1.0
    residual_gain: float = 1.0
    max_step_deg: float = 8.0
    max_correction_rms: float = 0.05
    max_backtracks: int = 11
    eps: float = 1e-6
    curve_std_epsilon_deg: float = 1e-4
    skeleton_path: str | None = None
    trace_enabled: bool = False

    def __post_init__(self) -> None:
        if self.protocol not in {PROTOCOL_NAME, RESIDUAL_ADAPTIVE_PROTOCOL_NAME, "v1"}:
            raise ValueError(
                f"protocol must be {PROTOCOL_NAME} or {RESIDUAL_ADAPTIVE_PROTOCOL_NAME}"
            )
        if self.guidance_strength < 0 or self.motion_weight < 0:
            raise ValueError("guidance_strength and motion_weight must be non-negative")
        if not 0 <= self.sigma_min <= self.sigma_max <= 1:
            raise ValueError("sigma window must satisfy 0 <= sigma_min <= sigma_max <= 1")
        if self.base_step_deg <= 0 or self.max_correction_rms <= 0:
            raise ValueError("base_step_deg and max_correction_rms must be positive")
        if self.residual_gain <= 0 or self.max_step_deg <= 0:
            raise ValueError("residual_gain and max_step_deg must be positive")
        if self.max_backtracks < 0:
            raise ValueError("max_backtracks must be non-negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "RelativeRootForwardConfig":
        values = values or {}
        defaults = cls()
        kwargs = asdict(defaults)
        # Configuration absence is historical no-op; explicit construction is
        # enabled for unit tests and for the opt-in YAML section.
        kwargs["enabled"] = bool(values.get("enabled", False))
        requested_protocol = str(values.get("protocol", PROTOCOL_NAME))
        for key in kwargs:
            if key in values:
                kwargs[key] = values[key]
        kwargs["protocol"] = PROTOCOL_NAME if requested_protocol == "v1" else requested_protocol
        if kwargs["skeleton_path"] is not None:
            kwargs["skeleton_path"] = str(kwargs["skeleton_path"])
        return cls(**kwargs)


@dataclass(frozen=True)
class RelativeRootForwardFinalOutputs:
    g0: torch.Tensor
    g0_valid_mask: torch.Tensor
    projection_audits: tuple[dict[str, Any], ...]
    whole_body_audits: tuple[dict[str, Any], ...] = ()
    protocol: str = PROTOCOL_NAME
    metrics: dict[str, Any] | None = None


class RelativeRootForwardGuidance:
    """Guide only a one-dimensional root-forward tangent at each frame."""

    PROTOCOL = PROTOCOL_NAME

    def __init__(
        self,
        *,
        baseline_motion_norm: torch.Tensor,
        valid_mask: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        target_delta_deg: float,
        config: RelativeRootForwardConfig | None = None,
        skeleton=None,
    ) -> None:
        if baseline_motion_norm.ndim != 3 or baseline_motion_norm.shape[-1] != 276:
            raise ValueError("baseline_motion_norm must have shape [B,T,276]")
        if valid_mask.shape != baseline_motion_norm.shape[:2] or valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must be bool[B,T]")
        if mean.shape[-1] != 276 or std.shape[-1] != 276:
            raise ValueError("mean and std must have 276 channels")
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("std must be finite and positive")
        self.config = config or RelativeRootForwardConfig()
        self.mean = mean.detach()
        self.std = std.detach()
        self.valid_mask = valid_mask.detach()
        if skeleton is None:
            skeleton = (
                default_smplx_neutral_22_skeleton()
                if self.config.skeleton_path is None
                else load_smplx_neutral_22_skeleton(self.config.skeleton_path)
            )
        self.skeleton = skeleton
        # The M0 endpoint is authoritative exactly once.  All later steps use
        # these detached physical tensors and frozen geometric targets.
        projection = authority_project(
            baseline_motion_norm.detach(),
            valid_mask=self.valid_mask,
            mean=self.mean,
            std=self.std,
            input_standardized=True,
            output_standardized=True,
            output_dtype=torch.float32,
            skeleton=skeleton,
        )
        self.baseline_motion_norm = projection.motion.detach()
        self.baseline_physical = projection.physical_motion.detach()
        self.baseline_projection_audits = projection.audits
        self.targets = prepare_targets(self.baseline_physical, self.valid_mask, target_delta_deg)
        self.target_delta_deg = float(target_delta_deg)
        self.last_diagnostics: dict[str, Any] = {"protocol": self.PROTOCOL, "enabled": self.enabled, "active": False}

    @classmethod
    def _from_cached(cls, source: "RelativeRootForwardGuidance", index: int) -> "RelativeRootForwardGuidance":
        obj = object.__new__(cls)
        obj.config = source.config
        obj.mean = source.mean if source.mean.ndim == 1 else source.mean[index:index + 1]
        obj.std = source.std if source.std.ndim == 1 else source.std[index:index + 1]
        obj.valid_mask = source.valid_mask[index:index + 1]
        obj.skeleton = source.skeleton
        obj.baseline_motion_norm = source.baseline_motion_norm[index:index + 1]
        obj.baseline_physical = source.baseline_physical[index:index + 1]
        obj.baseline_projection_audits = (source.baseline_projection_audits[index],)
        targets = source.targets
        obj.targets = RelativeRootForwardTargets(
            targets.f0[index:index + 1], targets.h0[index:index + 1], targets.r0[index:index + 1],
            targets.phi0_deg[index:index + 1], targets.target_forward[index:index + 1],
            targets.target_phi_deg[index:index + 1], targets.valid_mask[index:index + 1], targets.delta_deg,
        )
        obj.target_delta_deg = source.target_delta_deg
        obj.last_diagnostics = {"protocol": obj.PROTOCOL, "enabled": obj.enabled, "active": False}
        return obj

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def trace_enabled(self) -> bool:
        return bool(self.config.trace_enabled)

    def slice(self, index: int) -> "RelativeRootForwardGuidance":
        if self.baseline_motion_norm.shape[0] == 1:
            return self
        if not 0 <= index < self.baseline_motion_norm.shape[0]:
            raise IndexError(index)
        return self._from_cached(self, index)

    def protocol_record(self) -> dict[str, Any]:
        return {
            "protocol": self.PROTOCOL,
            "authority": "body_pose/root_rotation/root_translation",
            "derived": "J/dJ/dR/dT_from_fk_and_forward_difference",
            "guidance_degree_of_freedom": "per_frame_scalar_left_rotation_about_frozen_M0_right_axis",
            "guided_channels": ["root_rotation"],
            "non_guided_derived_channels": ["J", "dJ", "dR", "dT"],
            "target_delta_deg": self.target_delta_deg,
            "target_semantics": "relative_root_forward_downward_from_frozen_M0",
            "config": asdict(self.config),
            "baseline_projection_audits": list(self.baseline_projection_audits),
        }

    def _physical(self, motion_norm: torch.Tensor) -> torch.Tensor:
        mean, std = self.mean.to(motion_norm.device), self.std.to(motion_norm.device)
        if mean.ndim == 2:
            mean, std = mean[:, None, :], std[:, None, :]
        return motion_norm.float() * std.float() + mean.float()

    def _standardize(self, motion_physical: torch.Tensor) -> torch.Tensor:
        mean, std = self.mean.to(motion_physical.device), self.std.to(motion_physical.device)
        if mean.ndim == 2:
            mean, std = mean[:, None, :], std[:, None, :]
        return (motion_physical - mean.float()) / std.float()

    def _build_candidate(self, x0_physical: torch.Tensor, alpha_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source, pose_mask = _prefix_source(x0_physical, self.valid_mask.to(x0_physical.device))
        body, root, translation = _direct_streams(source)
        r0 = self.targets.r0.to(device=root.device, dtype=root.dtype)
        root = apply_root_forward_tangent(root, r0, alpha_deg)
        packed = _authority_from_streams(body, root, translation, pose_mask, skeleton=self.skeleton)
        return packed, pose_mask[:, :-1]

    def _make_proposal(
        self,
        *,
        base_phys: torch.Tensor,
        f_base: torch.Tensor,
        target_f: torch.Tensor,
        grad: torch.Tensor,
        grad_rms: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return the v1 fixed-RMS proposal and optional diagnostics."""

        del base_phys, f_base, target_f
        proposal = -self.config.guidance_strength * self.config.base_step_deg * grad / grad_rms
        proposal = proposal * frame_mask.to(proposal.dtype)
        return proposal, {}

    @staticmethod
    def _masked_motion_loss(candidate: torch.Tensor, baseline: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff = candidate - baseline
        mask_f = mask.to(dtype=diff.dtype).unsqueeze(-1)
        return (diff.square() * mask_f).sum() / (mask_f.sum().clamp_min(1.0) * diff.shape[-1])

    def forward_loss(self, motion_norm: torch.Tensor, *, target_forward: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate the geometric loss on a normalized endpoint.

        Supplying ``target_forward=self.targets.f0`` is the explicit bypass
        used by the zero-dose geometry test; it avoids the production
        zero-dose shortcut while checking the same vector implementation.
        """

        physical = self._physical(motion_norm)
        source, _ = _prefix_source(physical, self.valid_mask.to(physical.device))
        root = decode_rot6d_safe(source[..., MOTION_LAYOUT.root_rotation])
        forward = root @ torch.tensor([0., 0., 1.], dtype=root.dtype, device=root.device)
        target = self.targets.target_forward if target_forward is None else target_forward
        return forward_vector_loss(forward, target.to(device=forward.device, dtype=forward.dtype), self.valid_mask.to(forward.device))

    def tangent_gradient(
        self,
        motion_norm: torch.Tensor,
        *,
        target_forward: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the scalar tangent gradient used by the geometry test."""

        with torch.enable_grad():
            physical = self._physical(motion_norm.detach())
            alpha = torch.zeros(
                physical.shape[:2], dtype=torch.float32, device=physical.device, requires_grad=True
            )
            candidate, _ = self._build_candidate(physical, alpha)
            root = decode_rot6d_safe(candidate[..., MOTION_LAYOUT.root_rotation])
            forward = root @ torch.tensor([0., 0., 1.], dtype=root.dtype, device=root.device)
            target = self.targets.f0 if target_forward is None else target_forward
            loss = forward_vector_loss(
                forward,
                target.to(device=forward.device, dtype=forward.dtype),
                self.valid_mask.to(forward.device),
            )
            if float(loss.detach()) <= self.config.eps:
                return torch.zeros_like(alpha)
            return torch.autograd.grad(loss, alpha, allow_unused=False)[0]

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
        diagnostics: dict[str, Any] = {"protocol": self.PROTOCOL, "enabled": cfg.enabled, "active": False, "sigma": sigma_value, "target_delta_deg": self.target_delta_deg, "accepted": False}
        x0_hat = predict_x0(x_sigma.float(), velocity.float(), sigma_value).detach()
        inactive = (not math.isfinite(sigma_value) or not cfg.enabled or cfg.guidance_strength <= cfg.eps or abs(self.target_delta_deg) <= cfg.eps or sigma_value <= cfg.eps or sigma_value < cfg.sigma_min or sigma_value > cfg.sigma_max)
        if inactive:
            if return_trace:
                diagnostics["trace"] = {"velocity_model": velocity.detach().float().clone(), "v_cfg": velocity.detach().float().clone(), "x0_hat": x0_hat.clone(), "x0_guided": x0_hat.clone(), "x0_reconciled": x0_hat.clone()}
            self.last_diagnostics = diagnostics
            return velocity, diagnostics
        if not torch.isfinite(x0_hat).all():
            diagnostics["rejected_reason"] = "non_finite_x0_hat"
            self.last_diagnostics = diagnostics
            return velocity

        with torch.enable_grad(), torch.amp.autocast(device_type=x_sigma.device.type, enabled=False):
            x0_physical = self._physical(x0_hat)
            if not torch.isfinite(x0_physical).all():
                diagnostics["rejected_reason"] = "non_finite_physical_endpoint"
                self.last_diagnostics = diagnostics
                return velocity
            baseline = self.baseline_motion_norm.to(device=x_sigma.device, dtype=torch.float32)
            targets = self.targets
            alpha = torch.zeros(x_sigma.shape[:2], dtype=torch.float32, device=x_sigma.device, requires_grad=True)
            base_phys, _ = self._build_candidate(x0_physical, alpha)
            f_base = decode_rot6d_safe(base_phys[..., MOTION_LAYOUT.root_rotation]) @ torch.tensor([0., 0., 1.], device=x_sigma.device)
            target_f = targets.target_forward.to(device=x_sigma.device, dtype=torch.float32)
            frame_mask = valid_mask
            base_forward_loss = forward_vector_loss(f_base, target_f, frame_mask)
            base_norm = self._standardize(base_phys)
            base_motion_loss = self._masked_motion_loss(base_norm, baseline, frame_mask)
            base_total = base_forward_loss + cfg.motion_weight * base_motion_loss
            grad = torch.autograd.grad(base_total, alpha, allow_unused=False)[0]
            grad = grad * frame_mask.to(grad.dtype)
            grad_rms = torch.sqrt((grad.square() * frame_mask).sum() / frame_mask.sum().clamp_min(1)).clamp_min(cfg.eps)
            proposal, proposal_diagnostics = self._make_proposal(
                base_phys=base_phys,
                f_base=f_base,
                target_f=target_f,
                grad=grad,
                grad_rms=grad_rms,
                frame_mask=frame_mask,
            )
            if proposal_diagnostics.get("proposal_rejected", False):
                diagnostics.update(proposal_diagnostics)
                diagnostics["rejected_reason"] = proposal_diagnostics.get(
                    "rejected_reason", "proposal_rejected"
                )
                self.last_diagnostics = diagnostics
                return velocity, diagnostics
            test_phys, _ = self._build_candidate(x0_physical, proposal)
            test_norm = self._standardize(test_phys)
            correction_rms = torch.sqrt(((test_norm - base_norm).square() * frame_mask.unsqueeze(-1)).sum() / (frame_mask.sum().clamp_min(1) * 276))
            if correction_rms > cfg.max_correction_rms:
                proposal = proposal * (cfg.max_correction_rms / correction_rms.clamp_min(cfg.eps))
            accepted = None
            accepted_losses = None
            for attempt in range(cfg.max_backtracks + 1):
                scale = 0.5 ** attempt
                candidate_alpha = proposal * scale
                candidate_phys, _ = self._build_candidate(x0_physical, candidate_alpha)
                candidate_norm = self._standardize(candidate_phys)
                root_candidate = decode_rot6d_safe(candidate_phys[..., MOTION_LAYOUT.root_rotation])
                f_candidate = root_candidate @ torch.tensor([0., 0., 1.], device=x_sigma.device)
                forward_new = forward_vector_loss(f_candidate, target_f, frame_mask)
                motion_new = self._masked_motion_loss(candidate_norm, baseline, frame_mask)
                total_new = forward_new + cfg.motion_weight * motion_new
                finite = torch.isfinite(candidate_norm).all() and torch.isfinite(total_new)
                if bool(finite and forward_new <= base_forward_loss + cfg.eps and total_new <= base_total + cfg.eps):
                    accepted = (candidate_phys, candidate_norm, candidate_alpha, scale)
                    accepted_losses = (forward_new, motion_new, total_new)
                    break
            if accepted is None:
                corrected_velocity = velocity
                x0_guided = x0_hat
                x0_reconciled = x0_hat
                applied_correction_rms = 0.0
            else:
                candidate_phys, candidate_norm, candidate_alpha, scale = accepted
                corrected_velocity = velocity_from_x0(x_sigma.float(), candidate_norm, sigma_value).to(dtype=velocity.dtype)
                corrected_velocity = torch.where(valid_mask.unsqueeze(-1), corrected_velocity, velocity)
                x0_guided = candidate_norm.detach()
                x0_reconciled = candidate_norm.detach()
                applied_correction_rms = float(torch.sqrt(((candidate_norm - base_norm).square() * frame_mask.unsqueeze(-1)).sum() / (frame_mask.sum().clamp_min(1) * 276)).detach().cpu())
                alpha_valid = candidate_alpha[frame_mask]
                diagnostics.update({"accepted": True, "accepted_scale": float(scale), "forward_loss_new": float(accepted_losses[0].detach().cpu()), "motion_loss_new": float(accepted_losses[1].detach().cpu()), "total_loss_new": float(accepted_losses[2].detach().cpu()), "accepted_alpha_mean_deg": float(alpha_valid.mean().detach().cpu()), "accepted_alpha_rms_deg": float(torch.sqrt(alpha_valid.square().mean()).detach().cpu()), "accepted_alpha_max_deg": float(alpha_valid.abs().max().detach().cpu()), "backtrack_count": int(attempt)})
            diagnostics.update({"active": True, "forward_loss_old": float(base_forward_loss.detach().cpu()), "motion_loss_old": float(base_motion_loss.detach().cpu()), "total_loss_old": float(base_total.detach().cpu()), "gradient_rms": float(grad_rms.detach().cpu()), "proposal_rms_deg": float(torch.sqrt((proposal.square() * frame_mask).sum() / frame_mask.sum().clamp_min(1)).detach().cpu()), "correction_rms": applied_correction_rms, "valid_frames": int(frame_mask.sum().detach().cpu())})
            diagnostics.update(proposal_diagnostics)
            if return_trace:
                diagnostics["trace"] = {"velocity_model": velocity.detach().float().clone(), "v_cfg": velocity.detach().float().clone(), "x0_hat": x0_hat.clone(), "x0_guided": x0_guided.clone(), "x0_reconciled": x0_reconciled.clone()}
        self.last_diagnostics = diagnostics
        return corrected_velocity, diagnostics

    def finalize_outputs(self, official_norm: torch.Tensor) -> RelativeRootForwardFinalOutputs:
        if abs(self.target_delta_deg) <= self.config.eps:
            # Zero dose is a true identity on the frozen, already projected
            # baseline; it is not a second pass through smoothing or FK.
            outputs = RelativeRootForwardFinalOutputs(
                self.baseline_motion_norm.clone(),
                self.valid_mask.clone(),
                self.baseline_projection_audits,
                tuple(
                    whole_body_audit(self.baseline_physical[i:i + 1], self.baseline_physical[i:i + 1], self.valid_mask[i:i + 1])
                    for i in range(self.baseline_physical.shape[0])
                ),
                protocol=self.PROTOCOL,
            )
            from evaluation.relative_root_forward_v1 import compute_relative_root_forward_metrics
            return RelativeRootForwardFinalOutputs(
                outputs.g0, outputs.g0_valid_mask, outputs.projection_audits,
                outputs.whole_body_audits, outputs.protocol,
                compute_relative_root_forward_metrics(
                    self.baseline_physical,
                    self.baseline_physical,
                    self.valid_mask,
                    self.target_delta_deg,
                    skeleton=self.skeleton,
                    protocol_name=self.PROTOCOL,
                ),
            )
        projection = authority_project(
            official_norm.float(),
            valid_mask=self.valid_mask.to(official_norm.device),
            mean=self.mean,
            std=self.std,
            input_standardized=True,
            output_standardized=True,
            output_dtype=torch.float32,
            skeleton=self.skeleton,
        )
        outputs = RelativeRootForwardFinalOutputs(
            projection.motion,
            projection.valid_mask,
            projection.audits,
                tuple(
                whole_body_audit(
                    self.baseline_physical[i:i + 1],
                    projection.physical_motion[i:i + 1],
                    self.valid_mask[i:i + 1].to(projection.physical_motion.device),
                )
                    for i in range(projection.physical_motion.shape[0])
                ),
                protocol=self.PROTOCOL,
            )
        from evaluation.relative_root_forward_v1 import compute_relative_root_forward_metrics
        return RelativeRootForwardFinalOutputs(
            outputs.g0, outputs.g0_valid_mask, outputs.projection_audits,
            outputs.whole_body_audits, outputs.protocol,
            compute_relative_root_forward_metrics(
                self.baseline_physical,
                projection.physical_motion,
                self.valid_mask.to(projection.physical_motion.device),
                self.target_delta_deg,
                    skeleton=self.skeleton,
                    protocol_name=self.PROTOCOL,
                ),
        )


RelativeRootForwardStrategy = RelativeRootForwardGuidance


__all__ = [
    "PROTOCOL_NAME", "RelativeRootForwardConfig", "RelativeRootForwardFinalOutputs",
    "RelativeRootForwardGuidance", "predict_x0", "velocity_from_x0",
    "apply_root_forward_tangent",
    "RelativeRootForwardStrategy",
]
