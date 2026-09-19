"""Constraint-first root-forward guidance with trunk stabilization (v1.2).

The v1.2 protocol keeps the direct-pose authority boundary from v1.1, but it
does not add geometric errors with arbitrary coefficients.  Root pitch,
heading, and trunk-direction errors are separate acceptance constraints;
representation changes are independent budgets.  Only root rotation and the
three spine local rotations are editable.  Every accepted edit is passed
through FK and all velocity channels are rebuilt from the resulting pose.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch

from motion_rep.consistency_v2 import differentiable_forward_kinematics
from motion_rep.phase1 import MOTION_LAYOUT, SMPLX_22_JOINT_INDEX, decode_rot6d_safe
from motion_rep.rotation_transform import axis_angle_to_mat3x3
from motion_rep.pose_authority import (
    _authority_from_streams,
    _direct_streams,
    _geodesic,
    _prefix_source,
    _root_forward,
    authority_project,
    forward_vector_loss,
    whole_body_audit,
)
from sampling.relative_root_forward_guidance import (
    TRUNK_STABILIZED_PROTOCOL_NAME,
    RelativeRootForwardConfig,
    RelativeRootForwardFinalOutputs,
    RelativeRootForwardGuidance,
    _broadcast_sigma,
    predict_x0,
    velocity_from_x0,
)


PROTOCOL_NAME = TRUNK_STABILIZED_PROTOCOL_NAME
UP = torch.tensor([0.0, 0.0, 1.0])
SPINE_NAMES = ("spine1", "spine2", "spine3")
SPINE_BODY_INDICES = tuple(SMPLX_22_JOINT_INDEX[name] - 1 for name in SPINE_NAMES)
SPINE_JOINT_INDICES = tuple(SMPLX_22_JOINT_INDEX[name] for name in SPINE_NAMES)
EPS_DEG = 1e-4


def _unit(value: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    valid = torch.isfinite(value).all(-1) & torch.isfinite(norm[..., 0]) & (norm[..., 0] > eps)
    return value / norm.clamp_min(eps), valid


def signed_axis_residual_deg(
    current: torch.Tensor,
    target: torch.Tensor,
    axis: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed angle for rotating ``current`` to ``target`` about ``axis``.

    Both vectors are projected to the plane normal to the axis.  This is used
    for the root pitch, root heading, and trunk sagittal residuals.
    """

    if current.shape != target.shape or current.shape != axis.shape:
        raise ValueError("current, target and axis must have identical shapes")
    if current.ndim != 3 or current.shape[-1] != 3:
        raise ValueError("vectors must have shape [B,T,3]")
    if valid_mask.shape != current.shape[:2] or valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be bool[B,T]")
    axis_u, axis_valid = _unit(axis, eps)
    current_proj = current - (current * axis_u).sum(-1, keepdim=True) * axis_u
    target_proj = target - (target * axis_u).sum(-1, keepdim=True) * axis_u
    current_u, current_valid = _unit(current_proj, eps)
    target_u, target_valid = _unit(target_proj, eps)
    finite = (
        torch.isfinite(current_u).all(-1)
        & torch.isfinite(target_u).all(-1)
        & axis_valid
        & current_valid
        & target_valid
    )
    sine = (axis_u * torch.cross(current_u, target_u, dim=-1)).sum(-1)
    cosine = (current_u * target_u).sum(-1).clamp(-1.0, 1.0)
    residual = torch.atan2(sine, cosine) * (180.0 / math.pi)
    return residual, finite & valid_mask


def _apply_world_axis(
    rotation: torch.Tensor, axis: torch.Tensor, angle_deg: torch.Tensor
) -> torch.Tensor:
    angle = angle_deg.to(dtype=rotation.dtype, device=rotation.device)
    axis = axis.to(dtype=rotation.dtype, device=rotation.device)
    delta = axis_angle_to_mat3x3(axis * (angle.unsqueeze(-1) * math.pi / 180.0))
    return delta @ rotation


def _trunk_direction(physical: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    if valid_mask is None:
        valid_mask = torch.ones(physical.shape[:2], dtype=torch.bool, device=physical.device)
    source, _ = _prefix_source(
        physical, valid_mask.to(device=physical.device)
    )
    joints = source[..., MOTION_LAYOUT.joints].reshape(*source.shape[:-1], 22, 3)
    value = joints[..., SMPLX_22_JOINT_INDEX["neck"], :] - joints[..., SMPLX_22_JOINT_INDEX["spine1"], :]
    value, valid = _unit(value, 1e-7)
    if not bool(valid.all()):
        raise ValueError("trunk direction is degenerate or non-finite")
    return value


@dataclass(frozen=True)
class TrunkStabilizedRootForwardConfig(RelativeRootForwardConfig):
    """Configuration for the independent v1.2 constraint-first protocol."""

    protocol: str = PROTOCOL_NAME
    heading_gain: float = 0.75
    max_heading_step_deg: float = 2.0
    trunk_gain: float = 0.75
    max_trunk_step_deg: float = 6.0
    finite_diff_deg: float = 0.1
    control_tolerance_deg: float = 1e-4
    spine_budget_factor: float = 1.25

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None
    ) -> "TrunkStabilizedRootForwardConfig":
        values = dict(values or {})
        values.setdefault("protocol", PROTOCOL_NAME)
        return super().from_mapping(values)  # type: ignore[return-value]

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.protocol != PROTOCOL_NAME:
            raise ValueError(f"protocol must be {PROTOCOL_NAME}")
        if self.heading_gain <= 0 or self.trunk_gain <= 0:
            raise ValueError("heading_gain and trunk_gain must be positive")
        if self.max_heading_step_deg <= 0 or self.max_trunk_step_deg <= 0:
            raise ValueError("heading/trunk step limits must be positive")
        if self.finite_diff_deg <= 0 or self.control_tolerance_deg < 0:
            raise ValueError("finite_diff_deg must be positive and tolerance non-negative")
        if self.spine_budget_factor <= 0:
            raise ValueError("spine_budget_factor must be positive")


class TrunkStabilizedRootForwardGuidance(RelativeRootForwardGuidance):
    """Constraint-first root pitch/heading and world-trunk stabilization."""

    PROTOCOL = PROTOCOL_NAME

    def __init__(self, *args, config=None, **kwargs) -> None:
        if config is None:
            config = TrunkStabilizedRootForwardConfig()
        elif not isinstance(config, TrunkStabilizedRootForwardConfig):
            config = TrunkStabilizedRootForwardConfig.from_mapping(asdict(config))
        super().__init__(*args, config=config, **kwargs)
        self.baseline_trunk = _trunk_direction(self.baseline_physical, self.valid_mask).detach()

    @classmethod
    def _from_cached(cls, source: "TrunkStabilizedRootForwardGuidance", index: int):
        obj = super()._from_cached(source, index)
        obj.baseline_trunk = source.baseline_trunk[index:index + 1]
        return obj

    def protocol_record(self) -> dict[str, Any]:
        record = super().protocol_record()
        record.update(
            {
                "protocol": PROTOCOL_NAME,
                "guidance_degree_of_freedom": "root_pitch_heading_plus_spine1_spine2_spine3_sagittal_compensation",
                "guided_channels": ["root_rotation", "body_pose.spine1", "body_pose.spine2", "body_pose.spine3"],
                "loss_framework": "independent_control_constraints_and_change_budgets",
                "control_constraints": ["pitch", "full_forward", "heading", "trunk_direction"],
                "config": asdict(self.config),
            }
        )
        return record

    def _direct_with_root(
        self,
        x0_physical: torch.Tensor,
        pitch_deg: torch.Tensor,
        heading_deg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source, pose_mask = _prefix_source(x0_physical, self.valid_mask.to(x0_physical.device))
        body, root, translation = _direct_streams(source)
        r0 = self.targets.r0.to(device=root.device, dtype=root.dtype)
        up = UP.to(device=root.device, dtype=root.dtype).expand_as(r0)
        root = _apply_world_axis(root, r0, pitch_deg)
        root = _apply_world_axis(root, up, heading_deg)
        return body, root, translation, pose_mask, source

    def _apply_spine_local(
        self,
        body: torch.Tensor,
        root: torch.Tensor,
        translation: torch.Tensor,
        beta_deg: torch.Tensor,
    ) -> torch.Tensor:
        """Apply world-frozen right-axis edits as local spine rotations."""

        body = body.clone()
        r0 = self.targets.r0.to(device=body.device, dtype=body.dtype)
        for column, (body_index, joint_index) in enumerate(zip(SPINE_BODY_INDICES, SPINE_JOINT_INDICES)):
            fk = differentiable_forward_kinematics(
                body, root, translation, skeleton=self.skeleton
            )
            parent = int(self.skeleton.parents[joint_index])
            parent_world = fk.global_rotations[..., parent, :, :]
            axis_parent = (parent_world.transpose(-1, -2) @ r0.unsqueeze(-1)).squeeze(-1)
            local_delta = axis_parent * (beta_deg[..., column].unsqueeze(-1) * math.pi / 180.0)
            body[..., body_index, :, :] = axis_angle_to_mat3x3(local_delta) @ body[..., body_index, :, :]
        return body

    def _pack(
        self,
        body: torch.Tensor,
        root: torch.Tensor,
        translation: torch.Tensor,
        pose_mask: torch.Tensor,
        beta_deg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if beta_deg is not None:
            body = self._apply_spine_local(body, root, translation, beta_deg)
        return _authority_from_streams(body, root, translation, pose_mask, skeleton=self.skeleton)

    def _build_candidate(
        self,
        x0_physical: torch.Tensor,
        pitch_deg: torch.Tensor,
        heading_deg: torch.Tensor,
        beta_deg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        body, root, translation, pose_mask, _ = self._direct_with_root(
            x0_physical, pitch_deg, heading_deg
        )
        return self._pack(body, root, translation, pose_mask, beta_deg)

    def _metrics(self, physical: torch.Tensor) -> dict[str, torch.Tensor]:
        source, _ = _prefix_source(physical, self.valid_mask.to(physical.device))
        root = decode_rot6d_safe(source[..., MOTION_LAYOUT.root_rotation])
        f, h, _, phi = _root_forward(root)
        target_f = self.targets.target_forward.to(device=f.device, dtype=f.dtype)
        target_h = self.targets.h0.to(device=f.device, dtype=f.dtype)
        pitch_error = (self.targets.target_phi_deg.to(phi.device, phi.dtype) - phi)
        forward_cross = torch.linalg.vector_norm(torch.cross(f, target_f, dim=-1), dim=-1)
        forward_error = torch.atan2(
            forward_cross, (f * target_f).sum(-1).clamp(-1.0, 1.0)
        ) * 180.0 / math.pi
        heading_error = torch.atan2(
            torch.linalg.vector_norm(torch.cross(h, target_h, dim=-1), dim=-1),
            (h * target_h).sum(-1).clamp(-1.0, 1.0),
        ) * 180.0 / math.pi
        trunk = _trunk_direction(physical, self.valid_mask)
        trunk_target = self.baseline_trunk.to(device=trunk.device, dtype=trunk.dtype)
        trunk_error = torch.atan2(
            torch.linalg.vector_norm(torch.cross(trunk, trunk_target, dim=-1), dim=-1),
            (trunk * trunk_target).sum(-1).clamp(-1.0, 1.0),
        ) * 180.0 / math.pi
        return {
            "pitch": pitch_error,
            "forward": forward_error,
            "heading": heading_error,
            "trunk": trunk_error,
        }

    def _rms(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = value[mask]
        if selected.numel() == 0 or not torch.isfinite(selected).all():
            return torch.as_tensor(float("inf"), device=value.device)
        return torch.sqrt(selected.square().mean())

    def _control_constraints_ok(
        self,
        candidate_metrics: dict[str, torch.Tensor],
        reference_metrics: dict[str, torch.Tensor],
        frame_mask: torch.Tensor,
    ) -> tuple[bool, bool]:
        """Check the four v1.2 controls independently.

        The controls are all angular quantities, but they are deliberately
        never combined into a weighted scalar.  A candidate must be
        non-increasing for pitch, full-forward, heading, and trunk direction;
        it must also improve at least one of them.  This helper is used on the
        final composed candidate, after root and spine edits have both been
        applied.
        """

        names = ("pitch", "forward", "heading", "trunk")
        before = {name: self._rms(reference_metrics[name], frame_mask) for name in names}
        after = {name: self._rms(candidate_metrics[name], frame_mask) for name in names}
        finite = all(torch.isfinite(after[name]) and torch.isfinite(before[name]) for name in names)
        if not finite:
            return False, False
        non_increasing = all(
            after[name] <= before[name] + self.config.control_tolerance_deg
            for name in names
        )
        progress = any(
            after[name] < before[name] - self.config.control_tolerance_deg
            for name in names
        )
        return bool(non_increasing), bool(progress)

    def _spine_sensitivity(
        self,
        body: torch.Tensor,
        root: torch.Tensor,
        translation: torch.Tensor,
        pose_mask: torch.Tensor,
        frame_mask: torch.Tensor,
        trunk_current: torch.Tensor,
    ) -> torch.Tensor:
        """Central-difference derivative of sagittal trunk angle per spine."""

        eps = self.config.finite_diff_deg
        sensitivities = []
        h0 = self.targets.h0.to(device=body.device, dtype=body.dtype)
        up = UP.to(device=body.device, dtype=body.dtype).expand_as(h0)

        def tau(value: torch.Tensor) -> torch.Tensor:
            source, _ = _prefix_source(value, frame_mask)
            joints = source[..., MOTION_LAYOUT.joints].reshape(*source.shape[:-1], 22, 3)
            vector = joints[..., SMPLX_22_JOINT_INDEX["neck"], :] - joints[..., SMPLX_22_JOINT_INDEX["spine1"], :]
            return torch.atan2((vector * h0).sum(-1), (vector * up).sum(-1)) * 180.0 / math.pi

        for column in range(3):
            plus_beta = torch.zeros((*frame_mask.shape, 3), dtype=body.dtype, device=body.device)
            minus_beta = plus_beta.clone()
            plus_beta[..., column] = eps
            minus_beta[..., column] = -eps
            plus_body = self._apply_spine_local(body, root, translation, plus_beta)
            minus_body = self._apply_spine_local(body, root, translation, minus_beta)
            plus_motion = _authority_from_streams(plus_body, root, translation, pose_mask, skeleton=self.skeleton)
            minus_motion = _authority_from_streams(minus_body, root, translation, pose_mask, skeleton=self.skeleton)
            sensitivities.append((tau(plus_motion) - tau(minus_motion)) / (2.0 * eps))
        result = torch.stack(sensitivities, dim=-1)
        return result * frame_mask.unsqueeze(-1).to(result.dtype)

    def _make_spine_proposal(
        self,
        body: torch.Tensor,
        root: torch.Tensor,
        translation: torch.Tensor,
        pose_mask: torch.Tensor,
        frame_mask: torch.Tensor,
        trunk_current: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        baseline = self.baseline_trunk.to(device=body.device, dtype=body.dtype)
        r0 = self.targets.r0.to(device=body.device, dtype=body.dtype)
        residual, valid = signed_axis_residual_deg(
            trunk_current,
            baseline,
            r0,
            frame_mask,
            eps=self.config.eps,
        )
        if not bool(valid[frame_mask].all()):
            return torch.zeros((*frame_mask.shape, 3), device=body.device, dtype=body.dtype), {
                "proposal_rejected": True,
                "rejected_reason": "degenerate_trunk_projection",
                "invalid_trunk_frames": int((frame_mask & ~valid).sum().detach().cpu()),
            }
        sensitivity = self._spine_sensitivity(
            body, root, translation, pose_mask, frame_mask, trunk_current
        )
        denom = sensitivity.square().sum(-1)
        valid = valid & torch.isfinite(sensitivity).all(-1) & (denom > self.config.eps)
        if not bool(valid[frame_mask].all()):
            return torch.zeros((*frame_mask.shape, 3), device=body.device, dtype=body.dtype), {
                "proposal_rejected": True,
                "rejected_reason": "degenerate_spine_sensitivity",
                "invalid_sensitivity_frames": int((frame_mask & ~valid).sum().detach().cpu()),
            }
        # Linearizing the trunk residual as ``residual + S @ beta`` gives the
        # minimum-norm correction ``beta = - residual * S / ||S||²``.  The
        # minus sign is essential: ``residual`` is the signed rotation from
        # the current trunk direction to the frozen M0 direction, whereas
        # ``sensitivity`` is the change in trunk angle produced by a positive
        # local spine rotation.
        beta = -self.config.trunk_gain * residual.unsqueeze(-1) * sensitivity / denom.unsqueeze(-1).clamp_min(self.config.eps)
        beta = beta * frame_mask.unsqueeze(-1).to(beta.dtype)
        total = beta.abs().sum(-1)
        cap = min(self.config.max_trunk_step_deg, self.config.spine_budget_factor * abs(self.target_delta_deg))
        scale = torch.minimum(torch.ones_like(total), torch.as_tensor(cap, device=total.device) / total.clamp_min(self.config.eps))
        beta = beta * scale.unsqueeze(-1)
        selected = beta[frame_mask]
        return beta, {
            "trunk_residual_mean_deg": float(residual[frame_mask].mean().detach().cpu()),
            "trunk_residual_rms_deg": float(torch.sqrt(residual[frame_mask].square().mean()).detach().cpu()),
            "trunk_residual_p95_deg": float(torch.quantile(residual[frame_mask].abs(), 0.95).detach().cpu()),
            "spine_proposal_mean_deg": float(selected.mean().detach().cpu()),
            "spine_proposal_rms_deg": float(torch.sqrt(selected.square().mean()).detach().cpu()),
            "spine_proposal_max_deg": float(selected.abs().max().detach().cpu()),
            "spine_proposal_total_max_deg": float(total[frame_mask].max().detach().cpu()),
            "spine_sensitivity_min": float(torch.linalg.vector_norm(sensitivity[frame_mask], dim=-1).min().detach().cpu()),
        }

    def _proposal(self, base_phys: torch.Tensor, frame_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        source, _ = _prefix_source(base_phys, frame_mask)
        root = decode_rot6d_safe(source[..., MOTION_LAYOUT.root_rotation])
        f, h, _, _ = _root_forward(root)
        target_f = self.targets.target_forward.to(device=f.device, dtype=f.dtype)
        target_h = self.targets.h0.to(device=f.device, dtype=f.dtype)
        pitch, pitch_valid = signed_axis_residual_deg(
            f, target_f, self.targets.r0.to(f), frame_mask, eps=self.config.eps
        )
        up = UP.to(device=f.device, dtype=f.dtype).expand_as(f)
        heading, heading_valid = signed_axis_residual_deg(h, target_h, up, frame_mask, eps=self.config.eps)
        if not bool((pitch_valid & heading_valid)[frame_mask].all()):
            raise ValueError("degenerate root pitch or heading projection")
        pitch_proposal = (self.config.residual_gain * pitch).clamp(-self.config.max_step_deg, self.config.max_step_deg)
        heading_proposal = (self.config.heading_gain * heading).clamp(-self.config.max_heading_step_deg, self.config.max_heading_step_deg)
        body, root_after, translation, pose_mask, _ = self._direct_with_root(
            base_phys, pitch_proposal, heading_proposal
        )
        root_motion = self._pack(body, root_after, translation, pose_mask)
        trunk_current = _trunk_direction(root_motion, frame_mask)
        beta, diagnostics = self._make_spine_proposal(
            body, root_after, translation, pose_mask, frame_mask, trunk_current
        )
        diagnostics.update(
            {
                "pitch_residual_mean_deg": float(pitch[frame_mask].mean().detach().cpu()),
                "pitch_residual_rms_deg": float(torch.sqrt(pitch[frame_mask].square().mean()).detach().cpu()),
                "pitch_residual_p95_deg": float(torch.quantile(pitch[frame_mask].abs(), 0.95).detach().cpu()),
                "pitch_proposal_rms_deg": float(torch.sqrt(pitch_proposal[frame_mask].square().mean()).detach().cpu()),
                "heading_residual_mean_deg": float(heading[frame_mask].mean().detach().cpu()),
                "heading_residual_rms_deg": float(torch.sqrt(heading[frame_mask].square().mean()).detach().cpu()),
                "heading_residual_p95_deg": float(torch.quantile(heading[frame_mask].abs(), 0.95).detach().cpu()),
                "heading_proposal_rms_deg": float(torch.sqrt(heading_proposal[frame_mask].square().mean()).detach().cpu()),
            }
        )
        return pitch_proposal, heading_proposal, beta, diagnostics

    def _tail_budget_ok(self, before: torch.Tensor, after: torch.Tensor, mask: torch.Tensor) -> bool:
        source_before, _ = _prefix_source(before, mask)
        source_after, _ = _prefix_source(after, mask)
        root_before = decode_rot6d_safe(source_before[..., MOTION_LAYOUT.root_rotation])
        root_after = decode_rot6d_safe(source_after[..., MOTION_LAYOUT.root_rotation])
        step_before = _geodesic(root_before[:, 1:], root_before[:, :-1]) * 180.0 / math.pi
        step_after = _geodesic(root_after[:, 1:], root_after[:, :-1]) * 180.0 / math.pi
        pitch_before = _root_forward(root_before)[3]
        pitch_after = _root_forward(root_after)[3]
        delta_so3 = (step_after - step_before).abs()
        delta_pitch = ((pitch_after[:, 1:] - pitch_after[:, :-1]) - (pitch_before[:, 1:] - pitch_before[:, :-1])).abs()
        for index in range(mask.shape[0]):
            pairs = mask[index, 1:] & mask[index, :-1]
            if not bool(pairs.any()):
                return False
            all_so3 = delta_so3[index][pairs]
            all_pitch = delta_pitch[index][pairs]
            positions = torch.nonzero(pairs, as_tuple=False).flatten()[-8:]
            if not bool(torch.isfinite(all_so3).all() and torch.isfinite(all_pitch).all()):
                return False
            if float(all_so3.max()) > 2.0 or float(all_pitch.max()) > 2.0:
                return False
            if positions.numel() and (float(delta_so3[index, positions].max()) > 2.0 or float(delta_pitch[index, positions].max()) > 2.0):
                return False
        return True

    def correct_velocity(self, *, x_sigma, velocity, sigma, valid_mask, return_trace=False):
        if x_sigma.shape != velocity.shape or x_sigma.ndim != 3 or x_sigma.shape[-1] != 276:
            raise ValueError("x_sigma and velocity must have shape [B,T,276]")
        if valid_mask.shape != x_sigma.shape[:2] or valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must be bool[B,T]")
        if not torch.equal(valid_mask, self.valid_mask.to(valid_mask.device)):
            raise ValueError("sampler valid_mask differs from the frozen guidance mask")
        sigma_value = float(torch.as_tensor(sigma).detach().cpu().item())
        cfg = self.config
        diagnostics: dict[str, Any] = {
            "protocol": self.PROTOCOL,
            "enabled": cfg.enabled,
            "active": False,
            "sigma": sigma_value,
            "target_delta_deg": self.target_delta_deg,
            "accepted": False,
        }
        x0_hat = predict_x0(x_sigma.float(), velocity.float(), sigma_value).detach()
        inactive = (
            not math.isfinite(sigma_value)
            or not cfg.enabled
            or cfg.guidance_strength <= cfg.eps
            or abs(self.target_delta_deg) <= cfg.eps
            or sigma_value <= cfg.eps
            or sigma_value < cfg.sigma_min
            or sigma_value > cfg.sigma_max
        )
        if inactive:
            if return_trace:
                diagnostics["trace"] = {
                    "velocity_model": velocity.detach().float().clone(),
                    "v_cfg": velocity.detach().float().clone(),
                    "x0_hat": x0_hat.clone(),
                    "x0_guided": x0_hat.clone(),
                    "x0_reconciled": x0_hat.clone(),
                }
            self.last_diagnostics = diagnostics
            return velocity, diagnostics
        if not torch.isfinite(x0_hat).all():
            diagnostics["rejected_reason"] = "non_finite_x0_hat"
            self.last_diagnostics = diagnostics
            return velocity, diagnostics

        with torch.amp.autocast(device_type=x_sigma.device.type, enabled=False):
            x0_physical = self._physical(x0_hat)
            if not torch.isfinite(x0_physical).all():
                diagnostics["rejected_reason"] = "non_finite_physical_endpoint"
                self.last_diagnostics = diagnostics
                return velocity, diagnostics
            frame_mask = valid_mask
            base_phys = self._build_candidate(
                x0_physical,
                torch.zeros_like(frame_mask, dtype=torch.float32),
                torch.zeros_like(frame_mask, dtype=torch.float32),
            )
            base_norm = self._standardize(base_phys)
            base_metrics = self._metrics(base_phys)
            try:
                pitch, heading, beta, proposal_diag = self._proposal(base_phys, frame_mask)
            except ValueError as exc:
                diagnostics["rejected_reason"] = str(exc)
                self.last_diagnostics = diagnostics
                return velocity, diagnostics
            if proposal_diag.get("proposal_rejected", False):
                diagnostics.update(proposal_diag)
                diagnostics["rejected_reason"] = proposal_diag["rejected_reason"]
                self.last_diagnostics = diagnostics
                return velocity, diagnostics
            # Root control is accepted first.  Its 276D trust region must not
            # be consumed by the optional spine compensation; otherwise the
            # redundant high-frequency velocity channels can shrink a valid
            # root correction to an ineffective fraction.
            root_zero = torch.zeros_like(pitch)
            # Apply pitch and heading in two independent constraint searches.
            # A single combined test can reject a valid pitch edit because a
            # small yaw component transiently worsens the full-vector metric.
            root_rms_probe = self._standardize(
                self._build_candidate(x0_physical, pitch, root_zero, root_zero)
            )
            root_rms_probe = torch.sqrt(
                ((root_rms_probe - base_norm).square() * frame_mask.unsqueeze(-1)).sum()
                / (frame_mask.sum().clamp_min(1) * 276)
            )
            if float(root_rms_probe) > cfg.max_correction_rms:
                pitch = pitch * (cfg.max_correction_rms / root_rms_probe.clamp_min(cfg.eps))
            root_accepted = None
            pitch_only_accepted = None
            for root_attempt in range(cfg.max_backtracks + 1):
                scale = 0.5 ** root_attempt
                candidate_pitch = pitch * scale
                candidate_phys = self._build_candidate(
                    x0_physical, candidate_pitch, root_zero, root_zero
                )
                candidate_norm = self._standardize(candidate_phys)
                candidate_metrics = self._metrics(candidate_phys)
                finite = torch.isfinite(candidate_norm).all()
                controls_ok = all(
                    self._rms(candidate_metrics[name], frame_mask)
                    <= self._rms(base_metrics[name], frame_mask) + cfg.control_tolerance_deg
                    for name in ("pitch", "forward")
                )
                progress = any(
                    self._rms(candidate_metrics[name], frame_mask)
                    < self._rms(base_metrics[name], frame_mask) - cfg.control_tolerance_deg
                    for name in ("pitch", "forward")
                )
                candidate_rms = torch.sqrt(
                    ((candidate_norm - base_norm).square() * frame_mask.unsqueeze(-1)).sum()
                    / (frame_mask.sum().clamp_min(1) * 276)
                )
                if bool(finite and controls_ok and progress and candidate_rms <= cfg.max_correction_rms + cfg.eps):
                    pitch_only_accepted = (candidate_phys, candidate_norm, candidate_pitch, candidate_metrics, candidate_rms, scale)
                    break
            if pitch_only_accepted is not None:
                pitch_phys, pitch_norm, candidate_pitch, pitch_metrics, pitch_rms, pitch_scale = pitch_only_accepted
                # Recompute the yaw residual after the accepted pitch edit.
                # A pitch rotation about the frozen sagittal axis is exact
                # only when the current endpoint has zero heading error.  In
                # generated endpoints the two errors are coupled, so using a
                # heading residual measured before the pitch edit can leave a
                # systematic horizontal drift.  The heading proposal must be
                # based on the state that is actually being retained.
                pitch_source, _ = _prefix_source(pitch_phys, frame_mask)
                pitch_root = decode_rot6d_safe(pitch_source[..., MOTION_LAYOUT.root_rotation])
                pitch_forward, pitch_heading, _, _ = _root_forward(pitch_root)
                up_pitch = UP.to(device=pitch_heading.device, dtype=pitch_heading.dtype).expand_as(pitch_heading)
                accepted_heading_residual, accepted_heading_valid = signed_axis_residual_deg(
                    pitch_heading,
                    self.targets.h0.to(device=pitch_heading.device, dtype=pitch_heading.dtype),
                    up_pitch,
                    frame_mask,
                    eps=self.config.eps,
                )
                if not bool(accepted_heading_valid[frame_mask].all()):
                    accepted_heading_residual = heading
                heading = accepted_heading_residual
                heading_proposal = (self.config.heading_gain * heading).clamp(
                    -self.config.max_heading_step_deg, self.config.max_heading_step_deg
                )
                proposal_diag.update(
                    {
                        "heading_residual_mean_deg": float(heading[frame_mask].mean().detach().cpu()),
                        "heading_residual_rms_deg": float(torch.sqrt(heading[frame_mask].square().mean()).detach().cpu()),
                        "heading_residual_p95_deg": float(torch.quantile(heading[frame_mask].abs(), 0.95).detach().cpu()),
                        "heading_proposal_rms_deg": float(torch.sqrt(heading_proposal[frame_mask].square().mean()).detach().cpu()),
                    }
                )
                heading_accepted = None
                for heading_attempt in range(cfg.max_backtracks + 1):
                    heading_scale = 0.5 ** heading_attempt
                    candidate_heading = heading * heading_scale
                    candidate_phys = self._build_candidate(
                        x0_physical, candidate_pitch, candidate_heading, root_zero
                    )
                    candidate_norm = self._standardize(candidate_phys)
                    candidate_metrics = self._metrics(candidate_phys)
                    finite = torch.isfinite(candidate_norm).all()
                    controls_ok = all(
                        self._rms(candidate_metrics[name], frame_mask)
                        <= self._rms(base_metrics[name], frame_mask) + cfg.control_tolerance_deg
                        for name in ("pitch", "forward", "heading")
                    )
                    progress = (
                        self._rms(candidate_metrics["heading"], frame_mask)
                        < self._rms(pitch_metrics["heading"], frame_mask) - cfg.control_tolerance_deg
                    )
                    candidate_rms = torch.sqrt(
                        ((candidate_norm - pitch_norm).square() * frame_mask.unsqueeze(-1)).sum()
                        / (frame_mask.sum().clamp_min(1) * 276)
                    )
                    if bool(finite and controls_ok and progress and candidate_rms <= cfg.max_correction_rms + cfg.eps):
                        heading_accepted = (candidate_phys, candidate_norm, candidate_heading, candidate_metrics, candidate_rms, heading_scale)
                        break
                if heading_accepted is None:
                    root_accepted = (pitch_phys, pitch_norm, candidate_pitch, root_zero, pitch_scale, pitch_rms, pitch_metrics)
                else:
                    final_root_phys, final_root_norm, candidate_heading, final_root_metrics, final_rms, heading_scale = heading_accepted
                    root_accepted = (final_root_phys, final_root_norm, candidate_pitch, candidate_heading, pitch_scale, final_rms, final_root_metrics)
            diagnostics.update(proposal_diag)
            if root_accepted is None:
                diagnostics["rejected_reason"] = "no_feasible_constraint_candidate"
                diagnostics.update(
                    {
                        "forward_rms_before_deg": float(self._rms(base_metrics["forward"], frame_mask).detach().cpu()),
                        "heading_rms_before_deg": float(self._rms(base_metrics["heading"], frame_mask).detach().cpu()),
                        "trunk_rms_before_deg": float(self._rms(base_metrics["trunk"], frame_mask).detach().cpu()),
                        "correction_rms_276d": 0.0,
                        "backtrack_count": cfg.max_backtracks,
                    }
                )
                self.last_diagnostics = diagnostics
                return velocity, diagnostics
            root_phys, root_norm, candidate_pitch, candidate_heading, scale, candidate_rms, root_metrics = root_accepted

            # Recompute the spine proposal around the actually accepted root
            # endpoint.  This keeps the local-axis conversion and sensitivity
            # calculation tied to the state that will be written back.
            root_body, root_rotation, root_translation, root_pose_mask, _ = self._direct_with_root(
                x0_physical, candidate_pitch, candidate_heading
            )
            root_trunk = _trunk_direction(root_phys, frame_mask)
            try:
                candidate_beta, spine_diag = self._make_spine_proposal(
                    root_body, root_rotation, root_translation, root_pose_mask, frame_mask, root_trunk
                )
            except ValueError as exc:
                candidate_beta = torch.zeros((*frame_mask.shape, 3), dtype=torch.float32, device=x_sigma.device)
                spine_diag = {"proposal_rejected": True, "rejected_reason": str(exc)}
            diagnostics.update(spine_diag)
            final_phys, final_norm, final_metrics = root_phys, root_norm, root_metrics
            spine_accepted = False
            if not spine_diag.get("proposal_rejected", False):
                last_spine_check: dict[str, Any] = {}
                for spine_attempt in range(cfg.max_backtracks + 1):
                    spine_scale = 0.5 ** spine_attempt
                    pitch_try = candidate_pitch
                    heading_try = candidate_heading
                    beta_try = candidate_beta * spine_scale
                    spine_phys = self._build_candidate(
                        x0_physical, pitch_try, heading_try, beta_try
                    )
                    spine_norm = self._standardize(spine_phys)
                    total_rms = torch.sqrt(
                        ((spine_norm - base_norm).square() * frame_mask.unsqueeze(-1)).sum()
                        / (frame_mask.sum().clamp_min(1) * 276)
                    )
                    # Root and spine edits are proposed sequentially, but the
                    # v1.2 budget applies to their composed endpoint.  If the
                    # composition is just over budget, shrink all three
                    # editable pose components together toward the
                    # authoritative base before evaluating the independent
                    # angular constraints.  This preserves a single total
                    # budget rather than allowing three stage-wise budgets to
                    # accumulate.
                    if bool(torch.isfinite(total_rms) and total_rms > cfg.max_correction_rms + cfg.eps):
                        budget_scale = (
                            cfg.max_correction_rms
                            / total_rms.clamp_min(cfg.eps)
                        ) * 0.995
                        pitch_try = pitch_try * budget_scale
                        heading_try = heading_try * budget_scale
                        beta_try = beta_try * budget_scale
                        spine_phys = self._build_candidate(
                            x0_physical, pitch_try, heading_try, beta_try
                        )
                        spine_norm = self._standardize(spine_phys)
                        total_rms = torch.sqrt(
                            ((spine_norm - base_norm).square() * frame_mask.unsqueeze(-1)).sum()
                            / (frame_mask.sum().clamp_min(1) * 276)
                        )
                    spine_metrics = self._metrics(spine_phys)
                    spine_rms = torch.sqrt(
                        ((spine_norm - root_norm).square() * frame_mask.unsqueeze(-1)).sum()
                        / (frame_mask.sum().clamp_min(1) * 276)
                    )
                    spine_sum = beta_try.abs().sum(-1)[frame_mask]
                    controls_ok, controls_progress = self._control_constraints_ok(
                        spine_metrics, base_metrics, frame_mask
                    )
                    spine_budget_ok = bool(torch.isfinite(total_rms) and total_rms <= cfg.max_correction_rms + cfg.eps)
                    spine_budget_ok = spine_budget_ok and bool(torch.isfinite(spine_sum).all() and spine_sum.max() <= cfg.spine_budget_factor * abs(self.target_delta_deg) + cfg.eps)
                    last_spine_check = {
                        "attempt": int(spine_attempt),
                        "trunk_rms_before": float(self._rms(base_metrics["trunk"], frame_mask).detach().cpu()),
                        "trunk_rms_after": float(self._rms(spine_metrics["trunk"], frame_mask).detach().cpu()),
                        "spine_correction_rms_276d": float(spine_rms.detach().cpu()),
                        "total_correction_rms_276d": float(total_rms.detach().cpu()),
                        "spine_sum_max_deg": float(spine_sum.max().detach().cpu()),
                        "controls_ok": bool(controls_ok),
                        "controls_progress": bool(controls_progress),
                        "spine_budget_ok": bool(spine_budget_ok),
                        "finite": bool(torch.isfinite(spine_norm).all()),
                    }
                    if bool(torch.isfinite(spine_norm).all() and controls_ok and controls_progress and spine_budget_ok):
                        final_phys, final_norm, final_metrics = spine_phys, spine_norm, spine_metrics
                        candidate_beta = beta_try
                        candidate_pitch = pitch_try
                        candidate_heading = heading_try
                        spine_accepted = True
                        diagnostics["spine_backtrack_count"] = int(spine_attempt)
                        diagnostics["spine_correction_rms_276d"] = float(spine_rms.detach().cpu())
                        diagnostics["total_correction_rms_276d"] = float(total_rms.detach().cpu())
                        break
                if not spine_accepted:
                    diagnostics["spine_last_check"] = last_spine_check

            # A root-only endpoint is not a valid v1.2 result when the
            # optional spine compensation could not be composed: the trunk
            # control would otherwise be silently sacrificed.  Keep the
            # unedited authoritative endpoint unless the root candidate itself
            # satisfies all four independent controls and the *total* 276D
            # change budget.
            if not spine_accepted:
                root_controls_ok, root_controls_progress = self._control_constraints_ok(
                    root_metrics, base_metrics, frame_mask
                )
                root_total_rms = torch.sqrt(
                    ((root_norm - base_norm).square() * frame_mask.unsqueeze(-1)).sum()
                    / (frame_mask.sum().clamp_min(1) * 276)
                )
                root_budget_ok = bool(
                    torch.isfinite(root_total_rms)
                    and root_total_rms <= cfg.max_correction_rms + cfg.eps
                )
                diagnostics["root_fallback_total_correction_rms_276d"] = float(
                    root_total_rms.detach().cpu()
                )
                if not (root_controls_ok and root_controls_progress and root_budget_ok):
                    diagnostics.update(
                        {
                            "accepted": False,
                            "rejected_reason": "no_feasible_composed_constraint_candidate",
                            "correction_rms_276d": 0.0,
                            "backtrack_count": cfg.max_backtracks,
                        }
                    )
                    self.last_diagnostics = diagnostics
                    return velocity, diagnostics

            corrected_velocity = velocity_from_x0(x_sigma.float(), final_norm, sigma_value).to(dtype=velocity.dtype)
            corrected_velocity = torch.where(valid_mask.unsqueeze(-1), corrected_velocity, velocity)
            root_corr_rms = torch.sqrt(
                (candidate_pitch[frame_mask].square() + candidate_heading[frame_mask].square()).mean()
            )
            diagnostics.update(
                {
                    "active": True,
                    "accepted": True,
                    "accepted_scale": float(scale),
                    "accepted_root_pitch_rms_deg": float(torch.sqrt(candidate_pitch[frame_mask].square().mean()).detach().cpu()),
                    "accepted_heading_rms_deg": float(torch.sqrt(candidate_heading[frame_mask].square().mean()).detach().cpu()),
                    "accepted_spine_rms_deg": float(torch.sqrt(candidate_beta[frame_mask].square().mean()).detach().cpu()) if spine_accepted else 0.0,
                    "accepted_alpha_rms_deg": float(root_corr_rms.detach().cpu()),
                    "accepted_alpha_max_deg": float(torch.sqrt(candidate_pitch.square() + candidate_heading.square())[frame_mask].max().detach().cpu()),
                    "correction_rms_276d": float(torch.sqrt(((final_norm - base_norm).square() * frame_mask.unsqueeze(-1)).sum() / (frame_mask.sum().clamp_min(1) * 276)).detach().cpu()),
                    "forward_rms_before_deg": float(self._rms(base_metrics["forward"], frame_mask).detach().cpu()),
                    "forward_rms_after_deg": float(self._rms(final_metrics["forward"], frame_mask).detach().cpu()),
                    "heading_rms_before_deg": float(self._rms(base_metrics["heading"], frame_mask).detach().cpu()),
                    "heading_rms_after_deg": float(self._rms(final_metrics["heading"], frame_mask).detach().cpu()),
                    "trunk_rms_before_deg": float(self._rms(base_metrics["trunk"], frame_mask).detach().cpu()),
                    "trunk_rms_after_deg": float(self._rms(final_metrics["trunk"], frame_mask).detach().cpu()),
                    "backtrack_count": int(root_attempt),
                    "valid_frames": int(frame_mask.sum().detach().cpu()),
                }
            )
            x0_guided = final_norm.detach()
            if return_trace:
                diagnostics["trace"] = {
                    "velocity_model": velocity.detach().float().clone(),
                    "v_cfg": velocity.detach().float().clone(),
                    "x0_hat": x0_hat.clone(),
                    "x0_guided": x0_guided.clone(),
                    "x0_reconciled": x0_guided.clone(),
                }
        self.last_diagnostics = diagnostics
        return corrected_velocity, diagnostics

    def finalize_outputs(self, official_norm: torch.Tensor) -> RelativeRootForwardFinalOutputs:
        if abs(self.target_delta_deg) <= self.config.eps:
            return super().finalize_outputs(official_norm)
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
        audits = tuple(
            whole_body_audit(
                self.baseline_physical[i:i + 1],
                projection.physical_motion[i:i + 1],
                self.valid_mask[i:i + 1].to(projection.physical_motion.device),
            )
            for i in range(projection.physical_motion.shape[0])
        )
        from evaluation.relative_root_forward_v1 import compute_relative_root_forward_metrics
        metrics = compute_relative_root_forward_metrics(
            self.baseline_physical,
            projection.physical_motion,
            self.valid_mask.to(projection.physical_motion.device),
            self.target_delta_deg,
            skeleton=self.skeleton,
            protocol_name=self.PROTOCOL,
        )
        return RelativeRootForwardFinalOutputs(
            projection.motion,
            projection.valid_mask,
            projection.audits,
            audits,
            protocol=self.PROTOCOL,
            metrics=metrics,
        )


__all__ = [
    "PROTOCOL_NAME",
    "TrunkStabilizedRootForwardConfig",
    "TrunkStabilizedRootForwardGuidance",
    "signed_axis_residual_deg",
]
