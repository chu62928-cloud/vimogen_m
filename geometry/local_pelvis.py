"""Differentiable S1 pelvis--thorax relative-angle geometry.

The historical C0 evaluator measures the world orientation of the root.  S1
instead expresses the pelvis forward axis in the global ``spine3`` frame, so
premultiplying the whole body by the same rigid rotation leaves the control
quantity unchanged.
"""

from __future__ import annotations

import math

import torch

from geometry.authoritative_motion import authoritative_motion
from motion_rep.consistency_v2 import differentiable_forward_kinematics
from motion_rep.phase1 import (
    MOTION_LAYOUT,
    SMPLX_22_JOINT_INDEX,
    decode_rot6d_safe,
    encode_rot6d,
)
from motion_rep.sagittal_pelvis_angle import (
    apply_person_right_axis_rotation,
    pelvis_sagittal_tilt_degrees,
)


EVALUATOR_VERSION = "local_pelvis_spine3_relative_sagittal_v1"
S1_CONSTRAINT_PACK = "S1_RELATIVE"
S2_CONSTRAINT_PACK = "S2_RELATIVE_WORLD"
_EPS = 1.0e-8


def wrap_angle_deg(value: torch.Tensor) -> torch.Tensor:
    return torch.remainder(value + 180.0, 360.0) - 180.0


def _direct_rotations(motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if motion.ndim != 3 or motion.shape[-1] != MOTION_LAYOUT.total_dim:
        raise ValueError("motion must have shape [B,T,276]")
    body = decode_rot6d_safe(
        motion[..., MOTION_LAYOUT.body_pose].reshape(
            motion.shape[0], motion.shape[1], 21, 6
        )
    )
    root = decode_rot6d_safe(motion[..., MOTION_LAYOUT.root_rotation])
    return body, root


def relative_pelvis_angle_curve_deg(motion: torch.Tensor) -> torch.Tensor:
    """Return anterior-down pelvis pitch relative to global ``spine3``.

    SMPL-X body-local ``+z`` is anterior, ``-y`` is superior and ``+x`` is
    anatomical right in the calibrated project convention.  The leading
    minus sign makes anterior-down positive.
    """

    body, root = _direct_rotations(motion)
    translation = motion[..., MOTION_LAYOUT.root_translation]
    fk = differentiable_forward_kinematics(body, root, translation)
    thorax = fk.global_rotations[..., SMPLX_22_JOINT_INDEX["spine3"], :, :]
    pelvis_in_thorax = thorax.transpose(-1, -2) @ root
    anterior = torch.zeros(3, dtype=motion.dtype, device=motion.device)
    anterior[2] = 1.0
    superior = torch.zeros(3, dtype=motion.dtype, device=motion.device)
    superior[1] = -1.0
    forward = pelvis_in_thorax @ anterior
    sine_up = (forward * superior).sum(dim=-1)
    cosine_forward = (forward * anterior).sum(dim=-1)
    projection_norm = torch.sqrt(sine_up.square() + cosine_forward.square())
    if torch.any(projection_norm <= _EPS):
        raise ValueError("pelvis anterior axis is degenerate in the thorax sagittal plane")
    return wrap_angle_deg(
        -torch.atan2(sine_up, cosine_forward) * (180.0 / math.pi)
    )


def target_relative_angle_curve_deg(
    baseline_motion: torch.Tensor,
    target_dose_deg: float | torch.Tensor,
) -> torch.Tensor:
    baseline = relative_pelvis_angle_curve_deg(baseline_motion)
    dose = torch.as_tensor(
        target_dose_deg, dtype=baseline.dtype, device=baseline.device
    )
    try:
        dose = torch.broadcast_to(dose, baseline.shape)
    except RuntimeError as error:
        raise ValueError("target dose is not broadcastable to [B,T]") from error
    return wrap_angle_deg(baseline + dose)


def relative_angle_error_deg(
    motion: torch.Tensor,
    target_curve_deg: torch.Tensor,
) -> torch.Tensor:
    actual = relative_pelvis_angle_curve_deg(motion)
    if target_curve_deg.shape != actual.shape:
        raise ValueError("target curve must match motion [B,T]")
    return wrap_angle_deg(actual - target_curve_deg)


def world_pelvis_angle_curve_deg(motion: torch.Tensor) -> torch.Tensor:
    """Return corrected anterior-down world pelvis pitch."""

    _, root = _direct_rotations(motion)
    return wrap_angle_deg(-pelvis_sagittal_tilt_degrees(root))


def target_world_pelvis_angle_curve_deg(
    baseline_motion: torch.Tensor,
    target_dose_deg: float | torch.Tensor,
) -> torch.Tensor:
    baseline = world_pelvis_angle_curve_deg(baseline_motion)
    dose = torch.as_tensor(
        target_dose_deg, dtype=baseline.dtype, device=baseline.device
    )
    try:
        dose = torch.broadcast_to(dose, baseline.shape)
    except RuntimeError as error:
        raise ValueError("target dose is not broadcastable to [B,T]") from error
    return wrap_angle_deg(baseline + dose)


def world_pelvis_angle_error_deg(
    motion: torch.Tensor, target_curve_deg: torch.Tensor
) -> torch.Tensor:
    actual = world_pelvis_angle_curve_deg(motion)
    if target_curve_deg.shape != actual.shape:
        raise ValueError("world pelvis target curve must match motion [B,T]")
    return wrap_angle_deg(actual - target_curve_deg)


def direct_channel_mask_like(motion: torch.Tensor) -> torch.Tensor:
    """Mask the authoritative pose/root/translation variables only."""

    mask = torch.zeros_like(motion)
    mask[..., MOTION_LAYOUT.body_pose] = 1.0
    mask[..., MOTION_LAYOUT.root_rotation] = 1.0
    mask[..., MOTION_LAYOUT.root_translation] = 1.0
    return mask


def minimum_norm_relative_projection(
    motion: torch.Tensor,
    target_curve_deg: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    damping: float = 1.0e-5,
    max_step_deg: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Take one damped scalar-Jacobian projection step per valid frame.

    All authoritative direct variables are exposed, while variables with zero
    derivative receive no update.  This preserves the planned S1 semantics:
    no trunk/contact/trajectory term is silently introduced.
    """

    if damping < 0.0 or max_step_deg <= 0.0:
        raise ValueError("damping must be nonnegative and max_step_deg positive")
    if valid_mask.dtype is not torch.bool or valid_mask.shape != motion.shape[:2]:
        raise ValueError("valid_mask must be bool[B,T]")
    target = target_curve_deg.to(device=motion.device, dtype=motion.dtype)
    with torch.enable_grad():
        variable = motion.detach().float().requires_grad_(True)
        actual = relative_pelvis_angle_curve_deg(variable)
        residual = wrap_angle_deg(actual - target.float())
        selected = actual.masked_fill(~valid_mask, 0.0).sum()
        jacobian = torch.autograd.grad(selected, variable)[0]
        jacobian = jacobian * direct_channel_mask_like(variable)
        jacobian = jacobian * valid_mask.unsqueeze(-1)
        denominator = jacobian.square().sum(dim=-1) + float(damping)
        requested = (-residual).clamp(-max_step_deg, max_step_deg)
        requested = torch.where(valid_mask, requested, torch.zeros_like(requested))
        delta = requested.unsqueeze(-1) * jacobian / denominator.unsqueeze(-1)
        edited = variable + delta
    projected = authoritative_motion(
        edited.detach(), valid_mask=valid_mask
    ).motion
    realized = wrap_angle_deg(
        relative_pelvis_angle_curve_deg(projected).detach() - actual.detach()
    )
    active_norm = denominator.detach()[valid_mask]
    return projected, {
        "solver_iterations": 1,
        "jacobian_rank": int((active_norm > float(damping) + _EPS).sum().item()),
        "requested_step_rms_deg": float(
            torch.sqrt(requested.detach()[valid_mask].square().mean()).cpu()
        ),
        "realized_step_rms_deg": float(
            torch.sqrt(realized[valid_mask].square().mean()).cpu()
        ),
        "residual_mae_deg": float(residual.detach()[valid_mask].abs().mean().cpu()),
        "direct_update_rms": float(
            torch.sqrt(delta.detach()[valid_mask].square().mean()).cpu()
        ),
    }


def minimum_norm_s2_projection(
    motion: torch.Tensor,
    relative_target_curve_deg: torch.Tensor,
    world_target_curve_deg: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    damping: float = 1.0e-5,
    max_step_deg: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Project relative angle, then independently correct world pelvis pitch."""

    relative_projected, relative_audit = minimum_norm_relative_projection(
        motion,
        relative_target_curve_deg,
        valid_mask,
        damping=damping,
        max_step_deg=max_step_deg,
    )
    world_residual = world_pelvis_angle_error_deg(
        relative_projected, world_target_curve_deg.to(relative_projected.device)
    )
    corrected_step = (-world_residual).clamp(-max_step_deg, max_step_deg)
    corrected_step = torch.where(
        valid_mask, corrected_step, torch.zeros_like(corrected_step)
    )
    _, root = _direct_rotations(relative_projected)
    edited = relative_projected.clone()
    # The historical right-axis operator increases anterior-up pitch; S2 is
    # anterior-down positive, hence the negative operator angle.
    edited[..., MOTION_LAYOUT.root_rotation] = encode_rot6d(
        apply_person_right_axis_rotation(root, -corrected_step)
    )
    projected = authoritative_motion(edited, valid_mask=valid_mask).motion
    world_realized = wrap_angle_deg(
        world_pelvis_angle_curve_deg(projected)
        - world_pelvis_angle_curve_deg(relative_projected)
    )
    return projected, {
        **relative_audit,
        "world_requested_step_rms_deg": float(
            torch.sqrt(corrected_step[valid_mask].square().mean()).cpu()
        ),
        "world_realized_step_rms_deg": float(
            torch.sqrt(world_realized[valid_mask].square().mean()).cpu()
        ),
        "world_residual_mae_deg": float(
            world_residual[valid_mask].abs().mean().cpu()
        ),
    }


__all__ = [
    "EVALUATOR_VERSION",
    "S1_CONSTRAINT_PACK",
    "S2_CONSTRAINT_PACK",
    "direct_channel_mask_like",
    "minimum_norm_relative_projection",
    "minimum_norm_s2_projection",
    "relative_angle_error_deg",
    "relative_pelvis_angle_curve_deg",
    "target_relative_angle_curve_deg",
    "target_world_pelvis_angle_curve_deg",
    "world_pelvis_angle_curve_deg",
    "world_pelvis_angle_error_deg",
    "wrap_angle_deg",
]
