"""Pelvis angle helpers.

The original ``pelvis_pitch_degrees`` proxy is retained for v1 result
compatibility.  v2 adds a geometry-based local-sagittal definition that does
not use a per-frame world/person heading.  It derives the sagittal plane from
the hip axis and gravity, so a pure horizontal yaw rotates both the plane and
the spine vector together and leaves the signed angle unchanged.
"""

from __future__ import annotations

import math

import torch


def heading_from_velocity(
    velocity: torch.Tensor,
    *,
    min_speed: float = 0.05,
    up_axis: int = 2,
    carry_forward: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if velocity.ndim < 2 or velocity.shape[-1] != 3:
        raise ValueError(f"velocity must have shape [..., T, 3], got {tuple(velocity.shape)}")
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be 0, 1, or 2")
    if min_speed < 0:
        raise ValueError("min_speed must be non-negative")
    horizontal = velocity.clone()
    horizontal[..., up_axis] = 0
    speed = torch.linalg.vector_norm(horizontal, dim=-1)
    valid = speed >= min_speed
    direction = horizontal / speed.unsqueeze(-1).clamp_min(torch.finfo(horizontal.dtype).eps)
    if not carry_forward:
        return direction, valid
    flat_direction = direction.reshape(-1, direction.shape[-2], 3).clone()
    flat_valid = valid.reshape(-1, valid.shape[-1])
    canonical = torch.zeros(3, dtype=velocity.dtype, device=velocity.device)
    canonical[1] = 1.0
    for sequence in range(flat_direction.shape[0]):
        previous = canonical
        for frame in range(flat_direction.shape[1]):
            if bool(flat_valid[sequence, frame]):
                previous = flat_direction[sequence, frame]
            else:
                flat_direction[sequence, frame] = previous
    return flat_direction.reshape_as(direction), valid


def pelvis_pitch_degrees(
    root_rotation: torch.Tensor,
    heading: torch.Tensor,
    *,
    local_forward_axis: int = 2,
    up_axis: int = 2,
) -> torch.Tensor:
    """Legacy heading-relative model-space pitch proxy (v1)."""

    if root_rotation.shape[-2:] != (3, 3):
        raise ValueError(f"root_rotation must end in [3,3], got {tuple(root_rotation.shape)}")
    if heading.shape != root_rotation.shape[:-2] + (3,):
        raise ValueError(
            f"heading must have shape {root_rotation.shape[:-2] + (3,)}, got {tuple(heading.shape)}"
        )
    if local_forward_axis not in (0, 1, 2) or up_axis not in (0, 1, 2):
        raise ValueError("axis values must be 0, 1, or 2")
    heading_norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
    if torch.any(heading_norm <= torch.finfo(heading.dtype).eps):
        raise ValueError("heading contains a zero vector")
    heading = heading / heading_norm
    local_forward = torch.zeros(3, dtype=root_rotation.dtype, device=root_rotation.device)
    local_forward[local_forward_axis] = 1.0
    forward = root_rotation @ local_forward
    forward_up = forward[..., up_axis]
    up = torch.zeros(3, dtype=root_rotation.dtype, device=root_rotation.device)
    up[up_axis] = 1.0
    right = torch.cross(heading, up.expand_as(heading), dim=-1)
    along_heading = (forward * heading).sum(-1)
    lateral = (forward * right).sum(-1)
    horizontal_norm = torch.sqrt(along_heading.square() + lateral.square())
    return torch.atan2(forward_up, horizontal_norm.clamp_min(torch.finfo(forward.dtype).eps)) * (180.0 / math.pi)


def _axis_unit(axis: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    result = torch.zeros(3, dtype=dtype, device=device)
    result[axis] = 1.0
    return result


def _stable_unit(value: torch.Tensor, fallback: torch.Tensor, eps: float) -> torch.Tensor:
    """Finite unit vector with a deterministic, differentiable-safe fallback."""

    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    normalized = value / norm.clamp_min(eps)
    fallback = fallback.expand_as(normalized)
    return torch.where(norm > eps, normalized, fallback)


def pelvis_pitch_degrees_v2(
    joints: torch.Tensor,
    *,
    rest_offsets: torch.Tensor | None = None,
    pelvis_index: int = 0,
    left_hip_index: int = 1,
    right_hip_index: int = 2,
    spine_index: int = 3,
    left_shoulder_index: int = 16,
    right_shoulder_index: int = 17,
    up_axis: int = 2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Measure signed pelvis pitch in each frame's local sagittal plane.

    ``joints`` has shape ``[...,22,3]`` and is normally the output of the v2
    differentiable FK.  The lateral axis is the horizontal left/right hip
    axis (shoulders are a fallback); the sagittal forward axis is
    ``cross(lateral, up)``.  The pelvis-to-spine vector is projected onto that
    plane and measured from the upright direction.  ``rest_offsets`` can be
    supplied to subtract the neutral SMPL-X anatomical offset; it is optional
    for injected test skeletons whose neutral spine is upright.

    No root rotation or travel heading is used.  Consequently, jointly yawing
    all joints by any angle leaves the result unchanged.  Near-collinear hips,
    shoulders, or spine vectors use finite deterministic fallbacks and never
    divide by zero.
    """

    if joints.ndim < 3 or joints.shape[-2:] != (22, 3):
        raise ValueError(f"joints must have shape [...,22,3], got {tuple(joints.shape)}")
    if not torch.is_floating_point(joints) or not torch.isfinite(joints).all():
        raise ValueError("joints must be finite floating point")
    if up_axis not in (0, 1, 2) or eps <= 0:
        raise ValueError("up_axis must be 0,1,2 and eps must be positive")
    up = _axis_unit(up_axis, dtype=joints.dtype, device=joints.device)
    hip = joints[..., right_hip_index, :] - joints[..., left_hip_index, :]
    shoulders = joints[..., right_shoulder_index, :] - joints[..., left_shoulder_index, :]
    # Project candidate lateral axes into the horizontal plane first.  This
    # avoids pitch contaminating the plane normal and makes yaw cancellation
    # exact up to floating point arithmetic.
    hip = hip - (hip * up).sum(-1, keepdim=True) * up
    shoulders = shoulders - (shoulders * up).sum(-1, keepdim=True) * up
    hip_norm = torch.linalg.vector_norm(hip, dim=-1, keepdim=True)
    lateral_fallback = _axis_unit((up_axis + 1) % 3, dtype=joints.dtype, device=joints.device)
    lateral_fallback = lateral_fallback - (lateral_fallback * up).sum(-1, keepdim=True) * up
    lateral_fallback = lateral_fallback / torch.linalg.vector_norm(lateral_fallback).clamp_min(eps)
    lateral = torch.where(
        hip_norm > eps,
        hip / hip_norm.clamp_min(eps),
        _stable_unit(shoulders, lateral_fallback, eps),
    )
    forward = torch.cross(lateral, up.expand_as(lateral), dim=-1)
    forward = _stable_unit(forward, _axis_unit((up_axis + 2) % 3, dtype=joints.dtype, device=joints.device), eps)
    spine = joints[..., spine_index, :] - joints[..., pelvis_index, :]
    forward_component = (spine * forward).sum(-1)
    vertical_component = (spine * up).sum(-1)
    angle = torch.atan2(forward_component, vertical_component) * (180.0 / math.pi)
    if rest_offsets is not None:
        rest = torch.as_tensor(rest_offsets, dtype=joints.dtype, device=joints.device)
        if rest.shape != (22, 3):
            raise ValueError("rest_offsets must have shape [22,3]")
        neutral_lateral = rest[right_hip_index] - rest[left_hip_index]
        neutral_lateral = neutral_lateral - (neutral_lateral * up).sum(-1, keepdim=True) * up
        neutral_lateral = _stable_unit(neutral_lateral, lateral_fallback, eps)
        neutral_forward = torch.cross(neutral_lateral, up, dim=-1)
        neutral_forward = _stable_unit(
            neutral_forward,
            _axis_unit((up_axis + 2) % 3, dtype=joints.dtype, device=joints.device),
            eps,
        )
        neutral_spine = rest[spine_index] - rest[pelvis_index]
        neutral_angle = torch.atan2((neutral_spine * neutral_forward).sum(-1), (neutral_spine * up).sum(-1)) * (180.0 / math.pi)
        angle = angle - neutral_angle
    return angle


def relative_pitch_delta_degrees(
    base_rotation: torch.Tensor,
    edited_rotation: torch.Tensor,
    heading: torch.Tensor,
) -> torch.Tensor:
    """Legacy edited-minus-base proxy in degrees."""

    return pelvis_pitch_degrees(edited_rotation, heading) - pelvis_pitch_degrees(base_rotation, heading)


def pelvis_pitch_curve_v2(
    motion_phys: torch.Tensor,
    *,
    skeleton=None,
    rest_offsets: torch.Tensor | None = None,
    parents=None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute the v2 local-sagittal angle directly from packed 276-D data."""

    from .consistency_v2 import differentiable_forward_kinematics
    from .phase1 import MOTION_LAYOUT, decode_rot6d_safe

    if motion_phys.ndim not in (2, 3) or motion_phys.shape[-1] != 276:
        raise ValueError("motion_phys must have shape [T,276] or [B,T,276]")
    frames_axis = -2
    local = decode_rot6d_safe(
        motion_phys[..., MOTION_LAYOUT.body_pose].reshape(*motion_phys.shape[:-1], 21, 6)
    )
    root = decode_rot6d_safe(motion_phys[..., MOTION_LAYOUT.root_rotation])
    translation = motion_phys[..., MOTION_LAYOUT.root_translation]
    fk = differentiable_forward_kinematics(
        local,
        root,
        translation,
        skeleton=skeleton,
        rest_offsets=(None if skeleton is not None else rest_offsets),
        parents=(None if skeleton is not None else parents),
    )
    neutral_offsets = rest_offsets
    if neutral_offsets is None and skeleton is not None and hasattr(skeleton, "rest_offsets"):
        neutral_offsets = skeleton.rest_offsets
    return pelvis_pitch_degrees_v2(
        fk.joints,
        rest_offsets=neutral_offsets,
        eps=eps,
    )


__all__ = [
    "heading_from_velocity",
    "pelvis_pitch_degrees",
    "pelvis_pitch_degrees_v2",
    "pelvis_pitch_curve_v2",
    "relative_pitch_delta_degrees",
]
