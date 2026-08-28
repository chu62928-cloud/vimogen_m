"""Pose-authoritative projection and geometry for relative root-forward v1.

The packed 276-D representation contains a direct pose view and four
redundant views.  This module gives the direct local pose, root rotation and
root translation a single, explicit authority boundary.  FK and forward
differences are rebuilt from that boundary; no velocity stream is fused.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .consistency_v2 import (
    Skeleton22,
    default_smplx_neutral_22_skeleton,
    differentiable_forward_kinematics,
    load_smplx_neutral_22_skeleton,
)
from .finalize import finalize_motion
from .phase1 import MOTION_LAYOUT, SMPLX_22_JOINT_INDEX, decode_rot6d_safe, encode_rot6d, validate_motion_tensor
from .rotation_transform import axis_angle_to_mat3x3, mat3x3_to_axis_angle


PROTOCOL_NAME = "vimogen_relative_root_forward_v1_pose_authoritative"
UP_AXIS = 2
LOCAL_FORWARD_AXIS = 2
EPS = 1e-7


def _stats(value: torch.Tensor, motion: torch.Tensor, name: str) -> torch.Tensor:
    value = torch.as_tensor(value, dtype=motion.dtype, device=motion.device)
    if value.shape[-1] != MOTION_LAYOUT.total_dim:
        raise ValueError(f"{name} must end in 276 channels")
    if value.ndim == motion.ndim - 1:
        value = value.unsqueeze(-2)
    if value.ndim not in (1, motion.ndim) and value.shape != motion.shape:
        raise ValueError(f"{name} is not broadcastable to {tuple(motion.shape)}")
    return value


def _physical(motion: torch.Tensor, mean: torch.Tensor | None, std: torch.Tensor | None) -> torch.Tensor:
    if (mean is None) != (std is None):
        raise ValueError("mean and std must be supplied together")
    if mean is None:
        return motion.float()
    mean = _stats(mean, motion, "mean").float()
    std = _stats(std, motion, "std").float()
    if not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError("std must be finite and strictly positive")
    return motion.float() * std + mean


def _standardized(motion: torch.Tensor, mean: torch.Tensor | None, std: torch.Tensor | None) -> torch.Tensor:
    if (mean is None) != (std is None):
        raise ValueError("mean and std must be supplied together")
    if mean is None:
        return motion
    mean = _stats(mean, motion, "mean").float()
    std = _stats(std, motion, "std").float()
    if not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError("std must be finite and strictly positive")
    return (motion - mean) / std


def _validate_mask(mask: torch.Tensor, shape: tuple[int, int]) -> None:
    if tuple(mask.shape) != shape or mask.dtype is not torch.bool:
        raise ValueError(f"valid_mask must have shape {shape} and dtype bool")
    if torch.any(mask[:, 1:] & ~mask[:, :-1]):
        raise ValueError("valid_mask must be a contiguous valid prefix")
    if torch.any(mask.sum(dim=-1) < 1):
        raise ValueError("each sample must contain at least one valid frame")


def _prefix_source(motion: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Hold the last valid direct pose and make the hidden pose explicit."""

    source = motion.clone()
    lengths = valid_mask.sum(dim=-1).long()
    for batch, length in enumerate(lengths.tolist()):
        if length < motion.shape[1]:
            source[batch, length:] = source[batch, length - 1:length]
    pose_mask = torch.zeros(
        (motion.shape[0], motion.shape[1] + 1), dtype=torch.bool, device=motion.device
    )
    for batch, length in enumerate(lengths.tolist()):
        pose_mask[batch, : length + 1] = True
    return source, pose_mask


def _direct_streams(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = source.shape[1]
    body = decode_rot6d_safe(source[..., MOTION_LAYOUT.body_pose].reshape(source.shape[0], frames, 21, 6)).float()
    root = decode_rot6d_safe(source[..., MOTION_LAYOUT.root_rotation]).float()
    translation = source[..., MOTION_LAYOUT.root_translation].float()
    return body, root, translation


def _authority_from_streams(
    body: torch.Tensor,
    root: torch.Tensor,
    translation: torch.Tensor,
    pose_mask: torch.Tensor,
    *,
    skeleton: Skeleton22,
) -> torch.Tensor:
    """FK and repack a batched T+1 authoritative pose stream."""

    # Callers normally provide only the T direct output poses.  The hidden
    # endpoint is always the last valid pose and is represented explicitly for
    # FK/finalisation.  Accepting an already-expanded stream keeps the helper
    # useful for tests and for the guidance inner loop.
    output_frames = pose_mask.shape[-1] - 1
    if body.shape[1] == output_frames:
        body = torch.cat((body, body[:, -1:]), dim=1)
        root = torch.cat((root, root[:, -1:]), dim=1)
        translation = torch.cat((translation, translation[:, -1:]), dim=1)
    elif body.shape[1] != output_frames + 1:
        raise ValueError("pose stream length does not match pose_mask")

    fk = differentiable_forward_kinematics(
        body,
        root,
        translation,
        skeleton=skeleton,
    )
    result = finalize_motion(
        body,
        fk.joints,
        root,
        translation,
        valid_mask=pose_mask,
        output_dtype=torch.float32,
    )
    return result.motion


def _geodesic(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # ``acos(trace)`` loses several meaningful bits near the identity in
    # float32, which can turn an exactly rebuilt dR into a spurious ~0.03°
    # residual.  Compute the SO(3) angle from both sine and cosine in float64.
    relative = a.double() @ b.double().transpose(-1, -2)
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.atan2(sine, cosine)


def _summary(values: torch.Tensor) -> dict[str, float | None]:
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


@dataclass(frozen=True)
class AuthorityProjection:
    motion: torch.Tensor
    valid_mask: torch.Tensor
    physical_motion: torch.Tensor
    audits: tuple[dict[str, Any], ...]
    protocol: str = PROTOCOL_NAME


def _projection_audit(before: torch.Tensor, after: torch.Tensor, valid: torch.Tensor) -> dict[str, Any]:
    """Describe how much the redundant representation changed."""

    mask = valid
    before_root = decode_rot6d_safe(before[..., MOTION_LAYOUT.root_rotation])
    after_root = decode_rot6d_safe(after[..., MOTION_LAYOUT.root_rotation])
    before_body = decode_rot6d_safe(before[..., MOTION_LAYOUT.body_pose].reshape(before.shape[0], before.shape[1], 21, 6))
    after_body = decode_rot6d_safe(after[..., MOTION_LAYOUT.body_pose].reshape(after.shape[0], after.shape[1], 21, 6))
    root_delta = _geodesic(after_root, before_root)[mask]
    body_delta = _geodesic(after_body, before_body)[mask].reshape(-1)
    trans_delta = torch.linalg.vector_norm(
        after[..., MOTION_LAYOUT.root_translation] - before[..., MOTION_LAYOUT.root_translation], dim=-1
    )[mask]
    joint_delta = torch.linalg.vector_norm(
        after[..., MOTION_LAYOUT.joints].reshape(before.shape[0], before.shape[1], 22, 3)
        - before[..., MOTION_LAYOUT.joints].reshape(before.shape[0], before.shape[1], 22, 3), dim=-1
    )[mask].reshape(-1)
    full_delta = (after - before)[mask]
    channel_rms = torch.sqrt(full_delta.square().mean(dim=0)) if full_delta.numel() else torch.zeros(276, device=before.device)
    channel_p95 = torch.quantile(full_delta.abs(), 0.95, dim=0) if full_delta.numel() else torch.zeros(276, device=before.device)
    channel_max = full_delta.abs().max(dim=0).values if full_delta.numel() else torch.zeros(276, device=before.device)
    audit = {
        "valid_frames": int(mask.sum().item()),
        "body_pose_geodesic_deg": _summary(body_delta * 180.0 / math.pi),
        "root_rotation_geodesic_deg": _summary(root_delta * 180.0 / math.pi),
        "root_translation_euclidean_m": _summary(trans_delta),
        "J_rebuild_rms_m": float(torch.sqrt(joint_delta.square().mean()).item()) if joint_delta.numel() else None,
        "J_rebuild_p95_m": float(torch.quantile(joint_delta, 0.95).item()) if joint_delta.numel() else None,
        "J_rebuild_max_m": float(joint_delta.max().item()) if joint_delta.numel() else None,
        "276d_channel_rms": float(torch.sqrt(full_delta.square().mean()).item()) if full_delta.numel() else None,
        "276d_channel_p95": float(torch.quantile(full_delta.abs(), 0.95).item()) if full_delta.numel() else None,
        "276d_channel_max": float(full_delta.abs().max().item()) if full_delta.numel() else None,
        "276d_per_channel_rms": channel_rms.detach().cpu().tolist(),
        "276d_per_channel_p95": channel_p95.detach().cpu().tolist(),
        "276d_per_channel_max": channel_max.detach().cpu().tolist(),
    }
    for name, span in (("body_pose", MOTION_LAYOUT.body_pose), ("root_rotation", MOTION_LAYOUT.root_rotation), ("root_translation", MOTION_LAYOUT.root_translation)):
        raw = (after[..., span] - before[..., span])[mask]
        audit[f"{name}_channel_rms"] = float(torch.sqrt(raw.square().mean()).item()) if raw.numel() else None
    for name, span in (("dJ", MOTION_LAYOUT.joints_velocity), ("dR", MOTION_LAYOUT.root_rotation_velocity), ("dT", MOTION_LAYOUT.root_translation_velocity)):
        delta = (after[..., span] - before[..., span])[mask]
        if delta.numel():
            audit[f"{name}_rebuild_rms"] = float(torch.sqrt(delta.square().mean()).item())
        else:
            audit[f"{name}_rebuild_rms"] = None
    before_dj = before[..., MOTION_LAYOUT.joints_velocity].reshape(before.shape[0], before.shape[1], 22, 3)
    after_dj = after[..., MOTION_LAYOUT.joints_velocity].reshape(after.shape[0], after.shape[1], 22, 3)
    dj_delta = torch.linalg.vector_norm(after_dj - before_dj, dim=-1)[mask]
    audit["dJ_rebuild_rms_m"] = float(torch.sqrt(dj_delta.square().mean()).item()) if dj_delta.numel() else None
    before_dr = decode_rot6d_safe(before[..., MOTION_LAYOUT.root_rotation_velocity])
    after_dr = decode_rot6d_safe(after[..., MOTION_LAYOUT.root_rotation_velocity])
    dr_delta = _geodesic(after_dr, before_dr)[mask] * 180.0 / math.pi
    audit["dR_rebuild_geodesic_rms_deg"] = float(torch.sqrt(dr_delta.square().mean()).item()) if dr_delta.numel() else None
    dt_delta = after[..., MOTION_LAYOUT.root_translation_velocity] - before[..., MOTION_LAYOUT.root_translation_velocity]
    dt_delta = torch.linalg.vector_norm(dt_delta, dim=-1)[mask]
    audit["dT_rebuild_rms_m"] = float(torch.sqrt(dt_delta.square().mean()).item()) if dt_delta.numel() else None
    return audit


def authority_project(
    motion: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    input_standardized: bool = False,
    output_standardized: bool = False,
    output_dtype: torch.dtype | None = torch.float32,
    skeleton: Skeleton22 | Mapping[str, Any] | None = None,
    rest_offsets: torch.Tensor | None = None,
    parents: tuple[int, ...] | list[int] | torch.Tensor | None = None,
) -> AuthorityProjection:
    """Project physical or standardised ``[B,T,276]`` data once.

    Only the direct local rotations, root rotations and root translations are
    read.  The last valid direct pose is held for the hidden T+1 pose and for
    every padded row.  The returned mask has T output rows; padded rows are
    zeroed by :func:`finalize_motion`.
    """

    validate_motion_tensor(motion)
    unbatched = motion.ndim == 2
    if unbatched:
        motion = motion.unsqueeze(0)
        if valid_mask is not None:
            valid_mask = valid_mask.unsqueeze(0)
    if motion.ndim != 3:
        raise ValueError("authority_project expects [T,276] or [B,T,276]")
    if input_standardized:
        physical = _physical(motion, mean, std)
    else:
        physical = motion.float()
    mask = torch.ones(physical.shape[:2], dtype=torch.bool, device=physical.device) if valid_mask is None else valid_mask.to(device=physical.device)
    _validate_mask(mask, tuple(physical.shape[:2]))
    if skeleton is not None and (rest_offsets is not None or parents is not None):
        raise ValueError("provide skeleton or rest_offsets/parents, not both")
    if skeleton is None and rest_offsets is not None:
        from .consistency_v2 import _resolve_skeleton
        tree = _resolve_skeleton(None, rest_offsets, parents)
    else:
        tree = default_smplx_neutral_22_skeleton() if skeleton is None else skeleton
    if not isinstance(tree, Skeleton22):
        if isinstance(tree, Mapping):
            from .consistency_v2 import _resolve_skeleton
            tree = _resolve_skeleton(tree, None, None)
        else:
            raise TypeError("skeleton must be Skeleton22 or a mapping")
    source, pose_mask = _prefix_source(physical, mask)
    body, root, translation = _direct_streams(source)
    packed = _authority_from_streams(body, root, translation, pose_mask, skeleton=tree)
    # The helper above is batched; keep one audit record per batch item.
    audits = tuple(_projection_audit(source[i:i + 1], packed[i:i + 1], mask[i:i + 1]) for i in range(source.shape[0]))
    consistency = consistency_report(packed, mask, skeleton=tree)
    audits = tuple({**audit, "consistency": record} for audit, record in zip(audits, consistency))
    output = _standardized(packed, mean, std) if output_standardized else packed
    if output_standardized and (mean is None or std is None):
        raise ValueError("mean/std are required for output_standardized=True")
    if output_dtype is not None:
        output = output.to(output_dtype)
    if unbatched:
        return AuthorityProjection(output[0], mask[0], packed[0], (audits[0],))
    return AuthorityProjection(output, mask, packed, audits)


@dataclass(frozen=True)
class RelativeRootForwardTargets:
    f0: torch.Tensor
    h0: torch.Tensor
    r0: torch.Tensor
    phi0_deg: torch.Tensor
    target_forward: torch.Tensor
    target_phi_deg: torch.Tensor
    valid_mask: torch.Tensor
    delta_deg: float


def _root_forward(root: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ez = torch.zeros(3, dtype=root.dtype, device=root.device)
    ez[LOCAL_FORWARD_AXIS] = 1.0
    up = torch.zeros(3, dtype=root.dtype, device=root.device)
    up[UP_AXIS] = 1.0
    f = root @ ez
    horizontal = f - (f * up).sum(-1, keepdim=True) * up
    norm = torch.linalg.vector_norm(horizontal, dim=-1, keepdim=True)
    if torch.any(norm < EPS):
        raise ValueError("horizontal root-forward projection is degenerate")
    h = horizontal / norm
    r = torch.cross(h, up.expand_as(h), dim=-1)
    phi = torch.atan2((f * up).sum(-1), norm.squeeze(-1)) * 180.0 / math.pi
    return f, h, r, phi


def prepare_targets(m0_physical: torch.Tensor, valid_mask: torch.Tensor, delta_deg: float) -> RelativeRootForwardTargets:
    """Freeze the M0 forward basis and the requested downward target."""

    validate_motion_tensor(m0_physical)
    unbatched = m0_physical.ndim == 2
    if unbatched:
        m0_physical = m0_physical.unsqueeze(0)
        valid_mask = valid_mask.unsqueeze(0)
    if m0_physical.ndim != 3:
        raise ValueError("m0_physical must have shape [T,276] or [B,T,276]")
    _validate_mask(valid_mask, tuple(m0_physical.shape[:2]))
    delta = float(delta_deg)
    if not math.isfinite(delta) or not -10.0 <= delta <= 10.0:
        raise ValueError("target_delta_deg must lie in [-10,10]")
    source, _ = _prefix_source(m0_physical, valid_mask)
    root = decode_rot6d_safe(source[..., MOTION_LAYOUT.root_rotation])
    f, h, r, phi = _root_forward(root)
    if not torch.isfinite(f).all():
        raise ValueError("M0 root-forward basis is non-finite")
    correction = axis_angle_to_mat3x3(r * (-delta * math.pi / 180.0))
    target = correction @ f.unsqueeze(-1)
    target = target.squeeze(-1) if target.ndim == f.ndim + 1 else target
    target_phi = phi - delta
    if unbatched:
        return RelativeRootForwardTargets(f[0], h[0], r[0], phi[0], target[0], target_phi[0], valid_mask[0].detach(), delta)
    return RelativeRootForwardTargets(f, h, r, phi, target, target_phi, valid_mask.detach(), delta)


def forward_vector_loss(forward: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Mean squared full-vector geodesic error in degrees squared."""

    cross = torch.linalg.vector_norm(torch.cross(forward, target, dim=-1), dim=-1)
    dot = (forward * target).sum(-1).clamp(-1.0, 1.0)
    angle_deg = torch.atan2(cross, dot) * 180.0 / math.pi
    mask = valid_mask.to(dtype=angle_deg.dtype)
    return (angle_deg.square() * mask).sum() / mask.sum().clamp_min(1.0)


def whole_body_audit(
    baseline_physical: torch.Tensor,
    candidate_physical: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, Any]:
    """Audit rigid whole-body follow-through in the frozen M0 sagittal frame."""

    validate_motion_tensor(baseline_physical)
    unbatched = baseline_physical.ndim == 2
    if unbatched:
        baseline_physical = baseline_physical.unsqueeze(0)
        candidate_physical = candidate_physical.unsqueeze(0)
        valid_mask = valid_mask.unsqueeze(0)
    validate_motion_tensor(candidate_physical)
    if baseline_physical.shape != candidate_physical.shape:
        raise ValueError("baseline and candidate must have the same shape")
    _validate_mask(valid_mask, tuple(baseline_physical.shape[:2]))
    baseline_source, _ = _prefix_source(baseline_physical, valid_mask)
    candidate_source, _ = _prefix_source(candidate_physical, valid_mask)
    b_root = decode_rot6d_safe(baseline_source[..., MOTION_LAYOUT.root_rotation])
    c_root = decode_rot6d_safe(candidate_source[..., MOTION_LAYOUT.root_rotation])
    _, h0, _, phi0 = _root_forward(b_root)
    _, _, _, phig = _root_forward(c_root)
    root_change = phi0 - phig
    b_joints = baseline_source[..., MOTION_LAYOUT.joints].reshape(*baseline_physical.shape[:-1], 22, 3)
    c_joints = candidate_source[..., MOTION_LAYOUT.joints].reshape(*candidate_physical.shape[:-1], 22, 3)
    spine1 = SMPLX_22_JOINT_INDEX["spine1"]
    neck = SMPLX_22_JOINT_INDEX["neck"]
    up = torch.zeros(3, dtype=baseline_physical.dtype, device=baseline_physical.device); up[2] = 1
    def trunk_angle(joints: torch.Tensor) -> torch.Tensor:
        vec = joints[..., neck, :] - joints[..., spine1, :]
        return torch.atan2((vec * h0).sum(-1), (vec * up).sum(-1)) * 180.0 / math.pi
    tau0, taug = trunk_angle(b_joints), trunk_angle(c_joints)
    trunk_change = taug - tau0
    local_change = root_change - trunk_change
    mask = valid_mask
    def masked_summary(values: torch.Tensor) -> dict[str, float | None]:
        return _summary(values[mask].abs())
    root_abs = masked_summary(root_change)
    trunk_abs = masked_summary(trunk_change)
    local_abs = masked_summary(local_change)
    root_median = max(root_abs["median"] or 0.0, 0.5)
    q_rigid = (trunk_abs["median"] or 0.0) / root_median
    if q_rigid >= 0.8: label = "全身刚性随动主导"
    elif q_rigid <= 0.2: label = "躯干补偿主导"
    else: label = "混合变化"
    body_b = decode_rot6d_safe(baseline_source[..., MOTION_LAYOUT.body_pose].reshape(*baseline_physical.shape[:-1], 21, 6))
    body_c = decode_rot6d_safe(candidate_source[..., MOTION_LAYOUT.body_pose].reshape(*candidate_physical.shape[:-1], 21, 6))
    local_rot = _geodesic(body_c, body_b) * 180.0 / math.pi
    named_rot = {}
    for name in ("left_hip", "right_hip", "spine1", "spine2", "spine3"):
        # body-pose index is joint index minus one (pelvis has no local channel).
        index = SMPLX_22_JOINT_INDEX[name] - 1
        named_rot[name] = _summary(local_rot[..., index][mask])
    thigh = {}
    for side in ("left", "right"):
        hip = SMPLX_22_JOINT_INDEX[f"{side}_hip"]; knee = SMPLX_22_JOINT_INDEX[f"{side}_knee"]
        bv = b_joints[..., knee, :] - b_joints[..., hip, :]
        cv = c_joints[..., knee, :] - c_joints[..., hip, :]
        direction_angle = torch.acos(
            (F.normalize(cv, dim=-1)
             * F.normalize(bv, dim=-1)).sum(-1).clamp(-1, 1)
        ) * 180.0 / math.pi
        thigh[side] = _summary(direction_angle[mask])
    feet = {}
    for side in ("left", "right"):
        idx = SMPLX_22_JOINT_INDEX[f"{side}_foot"]
        displacement = torch.linalg.vector_norm(c_joints[..., idx, :] - b_joints[..., idx, :], dim=-1)
        vertical = (c_joints[..., idx, 2] - b_joints[..., idx, 2])[mask]
        feet[side] = {"position_rms_m": float(torch.sqrt(displacement[mask].square().mean()).item()), "position_p95_m": float(torch.quantile(displacement[mask], .95).item()), "vertical_offset_mean_m": float(vertical.mean().item())}
    # Contact frames are fixed from the M0 horizontal foot speed.
    contact = {}
    for side in ("left", "right"):
        idx = SMPLX_22_JOINT_INDEX[f"{side}_foot"]
        bp = b_joints[..., idx, :]
        cp = c_joints[..., idx, :]
        bv = torch.zeros_like(bp); cv = torch.zeros_like(cp)
        bv[:, 1:] = bp[:, 1:] - bp[:, :-1]; cv[:, 1:] = cp[:, 1:] - cp[:, :-1]
        speed = torch.linalg.vector_norm(bv[..., :2], dim=-1)
        fixed_contact = mask & (speed <= 0.02)
        increment = torch.linalg.vector_norm(cv[..., :2], dim=-1) - speed
        contact[side] = {"contact_frames": int(fixed_contact.sum().item()), "horizontal_speed_increment_mean_m": float(increment[fixed_contact].mean().item()) if fixed_contact.any() else None, "horizontal_speed_increment_p95_m": float(torch.quantile(increment[fixed_contact].abs(), .95).item()) if fixed_contact.any() else None}
    result = {"root_change_deg": root_abs, "trunk_change_deg": trunk_abs, "root_relative_trunk_change_deg": local_abs, "q_rigid": float(q_rigid), "interpretation": label, "local_rotation_change_deg": named_rot, "thigh_world_direction_change_deg": thigh, "feet": feet, "fixed_m0_contact": contact}
    return result


def consistency_report(
    motion_physical: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    skeleton: Skeleton22 | Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Check the four hard 276-D consistency residuals for each sample."""

    validate_motion_tensor(motion_physical)
    if motion_physical.ndim != 3:
        raise ValueError("motion_physical must have shape [B,T,276]")
    _validate_mask(valid_mask, tuple(motion_physical.shape[:2]))
    tree = default_smplx_neutral_22_skeleton() if skeleton is None else skeleton
    if not isinstance(tree, Skeleton22):
        from .consistency_v2 import _resolve_skeleton
        tree = _resolve_skeleton(tree, None, None)
    source, pose_mask = _prefix_source(motion_physical.float(), valid_mask)
    body, root, translation = _direct_streams(source)
    body_stream = torch.cat((body, body[:, -1:]), dim=1)
    root_stream = torch.cat((root, root[:, -1:]), dim=1)
    trans_stream = torch.cat((translation, translation[:, -1:]), dim=1)
    fk = differentiable_forward_kinematics(body_stream, root_stream, trans_stream, skeleton=tree)
    expected_j = fk.joints[:, :-1]
    actual_j = motion_physical[..., MOTION_LAYOUT.joints].reshape(motion_physical.shape[0], motion_physical.shape[1], 22, 3)
    expected_dj = fk.joints[:, 1:] - fk.joints[:, :-1]
    actual_dj = motion_physical[..., MOTION_LAYOUT.joints_velocity].reshape(motion_physical.shape[0], motion_physical.shape[1], 22, 3)
    actual_dr = decode_rot6d_safe(motion_physical[..., MOTION_LAYOUT.root_rotation_velocity])
    expected_dr = root_stream[:, 1:] @ root_stream[:, :-1].transpose(-1, -2)
    actual_dt = motion_physical[..., MOTION_LAYOUT.root_translation_velocity]
    expected_dt = trans_stream[:, 1:] - trans_stream[:, :-1]
    rows = []
    for index in range(motion_physical.shape[0]):
        m = valid_mask[index]
        j = (actual_j[index] - expected_j[index]).abs()[m]
        dj = (actual_dj[index] - expected_dj[index]).abs()[m]
        dr = _geodesic(actual_dr[index], expected_dr[index])[m] * 180.0 / math.pi
        dt = (actual_dt[index] - expected_dt[index]).abs()[m]
        rows.append({
            "J_FK_max_m": float(j.max()) if j.numel() else None,
            "dJ_max_m": float(dj.max()) if dj.numel() else None,
            "dR_max_deg": float(dr.max()) if dr.numel() else None,
            "dT_max_m": float(dt.max()) if dt.numel() else None,
            "thresholds": {"J_FK_m": 1e-5, "dJ_m": 1e-6, "dR_deg": 1e-4, "dT_m": 1e-6},
            "passed": bool(
                (not j.numel() or j.max() <= 1e-5)
                and (not dj.numel() or dj.max() <= 1e-6)
                and (not dr.numel() or dr.max() <= 1e-4)
                and (not dt.numel() or dt.max() <= 1e-6)
            ),
        })
    return tuple(rows)


# Descriptive aliases used by downstream evaluation scripts and notebooks.
rebuild_derived_channels = authority_project
pose_authority_project = authority_project
root_forward_geometry = _root_forward


__all__ = [
    "PROTOCOL_NAME", "AuthorityProjection", "RelativeRootForwardTargets",
    "authority_project", "prepare_targets", "forward_vector_loss", "whole_body_audit", "consistency_report",
    "rebuild_derived_channels", "pose_authority_project",
    "root_forward_geometry",
    "_authority_from_streams", "_direct_streams", "_prefix_source", "_root_forward",
]
