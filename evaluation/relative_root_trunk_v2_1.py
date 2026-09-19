"""Geometry and paired naturalness gates for root--trunk v2.1.

This module deliberately contains no optimiser code.  The v2.1 optimiser
uses only :func:`root_trunk_relative_angle_deg`; foot/contact quantities are
evaluated here after generation with a fixed M0 evidence mask.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from motion_rep.consistency_v2 import differentiable_forward_kinematics
from motion_rep.phase1 import MOTION_LAYOUT, SMPLX_22_JOINT_INDEX, decode_rot6d_safe, validate_motion_tensor
from motion_rep.pose_authority import _root_forward
from motion_rep.rotation_transform import rot6d_to_axis_angle


PASS = "PASS"
FAIL = "FAIL"
NOT_EVALUABLE = "NOT_EVALUABLE"
_EPS = 1.0e-8


def _up_like(value: torch.Tensor) -> torch.Tensor:
    up = torch.zeros(3, dtype=value.dtype, device=value.device)
    up[2] = 1.0
    return up


def wrap_angle_deg(value: torch.Tensor) -> torch.Tensor:
    """Wrap angles to ``[-180, 180)`` without a discontinuity at 360 degrees."""

    return torch.remainder(value + 180.0, 360.0) - 180.0


def _normalise_projected(value: torch.Tensor, name: str) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    if torch.any(norm <= _EPS):
        raise ValueError(f"{name} projection onto the M0 sagittal plane is degenerate")
    return value / norm


def root_trunk_relative_angle_deg(
    root_rotation: torch.Tensor,
    joints: torch.Tensor,
    *,
    m0_heading: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the signed root-forward/trunk angle in the M0 sagittal frame.

    ``root_rotation`` is the direct root rotation and ``joints`` are FK
    joints, with the final dimension ordered as ``[x,y,z]``.  The angle is
    measured from the trunk axis (spine1 -> neck) to the root forward vector,
    around the M0 right axis.  A supplied heading is broadcast across all
    frames; otherwise each input frame supplies its own M0 heading.  The
    latter is useful for evaluating an M0 trajectory, while the optimiser
    always freezes the heading from M0.
    """

    if root_rotation.shape[-2:] != (3, 3):
        raise ValueError("root_rotation must end in [3,3]")
    if joints.shape[-2:] != (22, 3):
        raise ValueError("joints must end in [22,3]")
    if root_rotation.shape[:-2] != joints.shape[:-2]:
        raise ValueError("root_rotation and joints must have matching prefixes")
    if not torch.isfinite(root_rotation).all() or not torch.isfinite(joints).all():
        raise ValueError("root_rotation and joints must be finite")

    root_forward, frame_heading, _, _ = _root_forward(root_rotation)
    heading = frame_heading if m0_heading is None else m0_heading
    if heading.shape[-1] != 3:
        raise ValueError("m0_heading must end in [3]")
    heading = _normalise_projected(heading, "M0 heading")
    up = _up_like(root_rotation)
    right = torch.cross(heading, up.expand_as(heading), dim=-1)
    trunk = joints[..., SMPLX_22_JOINT_INDEX["neck"], :] - joints[..., SMPLX_22_JOINT_INDEX["spine1"], :]
    root_in_plane = _normalise_projected(
        root_forward - (root_forward * right).sum(-1, keepdim=True) * right,
        "root-forward",
    )
    trunk_in_plane = _normalise_projected(
        trunk - (trunk * right).sum(-1, keepdim=True) * right,
        "trunk",
    )
    sine = (torch.cross(trunk_in_plane, root_in_plane, dim=-1) * right).sum(-1)
    cosine = (trunk_in_plane * root_in_plane).sum(-1).clamp(-1.0, 1.0)
    angle = torch.atan2(sine, cosine) * (180.0 / math.pi)
    return wrap_angle_deg(angle)


def relative_angle_target_deg(
    m0_root_rotation: torch.Tensor,
    m0_joints: torch.Tensor,
    target_delta_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Freeze the M0 heading and return ``(M0 angle, M0 + delta)``."""

    if not math.isfinite(float(target_delta_deg)):
        raise ValueError("target_delta_deg must be finite")
    _, heading, _, _ = _root_forward(m0_root_rotation)
    baseline = root_trunk_relative_angle_deg(
        m0_root_rotation, m0_joints, m0_heading=heading
    )
    return baseline, wrap_angle_deg(baseline + float(target_delta_deg))


def direct_joints_from_motion(motion_physical: torch.Tensor) -> torch.Tensor:
    """Rebuild FK joints from the three direct pose streams only."""

    validate_motion_tensor(motion_physical)
    if motion_physical.ndim != 3:
        raise ValueError("motion_physical must have shape [B,T,276]")
    body = decode_rot6d_safe(
        motion_physical[..., MOTION_LAYOUT.body_pose].reshape(
            motion_physical.shape[0], motion_physical.shape[1], 21, 6
        )
    ).float()
    root = decode_rot6d_safe(motion_physical[..., MOTION_LAYOUT.root_rotation]).float()
    translation = motion_physical[..., MOTION_LAYOUT.root_translation].float()
    fk = differentiable_forward_kinematics(body, root, translation)
    return fk.joints


def direct_smpl_parameters(motion_physical: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build SMPL-X inputs from direct channels without velocity recovery."""

    validate_motion_tensor(motion_physical)
    if motion_physical.ndim != 3:
        raise ValueError("motion_physical must have shape [B,T,276]")
    body = rot6d_to_axis_angle(
        motion_physical[..., MOTION_LAYOUT.body_pose].reshape(-1, 6)
    ).reshape(motion_physical.shape[0], motion_physical.shape[1], 21, 3)
    root = rot6d_to_axis_angle(
        motion_physical[..., MOTION_LAYOUT.root_rotation].reshape(-1, 6)
    ).reshape(motion_physical.shape[0], motion_physical.shape[1], 3)
    return {
        "body_pose": body,
        "global_orient": root,
        "transl": motion_physical[..., MOTION_LAYOUT.root_translation],
    }


def relative_angle_metrics(
    baseline_physical: torch.Tensor,
    candidate_physical: torch.Tensor,
    valid_mask: torch.Tensor,
    target_delta_deg: float,
) -> dict[str, Any]:
    """Evaluate relative-angle error from direct pose/FK channels."""

    validate_motion_tensor(baseline_physical)
    validate_motion_tensor(candidate_physical)
    if baseline_physical.shape != candidate_physical.shape:
        raise ValueError("baseline and candidate must have equal shapes")
    if valid_mask.shape != baseline_physical.shape[:2]:
        raise ValueError("valid_mask must match [B,T]")
    b_root = decode_rot6d_safe(baseline_physical[..., MOTION_LAYOUT.root_rotation])
    c_root = decode_rot6d_safe(candidate_physical[..., MOTION_LAYOUT.root_rotation])
    b_joints = direct_joints_from_motion(baseline_physical)
    c_joints = direct_joints_from_motion(candidate_physical)
    _, heading, _, _ = _root_forward(b_root)
    baseline_angle = root_trunk_relative_angle_deg(b_root, b_joints, m0_heading=heading)
    candidate_angle = root_trunk_relative_angle_deg(c_root, c_joints, m0_heading=heading)
    target = wrap_angle_deg(baseline_angle + float(target_delta_deg))
    error = wrap_angle_deg(candidate_angle - target).abs()
    delta = wrap_angle_deg(candidate_angle - baseline_angle)
    mask = valid_mask.bool()
    values = error[mask]
    dose = delta[mask]
    if values.numel() == 0:
        return {
            "relative_angle_mae_deg": None,
            "relative_angle_p95_deg": None,
            "relative_angle_delta_mean_deg": None,
            "dose_sign_correct": False,
            "valid_count": 0,
        }
    return {
        "relative_angle_mae_deg": float(values.mean().item()),
        "relative_angle_p95_deg": float(torch.quantile(values, 0.95).item()),
        "relative_angle_delta_mean_deg": float(dose.mean().item()),
        "dose_sign_correct": bool(torch.sign(dose.mean()) == torch.sign(torch.as_tensor(target_delta_deg, device=dose.device))),
        "valid_count": int(values.numel()),
        "baseline_angle_mean_deg": float(baseline_angle[mask].mean().item()),
        "candidate_angle_mean_deg": float(candidate_angle[mask].mean().item()),
    }


def relative_angle_loss(
    baseline_physical: torch.Tensor,
    candidate_physical: torch.Tensor,
    valid_mask: torch.Tensor,
    target_delta_deg: float,
    *,
    temperature: float = 5.0,
) -> torch.Tensor:
    """Differentiable soft maximum of direct root--trunk angle error."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    b_root = decode_rot6d_safe(baseline_physical[..., MOTION_LAYOUT.root_rotation])
    c_root = decode_rot6d_safe(candidate_physical[..., MOTION_LAYOUT.root_rotation])
    b_joints = direct_joints_from_motion(baseline_physical)
    c_joints = direct_joints_from_motion(candidate_physical)
    _, heading, _, _ = _root_forward(b_root)
    baseline_angle = root_trunk_relative_angle_deg(b_root, b_joints, m0_heading=heading)
    candidate_angle = root_trunk_relative_angle_deg(c_root, c_joints, m0_heading=heading)
    error = wrap_angle_deg(candidate_angle - baseline_angle - float(target_delta_deg)).abs()
    masked = (float(temperature) * error).masked_fill(~valid_mask.bool(), float("-inf"))
    return (torch.logsumexp(masked, dim=-1) / float(temperature)).mean()


def paired_contact_evidence(
    heel: torch.Tensor,
    toe: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    contact_height_m: float = 0.025,
    contact_speed_m_per_frame: float = 0.030,
    flat_gap_m: float = 0.020,
    min_height_frames: int = 3,
    min_sliding_pairs: int = 3,
) -> dict[str, Any]:
    """Build fixed-M0 contact evidence with explicit frame/pair masks.

    The first frame has no valid speed.  General contact is used for lift,
    penetration, and sliding; flat contact is used only by toe-dominance
    diagnostics.  Sliding is restricted to pairs for which both adjacent M0
    frames are general contacts.
    """

    if heel.shape != toe.shape or heel.ndim != 2 or heel.shape[-1] != 3:
        raise ValueError("heel and toe must both have shape [T,3]")
    frames = heel.shape[0]
    if frames < 1 or not torch.isfinite(heel).all() or not torch.isfinite(toe).all():
        raise ValueError("heel and toe must be non-empty and finite")
    if valid_mask is None:
        valid = torch.ones(frames, dtype=torch.bool, device=heel.device)
    else:
        if valid_mask.shape != (frames,) or valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must have shape [T] and bool dtype")
        valid = valid_mask.to(device=heel.device)
    if torch.any(valid[1:] & ~valid[:-1]):
        raise ValueError("valid_mask must be a contiguous prefix")

    sole = torch.minimum(heel[:, 2], toe[:, 2])
    floor = torch.quantile(sole[valid], 0.05) if valid.any() else torch.tensor(float("nan"), device=heel.device)
    centre = 0.5 * (heel + toe)
    speed = torch.full((frames,), float("nan"), dtype=heel.dtype, device=heel.device)
    if frames > 1:
        speed[1:] = torch.linalg.vector_norm(centre[1:, :2] - centre[:-1, :2], dim=-1)
    speed_valid = torch.zeros(frames, dtype=torch.bool, device=heel.device)
    if frames > 1:
        speed_valid[1:] = valid[1:] & valid[:-1]
    general = valid & speed_valid & (sole <= floor + contact_height_m) & (speed <= contact_speed_m_per_frame)
    flat = general & ((heel[:, 2] - toe[:, 2]).abs() <= flat_gap_m)
    pair_mask = general[1:] & general[:-1] if frames > 1 else general[:0]
    flat_pair_mask = flat[1:] & flat[:-1] if frames > 1 else flat[:0]

    def evidence(values: torch.Tensor, minimum: int) -> dict[str, Any]:
        values = values[torch.isfinite(values)]
        if values.numel() < minimum:
            return {"mean": None, "p95": None, "count": int(values.numel()), "status": NOT_EVALUABLE}
        return {
            "mean": float(values.mean().item()),
            "p95": float(torch.quantile(values, 0.95).item()),
            "count": int(values.numel()),
            "status": PASS,
        }

    return {
        "floor_height_m": float(floor.item()),
        "contact_frames": int(general.sum().item()),
        "flat_contact_frames": int(flat.sum().item()),
        "continuous_contact_pairs": int(pair_mask.sum().item()),
        "continuous_flat_pairs": int(flat_pair_mask.sum().item()),
        "height_evidence": evidence((sole - floor).clamp_min(0.0)[general], min_height_frames),
        "sliding_evidence_m_per_frame": evidence(speed[1:][pair_mask], min_sliding_pairs),
        "valid_masks": {
            "valid": valid.detach().cpu().tolist(),
            "general_contact": general.detach().cpu().tolist(),
            "flat_contact": flat.detach().cpu().tolist(),
            "continuous_contact_pair": pair_mask.detach().cpu().tolist(),
            "continuous_flat_pair": flat_pair_mask.detach().cpu().tolist(),
        },
        "thresholds": {
            "contact_height_m": contact_height_m,
            "contact_speed_m_per_frame": contact_speed_m_per_frame,
            "flat_gap_m": flat_gap_m,
            "minimum_height_frames": min_height_frames,
            "minimum_sliding_pairs": min_sliding_pairs,
            "first_frame_speed_is_valid": False,
        },
    }


def _summary(values: torch.Tensor) -> dict[str, Any]:
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"mean": None, "p95": None, "count": 0}
    return {
        "mean": float(values.mean().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "count": int(values.numel()),
    }


def allowed_increase(baseline: float | None) -> float:
    return 0.001 if baseline is None else max(abs(float(baseline)) * 0.05, 0.001)


def paired_metric_status(candidate: dict[str, Any], baseline: dict[str, Any], minimum_count: int = 3) -> str:
    if int(candidate.get("count", 0)) < minimum_count or int(baseline.get("count", 0)) < minimum_count:
        return NOT_EVALUABLE
    for key in ("mean", "p95"):
        value, reference = candidate.get(key), baseline.get(key)
        if value is None or reference is None or float(value) > float(reference) + allowed_increase(reference):
            return FAIL
    return PASS


def evaluate_paired_foot_metrics(
    m0_heel: torch.Tensor,
    m0_toe: torch.Tensor,
    candidate_heel: torch.Tensor,
    candidate_toe: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compare candidate foot metrics with the same M0 evidence masks."""

    if m0_heel.shape != candidate_heel.shape or m0_toe.shape != candidate_toe.shape:
        raise ValueError("M0 and candidate foot patches must have equal shapes")
    evidence = paired_contact_evidence(m0_heel, m0_toe, valid_mask=valid_mask)
    masks = evidence["valid_masks"]
    general = torch.tensor(masks["general_contact"], dtype=torch.bool, device=m0_heel.device)
    pairs = torch.tensor(masks["continuous_contact_pair"], dtype=torch.bool, device=m0_heel.device)
    floor = torch.as_tensor(evidence["floor_height_m"], dtype=m0_heel.dtype, device=m0_heel.device)
    m0_sole = torch.minimum(m0_heel[:, 2], m0_toe[:, 2])
    candidate_sole = torch.minimum(candidate_heel[:, 2], candidate_toe[:, 2])
    m0_centre = 0.5 * (m0_heel + m0_toe)
    candidate_centre = 0.5 * (candidate_heel + candidate_toe)
    m0_speed = torch.linalg.vector_norm(m0_centre[1:, :2] - m0_centre[:-1, :2], dim=-1)
    candidate_speed = torch.linalg.vector_norm(candidate_centre[1:, :2] - candidate_centre[:-1, :2], dim=-1)
    baseline = {
        "sliding_m_per_frame": _summary(m0_speed[pairs]),
        "lift_m": _summary((m0_sole - floor).clamp_min(0.0)[general]),
        "penetration_m": _summary((floor - m0_sole).clamp_min(0.0)[general]),
    }
    candidate = {
        "sliding_m_per_frame": _summary(candidate_speed[pairs]),
        "lift_m": _summary((candidate_sole - floor).clamp_min(0.0)[general]),
        "penetration_m": _summary((floor - candidate_sole).clamp_min(0.0)[general]),
    }
    statuses = {
        key: paired_metric_status(candidate[key], baseline[key])
        for key in baseline
    }
    return {
        "contact_evidence": evidence,
        "baseline": baseline,
        "candidate": candidate,
        "statuses": statuses,
        "status": FAIL if FAIL in statuses.values() else NOT_EVALUABLE if NOT_EVALUABLE in statuses.values() else PASS,
        "allowed_increase": {key: {name: allowed_increase(baseline[key].get(name)) for name in ("mean", "p95")} for key in baseline},
    }


__all__ = [
    "PASS", "FAIL", "NOT_EVALUABLE", "wrap_angle_deg",
    "root_trunk_relative_angle_deg", "relative_angle_target_deg",
    "direct_joints_from_motion", "direct_smpl_parameters",
    "relative_angle_metrics", "relative_angle_loss",
    "paired_contact_evidence", "evaluate_paired_foot_metrics",
    "allowed_increase", "paired_metric_status",
]
