"""Independent paired metrics for the S1 local-pelvis experiment."""

from __future__ import annotations

from typing import Any

import torch

from geometry.local_pelvis import (
    EVALUATOR_VERSION as ANGLE_EVALUATOR_VERSION,
    relative_pelvis_angle_curve_deg,
    wrap_angle_deg,
)
from motion_rep.consistency_v2 import differentiable_forward_kinematics
from motion_rep.phase1 import (
    MOTION_LAYOUT,
    SMPLX_22_JOINT_INDEX,
    decode_rot6d_safe,
)
from motion_rep.sagittal_pelvis_angle import pelvis_sagittal_tilt_degrees


EVALUATOR_VERSION = "local_pelvis_paired_metrics_v1"


def _summary(value: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(value.mean().detach().cpu()),
        "p95": float(torch.quantile(value, 0.95).detach().cpu()),
        "max": float(value.max().detach().cpu()),
    }


def _rotations_and_fk(motion: torch.Tensor):
    body = decode_rot6d_safe(
        motion[..., MOTION_LAYOUT.body_pose].reshape(
            motion.shape[0], motion.shape[1], 21, 6
        )
    )
    root = decode_rot6d_safe(motion[..., MOTION_LAYOUT.root_rotation])
    fk = differentiable_forward_kinematics(
        body, root, motion[..., MOTION_LAYOUT.root_translation]
    )
    return root, fk


def _geodesic_deg(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    relative = left.transpose(-1, -2) @ right
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1.0, 1.0
    )
    return torch.rad2deg(torch.acos(cosine))


def _direction_change_deg(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left / torch.linalg.vector_norm(left, dim=-1, keepdim=True).clamp_min(1.0e-8)
    right = right / torch.linalg.vector_norm(right, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return torch.rad2deg(torch.acos((left * right).sum(-1).clamp(-1.0, 1.0)))


def evaluate_local_pelvis_metrics(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    valid_mask: torch.Tensor,
    target_dose_deg: float,
) -> dict[str, Any]:
    if candidate.shape != baseline.shape or candidate.ndim != 3 or candidate.shape[-1] != 276:
        raise ValueError("candidate and baseline must share shape [B,T,276]")
    if valid_mask.dtype is not torch.bool or valid_mask.shape != candidate.shape[:2]:
        raise ValueError("valid_mask must be bool[B,T]")
    c_relative = relative_pelvis_angle_curve_deg(candidate)
    b_relative = relative_pelvis_angle_curve_deg(baseline)
    relative_delta = wrap_angle_deg(c_relative - b_relative)
    relative_error = wrap_angle_deg(relative_delta - float(target_dose_deg))

    c_root, c_fk = _rotations_and_fk(candidate)
    b_root, b_fk = _rotations_and_fk(baseline)
    # Historical C0 is anterior-up positive, so negate it for anterior-down.
    world_delta = wrap_angle_deg(
        -pelvis_sagittal_tilt_degrees(c_root)
        + pelvis_sagittal_tilt_degrees(b_root)
    )
    world_error = wrap_angle_deg(world_delta - float(target_dose_deg))
    spine3 = SMPLX_22_JOINT_INDEX["spine3"]
    trunk_rotation = _geodesic_deg(
        c_fk.global_rotations[..., spine3, :, :],
        b_fk.global_rotations[..., spine3, :, :],
    )
    neck = SMPLX_22_JOINT_INDEX["neck"]
    spine1 = SMPLX_22_JOINT_INDEX["spine1"]
    trunk_direction = _direction_change_deg(
        c_fk.joints[..., neck, :] - c_fk.joints[..., spine1, :],
        b_fk.joints[..., neck, :] - b_fk.joints[..., spine1, :],
    )
    foot_directions = []
    for ankle_name, foot_name in (
        ("left_ankle", "left_foot"),
        ("right_ankle", "right_foot"),
    ):
        ankle = SMPLX_22_JOINT_INDEX[ankle_name]
        foot = SMPLX_22_JOINT_INDEX[foot_name]
        foot_directions.append(
            _direction_change_deg(
                c_fk.joints[..., foot, :] - c_fk.joints[..., ankle, :],
                b_fk.joints[..., foot, :] - b_fk.joints[..., ankle, :],
            )
        )
    foot_direction = torch.maximum(*foot_directions)
    root_translation_mm = 1000.0 * torch.linalg.vector_norm(
        candidate[..., MOTION_LAYOUT.root_translation]
        - baseline[..., MOTION_LAYOUT.root_translation],
        dim=-1,
    )
    correction_step = wrap_angle_deg(relative_delta[:, 1:] - relative_delta[:, :-1]).abs()
    valid_pairs = valid_mask[:, 1:] & valid_mask[:, :-1]

    rows = []
    for index in range(candidate.shape[0]):
        mask = valid_mask[index]
        if not mask.any():
            raise ValueError("every sequence needs at least one valid frame")
        rel_abs = relative_error[index][mask].abs()
        world_abs = world_error[index][mask].abs()
        mean_delta = relative_delta[index][mask].mean()
        sign_correct = (
            None
            if abs(float(target_dose_deg)) <= 1.0e-8
            else bool(
                torch.sign(mean_delta)
                == torch.sign(torch.as_tensor(target_dose_deg, device=mean_delta.device))
            )
        )
        temporal = correction_step[index][valid_pairs[index]]
        temporal_summary = _summary(temporal) if temporal.numel() else None
        row = {
            "sequence_index": index,
            "valid_frames": int(mask.sum().item()),
            "relative_angle_error_deg": _summary(rel_abs),
            "relative_angle_delta_mean_deg": float(mean_delta.detach().cpu()),
            "relative_angle_sign_correct": sign_correct,
            "world_pelvis_error_deg": _summary(world_abs),
            "world_pelvis_delta_mean_deg": float(world_delta[index][mask].mean().detach().cpu()),
            "trunk_rotation_change_deg": _summary(trunk_rotation[index][mask]),
            "trunk_direction_change_deg": _summary(trunk_direction[index][mask]),
            "foot_direction_change_deg": _summary(foot_direction[index][mask]),
            "root_translation_deviation_mm": _summary(root_translation_mm[index][mask]),
            "relative_angle_correction_step_deg": temporal_summary,
            "target_hit": bool(
                rel_abs.mean() <= 1.0
                and torch.quantile(rel_abs, 0.95) <= 2.0
                and sign_correct is not False
            ),
            "world_pelvis_gate": bool(
                world_abs.mean() <= 1.0 and torch.quantile(world_abs, 0.95) <= 2.0
            ),
            "trunk_gate": bool(
                torch.quantile(trunk_rotation[index][mask], 0.95) <= 2.0
                and torch.quantile(trunk_direction[index][mask], 0.95) <= 2.0
            ),
            "root_trajectory_gate": bool(
                torch.quantile(root_translation_mm[index][mask], 0.95) <= 20.0
            ),
            "temporal_gate": bool(
                temporal.numel() > 0 and torch.quantile(temporal, 0.95) <= 1.0
            ),
        }
        rows.append(row)
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "angle_evaluator_version": ANGLE_EVALUATOR_VERSION,
        "target_dose_deg": float(target_dose_deg),
        "per_sequence": rows,
        "summary": {
            "sequence_count": len(rows),
            "target_hit_count": sum(int(row["target_hit"]) for row in rows),
            "world_pelvis_gate_count": sum(int(row["world_pelvis_gate"]) for row in rows),
            "trunk_gate_count": sum(int(row["trunk_gate"]) for row in rows),
            "root_trajectory_gate_count": sum(int(row["root_trajectory_gate"]) for row in rows),
            "temporal_gate_count": sum(int(row["temporal_gate"]) for row in rows),
            "contact_gate_status": "PENDING_MESH_MARKERS",
            "visual_review_status": "PENDING_USER_FINAL_REVIEW",
        },
    }


__all__ = ["EVALUATOR_VERSION", "evaluate_local_pelvis_metrics"]
