"""专项 tests for the v2 absolute-mean pelvis evaluation boundary."""

import math

import pytest
import torch

from evaluation.absolute_mean_pelvis_v2 import (
    DEFAULT_CONSISTENCY_TOLERANCES,
    audit_motion_consistency,
    control_success_gate,
    evaluate_single,
    summarize_rows,
)
from motion_rep.consistency_v2 import SMPLX_22_PARENTS, differentiable_forward_kinematics
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe, encode_rot6d
from motion_rep.sagittal_pelvis_angle import pelvis_sagittal_tilt_degrees


def _rz(angle: float | torch.Tensor) -> torch.Tensor:
    angle = torch.as_tensor(angle, dtype=torch.float32)
    c, s = torch.cos(angle), torch.sin(angle)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    return torch.stack((c, -s, z, s, c, z, z, z, o), dim=-1).reshape(
        *angle.shape, 3, 3
    )


def _rx(angle: float | torch.Tensor) -> torch.Tensor:
    angle = torch.as_tensor(angle, dtype=torch.float32)
    c, s = torch.cos(angle), torch.sin(angle)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    return torch.stack((o, z, z, z, c, -s, z, s, c), dim=-1).reshape(
        *angle.shape, 3, 3
    )


def _neutral_root() -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=torch.float32,
    )


def _consistent_motion(frame_count: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = torch.zeros(22, 3)
    offsets[1:, 2] = 0.1
    local = torch.eye(3).reshape(1, 1, 3, 3).expand(frame_count, 21, 3, 3).clone()
    yaws = torch.linspace(-0.7, 2.4, frame_count)
    tilts = torch.linspace(-0.12, 0.18, frame_count)
    roots = torch.stack(
        [_rz(float(yaw)) @ _rx(float(tilt)) @ _neutral_root() for yaw, tilt in zip(yaws, tilts)]
    )
    translation = torch.zeros(frame_count, 3)
    translation[:, 1] = torch.arange(frame_count) * 0.2
    joints = differentiable_forward_kinematics(
        local, roots, translation, rest_offsets=offsets, parents=SMPLX_22_PARENTS
    ).joints
    motion = torch.zeros(frame_count, MOTION_LAYOUT.total_dim)
    motion[:, MOTION_LAYOUT.body_pose] = encode_rot6d(local).reshape(frame_count, -1)
    motion[:, MOTION_LAYOUT.joints] = joints.reshape(frame_count, -1)
    motion[:, MOTION_LAYOUT.joints_velocity] = torch.cat(
        (joints[1:] - joints[:-1], torch.zeros(1, 22, 3)), dim=0
    ).reshape(frame_count, -1)
    motion[:, MOTION_LAYOUT.root_rotation] = encode_rot6d(roots)
    root_velocity = torch.cat(
        (roots[1:] @ roots[:-1].transpose(-1, -2), torch.eye(3).unsqueeze(0)), dim=0
    )
    motion[:, MOTION_LAYOUT.root_rotation_velocity] = encode_rot6d(root_velocity)
    motion[:, MOTION_LAYOUT.root_translation] = translation
    motion[:, MOTION_LAYOUT.root_translation_velocity] = torch.cat(
        (translation[1:] - translation[:-1], torch.zeros(1, 3)), dim=0
    )
    return motion, offsets, roots


def _evaluate(motion: torch.Tensor, offsets: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
    if mask is None:
        mask = torch.ones(motion.shape[0], dtype=torch.bool)
    root = motion[:, MOTION_LAYOUT.root_rotation]
    target = float(pelvis_sagittal_tilt_degrees(decode_rot6d_safe(root))[mask].mean())
    return evaluate_single(
        sample_id="synthetic",
        method="g0",
        seed=0,
        target_mean_deg=target,
        baseline_phys=motion,
        candidate_phys=motion,
        valid_mask=mask,
        rest_offsets=offsets,
        parents=SMPLX_22_PARENTS,
    )


def test_pure_yaw_does_not_change_v2_angle_authority():
    _, _, roots = _consistent_motion()
    tilts = torch.tensor([-0.12, -0.02, 0.08, 0.18])
    roots = torch.stack(
        [_rz(yaw) @ _rx(float(tilt)) @ _neutral_root() for yaw, tilt in zip((0.0, math.pi / 4, math.pi / 2, math.pi), tilts)]
    )
    values = pelvis_sagittal_tilt_degrees(roots)
    torch.testing.assert_close(values, torch.rad2deg(tilts), atol=1e-5, rtol=0)


def test_complete_fk_and_all_velocity_channels_pass_consistency_audit():
    motion, offsets, _ = _consistent_motion()
    mask = torch.tensor([True] * 6 + [False, False])
    row = _evaluate(motion, offsets, mask)
    for key, tolerance in DEFAULT_CONSISTENCY_TOLERANCES.items():
        assert row[key] <= tolerance
    assert row["valid_frames"] == 6
    assert row["joint_velocity_residual_count"] == 5 * 22
    assert row["root_rotation_velocity_residual_count"] == 5
    assert row["root_translation_velocity_residual_count"] == 5
    summary = summarize_rows([row])
    assert control_success_gate({"target": summary})["passed"]


@pytest.mark.parametrize(
    ("field", "amount", "metric"),
    (
        ("joints", 0.01, "joint_fk_residual_max_m"),
        ("joints_velocity", 0.02, "joint_velocity_residual_max_m"),
        ("root_translation_velocity", 0.02, "root_translation_velocity_residual_max_m"),
    ),
)
def test_forged_j_dj_dt_channels_are_detected(field: str, amount: float, metric: str):
    motion, offsets, _ = _consistent_motion()
    forged = motion.clone()
    forged[:, getattr(MOTION_LAYOUT, field)] += amount
    audit = audit_motion_consistency(
        forged,
        torch.ones(motion.shape[0], dtype=torch.bool),
        rest_offsets=offsets,
        parents=SMPLX_22_PARENTS,
    )
    assert audit[metric] > amount * 0.5


def test_forged_root_rotation_velocity_is_detected_by_so3_residual():
    motion, offsets, _ = _consistent_motion()
    forged = motion.clone()
    forged[2, MOTION_LAYOUT.root_rotation_velocity] = encode_rot6d(_rz(0.3))
    audit = audit_motion_consistency(
        forged,
        torch.ones(motion.shape[0], dtype=torch.bool),
        rest_offsets=offsets,
        parents=SMPLX_22_PARENTS,
    )
    assert audit["root_rotation_velocity_residual_max_deg"] > 1.0


def test_consistency_failure_blocks_v2_success_gate():
    motion, offsets, _ = _consistent_motion()
    row = _evaluate(motion, offsets)
    summary = summarize_rows([row])
    assert control_success_gate({"target": summary})["passed"]
    summary["joint_velocity_residual_max_m"] = 1e-2
    failed = control_success_gate({"target": summary})
    assert failed["passed"] is False
    assert any("joint_velocity_residual_max_m" in item for item in failed["failures"])
