"""Regression tests for the tail-safe absolute-mean pelvis v3 boundary."""

import math

import torch

from motion_rep.phase1 import encode_rot6d
from motion_rep.rotation_transform import axis_angle_to_mat3x3, mat3x3_to_axis_angle
from motion_rep.sagittal_pelvis_angle import pelvis_sagittal_tilt_degrees
from motion_rep.consistency_v3 import fuse_root_rotation_tail_safe
from motion_rep.consistency_v3 import reconcile_motion_tensor_v3
from motion_rep.consistency_v2 import SMPLX_22_PARENTS
from motion_rep.finalize import finalize_motion
from motion_rep.phase1 import MOTION_LAYOUT


def _synthetic_root_stream(frames: int = 20) -> torch.Tensor:
    angles = torch.linspace(0.05, 0.18, frames, dtype=torch.float32)
    axis_angle = torch.zeros(frames, 3, dtype=torch.float32)
    axis_angle[:, 0] = angles
    return axis_angle_to_mat3x3(axis_angle).detach().requires_grad_(True)


def test_tail_safe_fusion_does_not_amplify_last_direct_root_gradient() -> None:
    direct = _synthetic_root_stream()
    identity = torch.eye(3, dtype=torch.float32).expand(direct.shape[0], 3, 3)
    velocity = encode_rot6d(identity)
    fused = fuse_root_rotation_tail_safe(direct, velocity, window=9, weight=1.0)
    objective = pelvis_sagittal_tilt_degrees(fused[:-1]).mean()
    gradient = torch.autograd.grad(objective, direct)[0].norm(dim=(-2, -1))
    ratio = float((gradient[-1] / gradient[-2].clamp_min(1e-8)).detach())
    assert ratio < 2.0


def test_tail_safe_hidden_pose_is_held_at_last_output_pose() -> None:
    direct = _synthetic_root_stream()
    identity = torch.eye(3, dtype=torch.float32).expand(direct.shape[0], 3, 3)
    velocity = encode_rot6d(identity)
    fused = fuse_root_rotation_tail_safe(direct, velocity, window=9, weight=1.0)
    hidden_increment = fused[-1] @ fused[-2].transpose(-1, -2)
    hidden_angle = float(mat3x3_to_axis_angle(hidden_increment).norm() * (180.0 / math.pi))
    assert hidden_angle < 1e-5


def test_v3_repack_is_fully_consistent_and_has_zero_hidden_velocity() -> None:
    offsets = torch.zeros(22, 3)
    offsets[0] = torch.tensor([0.25, -0.1, 0.4])
    offsets[1] = torch.tensor([0.2, 0.0, 0.0])
    offsets[2] = torch.tensor([-0.2, 0.0, 0.0])
    offsets[3] = torch.tensor([0.0, 0.0, 1.0])
    offsets[4:] = torch.tensor([0.0, 0.0, 0.2])
    stream_frames = 6
    local = torch.eye(3).reshape(1, 1, 3, 3).repeat(stream_frames, 21, 1, 1)
    roots = axis_angle_to_mat3x3(
        torch.stack((torch.linspace(0.0, 0.15, stream_frames), torch.zeros(stream_frames), torch.zeros(stream_frames)), dim=-1)
    )
    translations = torch.zeros(stream_frames, 3)
    translations[:, 1] = torch.arange(stream_frames) * 0.1
    joints = torch.zeros(stream_frames, 22, 3) + 10.0
    packed = finalize_motion(local, joints, roots, translations).motion
    result = reconcile_motion_tensor_v3(
        packed,
        fusion_window=9,
        anchor_weight=1.0,
        root_rotation_anchor_weight=1.0,
        rest_offsets=offsets,
        parents=SMPLX_22_PARENTS,
    )
    motion = result.motion
    root_velocity = motion[:, MOTION_LAYOUT.root_rotation_velocity]
    identity_rot6d = encode_rot6d(torch.eye(3)).expand_as(root_velocity)
    torch.testing.assert_close(root_velocity[-1], identity_rot6d[-1], atol=1e-5, rtol=0)
    joint_velocity = motion[:, MOTION_LAYOUT.joints_velocity].reshape(-1, 22, 3)
    positions = motion[:, MOTION_LAYOUT.joints].reshape(-1, 22, 3)
    torch.testing.assert_close(joint_velocity[:-1], positions[1:] - positions[:-1], atol=1e-6, rtol=0)
    assert result.protocol == "vimogen_276d_consistency_v3_tail_safe"
