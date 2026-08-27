"""Yaw invariance, gradients and degeneracy tests for the v2 pelvis angle."""

import math

import pytest
import torch

pytest.importorskip("torch")

from motion_rep.pelvis_angle import pelvis_pitch_degrees_v2  # noqa: E402
from sampling.absolute_mean_pelvis_guidance import (  # noqa: E402
    apply_local_sagittal_correction,
    local_sagittal_normal,
)
from motion_rep.consistency_v2 import differentiable_forward_kinematics  # noqa: E402
from motion_rep.rotation_transform import axis_angle_to_mat3x3  # noqa: E402


def _joints(yaw: float = 0.0, pitch: float = 0.2) -> torch.Tensor:
    joints = torch.zeros(22, 3)
    joints[1] = torch.tensor([0.2, 0.0, 0.0])
    joints[2] = torch.tensor([-0.2, 0.0, 0.0])
    joints[3] = torch.tensor([math.sin(pitch), math.cos(pitch), 1.0])
    joints[16] = torch.tensor([0.2, 0.0, 0.5])
    joints[17] = torch.tensor([-0.2, 0.0, 0.5])
    if yaw:
        rotation = axis_angle_to_mat3x3(torch.tensor([[0.0, 0.0, yaw]]))[0]
        joints = joints @ rotation.T
    return joints


def test_local_sagittal_pitch_is_invariant_to_pure_yaw():
    values = [pelvis_pitch_degrees_v2(_joints(yaw=y)) for y in (0.0, math.pi / 2, math.pi, -math.pi / 2)]
    for value in values[1:]:
        assert value.item() == pytest.approx(values[0].item(), abs=1e-5)
    axes = [local_sagittal_normal(_joints(yaw=y)) for y in (0.0, math.pi / 2, math.pi)]
    assert axes[0].norm().item() == pytest.approx(1.0, abs=1e-5)
    assert axes[1].norm().item() == pytest.approx(1.0, abs=1e-5)


def test_local_sagittal_pitch_has_finite_gradient():
    joints = _joints().requires_grad_(True)
    loss = pelvis_pitch_degrees_v2(joints).square()
    loss.backward()
    assert joints.grad is not None and torch.isfinite(joints.grad).all()


def test_degenerate_hip_and_spine_directions_are_finite():
    joints = torch.zeros(22, 3, requires_grad=True)
    angle = pelvis_pitch_degrees_v2(joints)
    assert torch.isfinite(angle).all()
    angle.square().backward()
    assert joints.grad is not None and torch.isfinite(joints.grad).all()


def test_local_g1_correction_has_same_sign_and_magnitude_for_yaw_0_90_180():
    offsets = torch.zeros(22, 3)
    offsets[1] = torch.tensor([0.2, 0.0, 0.0])
    offsets[2] = torch.tensor([-0.2, 0.0, 0.0])
    offsets[3] = torch.tensor([0.0, 0.0, 1.0])
    for index in range(4, 22):
        offsets[index] = torch.tensor([0.0, 0.0, 0.2])
    local = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 21, 1, 1)
    before, after = [], []
    for yaw in (0.0, math.pi / 2, math.pi):
        root = axis_angle_to_mat3x3(torch.tensor([[0.0, 0.0, yaw]]))
        joints = differentiable_forward_kinematics(
            local, root, torch.zeros(1, 3), rest_offsets=offsets
        ).joints
        edited_root = apply_local_sagittal_correction(root, joints, 1.0)
        edited_joints = differentiable_forward_kinematics(
            local, edited_root, torch.zeros(1, 3), rest_offsets=offsets
        ).joints
        before.append(pelvis_pitch_degrees_v2(joints).item())
        after.append(pelvis_pitch_degrees_v2(edited_joints).item())
    deltas = [a - b for a, b in zip(after, before)]
    assert deltas[0] > 0
    assert deltas == pytest.approx([deltas[0]] * 3, abs=1e-4)
