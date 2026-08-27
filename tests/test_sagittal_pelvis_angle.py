"""专项 tests for root-rotation local-sagittal pelvis tilt v2."""

import math

import pytest
import torch

from motion_rep.sagittal_pelvis_angle import (
    apply_person_right_axis_rotation,
    pelvis_sagittal_tilt_degrees,
    person_forward_horizontal_axis,
    person_right_axis,
    person_yaw_radians,
    remove_person_yaw,
)


def _rz(angle: float | torch.Tensor) -> torch.Tensor:
    angle = torch.as_tensor(angle, dtype=torch.float64)
    c, s = torch.cos(angle), torch.sin(angle)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    return torch.stack((c, -s, z, s, c, z, z, z, o), dim=-1).reshape(
        *angle.shape, 3, 3
    )


def _rx(angle: float | torch.Tensor) -> torch.Tensor:
    angle = torch.as_tensor(angle, dtype=torch.float64)
    c, s = torch.cos(angle), torch.sin(angle)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    return torch.stack((o, z, z, z, c, -s, z, s, c), dim=-1).reshape(
        *angle.shape, 3, 3
    )


def _neutral_root() -> torch.Tensor:
    # SMPL-X local +z is canonical +y at yaw/pitch zero.
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=torch.float64,
    )


def _root(yaw: float, tilt: float = 0.2, roll: float = 0.0) -> torch.Tensor:
    # World yaw and pitch are applied to the neutral SMPL-X frame.  The final
    # local roll is around local +z, so it must not change the forward axis.
    return _rz(yaw) @ _rx(tilt) @ _neutral_root() @ _rz(roll)


def test_v2_angle_is_yaw_invariant_for_cardinal_and_diagonal_yaws():
    roots = torch.stack([_root(yaw) for yaw in (0.0, math.pi / 4, math.pi / 2, math.pi)])
    values = pelvis_sagittal_tilt_degrees(roots)
    torch.testing.assert_close(values, torch.full_like(values, math.degrees(0.2)), atol=1e-10, rtol=0)

    yaws = person_yaw_radians(roots)
    torch.testing.assert_close(yaws, torch.tensor([0.0, math.pi / 4, math.pi / 2, math.pi], dtype=torch.float64), atol=1e-10, rtol=0)
    de_yawed = remove_person_yaw(roots)
    forward = de_yawed @ torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(forward[:, 0], torch.zeros(4, dtype=torch.float64), atol=1e-10, rtol=0)
    assert torch.all(forward[:, 1] > 0)


def test_v2_angle_is_invariant_over_a_time_varying_turn_sequence():
    yaws = torch.linspace(-1.2, 2.8, 17, dtype=torch.float64)
    roots = torch.stack([_root(float(yaw), tilt=-0.13) for yaw in yaws])
    values = pelvis_sagittal_tilt_degrees(roots)
    torch.testing.assert_close(values, torch.full_like(values, math.degrees(-0.13)), atol=1e-10, rtol=0)


def test_local_roll_about_forward_axis_is_isolated():
    base = pelvis_sagittal_tilt_degrees(_root(0.7, tilt=0.25, roll=0.0))
    rolled = pelvis_sagittal_tilt_degrees(_root(0.7, tilt=0.25, roll=1.1))
    assert rolled.item() == pytest.approx(base.item(), abs=1e-10)


def test_degenerate_horizontal_projection_has_finite_value_and_gradient():
    # Identity maps SMPL-X local +z directly to canonical +z, leaving no
    # horizontal forward direction.  The API should still return a finite
    # vertical limit and finite gradients.
    root = torch.eye(3, dtype=torch.float64, requires_grad=True)
    angle = pelvis_sagittal_tilt_degrees(root)
    assert angle.item() == pytest.approx(90.0, abs=1e-8)
    angle.square().backward()
    assert root.grad is not None and torch.isfinite(root.grad).all()
    heading, horizontal_norm = person_forward_horizontal_axis(root.detach())
    torch.testing.assert_close(heading, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))
    assert horizontal_norm.item() == pytest.approx(0.0, abs=1e-12)


def test_person_right_axis_left_correction_adds_half_degree_at_each_yaw():
    roots = torch.stack([_root(yaw, tilt=0.1) for yaw in (0.0, math.pi / 4, math.pi / 2, math.pi)])
    before = pelvis_sagittal_tilt_degrees(roots)
    after = pelvis_sagittal_tilt_degrees(apply_person_right_axis_rotation(roots, 0.5))
    torch.testing.assert_close(after - before, torch.full_like(before, 0.5), atol=1e-8, rtol=0)

    right = person_right_axis(roots)
    expected = roots @ torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(right, expected, atol=1e-10, rtol=0)


def test_g1_axis_stays_normal_to_sagittal_plane_under_pelvis_roll():
    roots = torch.stack([_root(yaw, tilt=0.1, roll=0.8) for yaw in (0.0, math.pi / 2, math.pi)])
    right = person_right_axis(roots)
    torch.testing.assert_close(right[..., 2], torch.zeros(3, dtype=torch.float64), atol=1e-10, rtol=0)
    before = pelvis_sagittal_tilt_degrees(roots)
    after = pelvis_sagittal_tilt_degrees(apply_person_right_axis_rotation(roots, 0.5))
    torch.testing.assert_close(after - before, torch.full_like(before, 0.5), atol=1e-8, rtol=0)


def test_batched_shape_and_per_frame_correction_broadcasting():
    roots = torch.stack([_root(0.0), _root(math.pi / 2)]).reshape(1, 2, 3, 3)
    deltas = torch.tensor([[0.25, -0.25]], dtype=torch.float64)
    corrected = apply_person_right_axis_rotation(roots, deltas)
    assert corrected.shape == roots.shape
    delta = pelvis_sagittal_tilt_degrees(corrected) - pelvis_sagittal_tilt_degrees(roots)
    torch.testing.assert_close(delta, deltas, atol=1e-8, rtol=0)
