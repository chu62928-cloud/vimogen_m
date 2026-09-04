"""Tests for the independent pelvis/contact sampling projection protocol."""

from __future__ import annotations

import json
import math

import pytest
import torch

from evaluation.pelvis_contact_compensation_v3 import (
    pelvis_pitch_delta_deg,
    target_root_rotation,
)
from motion_rep.phase1 import MOTION_LAYOUT, encode_rot6d
from motion_rep.pose_authority import authority_project
from sampling.pelvis_contact_flow_projection_v0_1 import (
    EUCLIDEAN_METRIC,
    KINEMATIC_TEMPORAL_METRIC,
    VARIABLES_PER_FRAME,
    PelvisContactFlowProjector,
    ProjectorConfig,
    autograd_jacobian,
    build_projection_metric,
    finite_difference_jacobian,
    predict_clean_endpoint,
    project_increment_norms,
    recompose_velocity,
    so3_exp,
    so3_log,
    solve_local_projection,
    write_strict_json,
)


def test_so3_exp_log_round_trip_and_zero_identity() -> None:
    tangent = torch.tensor(
        [[0.0, 0.0, 0.0], [0.12, -0.07, 0.04]], dtype=torch.float64
    )
    rotation = so3_exp(tangent)
    assert torch.equal(rotation[0], torch.eye(3, dtype=torch.float64))
    assert torch.allclose(so3_exp(so3_log(rotation)), rotation, atol=1.0e-9)


@pytest.mark.parametrize("dose", [0.0, 2.0, 5.0, 10.0])
def test_target_construction_reuses_frozen_v1_3_physical_sign(dose: float) -> None:
    m0 = so3_exp(torch.tensor([[0.1, -0.2, math.pi / 2.0]], dtype=torch.float64))
    target = target_root_rotation(m0, dose)
    if dose == 0.0:
        assert torch.equal(target, m0)
    observed = pelvis_pitch_delta_deg(m0, target)
    assert float(observed[0]) == pytest.approx(dose, abs=1.0e-7)


def test_clean_endpoint_recomposition_and_identity_scheduler_update() -> None:
    generator = torch.Generator().manual_seed(7)
    x_sigma = torch.randn((1, 4, 6), generator=generator)
    velocity = torch.randn((1, 4, 6), generator=generator)
    sigma = torch.tensor(0.43)
    sigma_next = torch.tensor(0.31)
    clean = predict_clean_endpoint(x_sigma, velocity, sigma)
    restored = recompose_velocity(x_sigma, clean, sigma)
    assert torch.allclose(restored, velocity, atol=2.0e-6)
    original_next = x_sigma + (sigma_next - sigma) * velocity
    identity_next = x_sigma + (sigma_next - sigma) * restored
    assert torch.allclose(identity_next, original_next, atol=2.0e-6)


def test_disabled_and_zero_dose_projection_are_velocity_identity() -> None:
    x_sigma = torch.randn(1, 5, 276)
    velocity = torch.randn_like(x_sigma)
    for config, dose in (
        (ProjectorConfig(enabled=False), 2.0),
        (ProjectorConfig(enabled=True), 0.0),
    ):
        projector = PelvisContactFlowProjector(config=config, target_dose=dose)
        result, diagnostics = projector.correct_velocity(
            x_sigma=x_sigma,
            velocity=velocity,
            sigma=0.5,
            valid_mask=torch.ones((1, 5), dtype=torch.bool),
        )
        assert torch.equal(result, velocity)
        assert diagnostics["projection_enabled"] is False


def test_trust_region_uses_vector_norms() -> None:
    value = torch.zeros((2, VARIABLES_PER_FRAME))
    value[:, :3] = 0.01
    rotations = value[:, 3:].reshape(2, -1, 3)
    rotations[:] = math.radians(5.0)
    projected = project_increment_norms(
        value,
        max_joint_increment_deg=5.0,
        max_root_translation_m=0.010,
    )
    assert float(torch.linalg.vector_norm(projected[:, :3], dim=-1).max()) <= 0.0100001
    rotation_norm = torch.linalg.vector_norm(
        projected[:, 3:].reshape(2, -1, 3), dim=-1
    )
    assert float(rotation_norm.max()) <= math.radians(5.0) + 1.0e-7
    assert projected[0, 0] < 0.01
    assert projected[0, 3] < math.radians(5.0)


def test_local_projection_reduces_pelvis_and_contact_residuals() -> None:
    metric = torch.eye(3, dtype=torch.float64)
    pelvis_j = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    contact_j = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float64)
    empty = torch.zeros((0, 3), dtype=torch.float64)
    step, _ = solve_local_projection(
        metric,
        pelvis_j,
        torch.tensor([0.2], dtype=torch.float64),
        contact_j,
        torch.tensor([-0.1], dtype=torch.float64),
        empty,
        torch.zeros(0, dtype=torch.float64),
        contact_weight=1.0e5,
    )
    assert abs(float(pelvis_j @ step - 0.2)) < 1.0e-8
    assert abs(float(contact_j @ step + 0.1)) < 1.0e-5


def test_metric_changes_distribution_and_spreads_correction() -> None:
    euclidean = build_projection_metric(
        2,
        ProjectorConfig(metric=EUCLIDEAN_METRIC),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    smooth = build_projection_metric(
        2,
        ProjectorConfig(metric=KINEMATIC_TEMPORAL_METRIC),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert not torch.allclose(euclidean, smooth)
    variables = euclidean.shape[0]
    constraint = torch.zeros((1, variables), dtype=torch.float64)
    # One contact task can be met by either of two adjacent rotation blocks.
    constraint[0, 3] = 1.0
    constraint[0, 6] = 1.0
    empty = torch.zeros((0, variables), dtype=torch.float64)
    rhs = torch.tensor([1.0], dtype=torch.float64)
    euclidean_step, _ = solve_local_projection(
        euclidean,
        empty,
        torch.zeros(0, dtype=torch.float64),
        constraint,
        rhs,
        empty,
        torch.zeros(0, dtype=torch.float64),
        contact_weight=1.0e6,
    )
    smooth_step, _ = solve_local_projection(
        smooth,
        empty,
        torch.zeros(0, dtype=torch.float64),
        constraint,
        rhs,
        empty,
        torch.zeros(0, dtype=torch.float64),
        contact_weight=1.0e6,
    )
    assert not torch.allclose(euclidean_step, smooth_step, atol=1.0e-5)
    assert int((smooth_step.abs() > 1.0e-7).sum()) > 2


def _toy_fk(value: torch.Tensor) -> torch.Tensor:
    """Small differentiable root/spine/leg chain for Jacobian validation."""

    root_translation = value[:3]
    rotations = so3_exp(value[3:].reshape(6, 3))
    offsets = torch.tensor(
        [
            [0.0, 0.0, 0.35],
            [0.0, 0.0, 0.30],
            [0.08, 0.0, -0.18],
            [0.0, 0.0, -0.35],
            [0.0, 0.0, -0.28],
            [0.0, 0.12, -0.04],
        ],
        dtype=value.dtype,
        device=value.device,
    )
    point = root_translation
    world = torch.eye(3, dtype=value.dtype, device=value.device)
    points = []
    for rotation, offset in zip(rotations, offsets):
        world = world @ rotation
        point = point + world @ offset
        points.append(point)
    pelvis_orientation = so3_log(rotations[0])
    heel = points[-1]
    toe = points[-1] + world @ torch.tensor(
        [0.0, 0.10, 0.0], dtype=value.dtype, device=value.device
    )
    return torch.cat((pelvis_orientation, heel, toe))


def test_fk_jacobian_matches_finite_difference_for_active_chain() -> None:
    point = torch.linspace(-0.03, 0.04, 21, dtype=torch.float64)
    automatic = autograd_jacobian(_toy_fk, point)
    finite = finite_difference_jacobian(_toy_fk, point, step=1.0e-5)
    assert torch.allclose(automatic, finite, atol=3.0e-5, rtol=2.0e-4)
    # Root translation, root, spine, hip, knee, ankle and foot variables all
    # participate in at least one tested pelvis/heel/toe component.
    assert bool((automatic.abs().amax(dim=0) > 1.0e-7).all())


def test_direct_pose_update_rebuilds_derived_276d_channels() -> None:
    frames = 4
    body = torch.eye(3).repeat(frames, 21, 1, 1)
    root = torch.eye(3).repeat(frames, 1, 1)
    motion = torch.zeros((frames, 276), dtype=torch.float32)
    motion[:, MOTION_LAYOUT.body_pose] = encode_rot6d(body).reshape(frames, 126)
    motion[:, MOTION_LAYOUT.root_rotation] = encode_rot6d(root)
    baseline = authority_project(motion).physical_motion
    edited = baseline.clone()
    edited_root = so3_exp(
        torch.stack(
            (
                torch.linspace(0.0, 0.15, frames),
                torch.zeros(frames),
                torch.zeros(frames),
            ),
            dim=-1,
        )
    ) @ root
    edited[:, MOTION_LAYOUT.root_rotation] = encode_rot6d(edited_root)
    rebuilt = authority_project(edited).physical_motion
    assert not torch.equal(
        rebuilt[:, MOTION_LAYOUT.joints], baseline[:, MOTION_LAYOUT.joints]
    )
    assert not torch.equal(
        rebuilt[:, MOTION_LAYOUT.root_rotation_velocity],
        edited[:, MOTION_LAYOUT.root_rotation_velocity],
    )


def test_strict_json_rejects_non_finite_values(tmp_path) -> None:
    output = tmp_path / "record.json"
    write_strict_json(output, {"finite": 1.0, "items": [2, 3]})
    assert json.loads(output.read_text(encoding="utf-8"))["finite"] == 1.0
    with pytest.raises(ValueError, match="NaN and Infinity"):
        write_strict_json(output, {"bad": float("nan")})
