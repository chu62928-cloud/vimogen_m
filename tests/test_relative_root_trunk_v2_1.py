"""Deterministic geometry and paired-evidence tests for root--trunk v2.1."""

from __future__ import annotations

import math

import pytest
import torch

from evaluation.relative_root_trunk_v2_1 import (
    FAIL,
    NOT_EVALUABLE,
    PASS,
    evaluate_paired_foot_metrics,
    paired_contact_evidence,
    root_trunk_relative_angle_deg,
)
from sampling.differentiable_flow_sampler import DifferentiableSamplerConfig
from sampling.relative_root_trunk_guidance_v2_1 import (
    PROTOCOL_NAME,
    RelativeRootTrunkConfig,
)


def _rotation_x(angle_deg: float) -> torch.Tensor:
    angle = torch.tensor(angle_deg * math.pi / 180.0)
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_y(angle_deg: float) -> torch.Tensor:
    angle = torch.tensor(angle_deg * math.pi / 180.0)
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _motion_geometry() -> tuple[torch.Tensor, torch.Tensor]:
    # ViMoGen's direct forward axis is local z.  Rotate it into the world
    # horizontal plane for a non-degenerate synthetic M0.
    root = _rotation_y(90.0).reshape(1, 1, 3, 3).expand(1, 3, 3, 3).clone()
    joints = torch.zeros(1, 3, 22, 3)
    joints[..., 9, :] = torch.tensor([0.0, 0.0, 0.0])  # spine1
    joints[..., 12, :] = torch.tensor([1.0, 0.0, 1.0])  # neck
    return root, joints


def test_full_rigid_rotation_about_m0_right_axis_preserves_relative_angle() -> None:
    root, joints = _motion_geometry()
    _, heading, right, _ = __import__("motion_rep.pose_authority", fromlist=["_root_forward"])._root_forward(root)
    from motion_rep.rotation_transform import axis_angle_to_mat3x3

    rotation = axis_angle_to_mat3x3(right[0, 0] * (37.0 * math.pi / 180.0))
    rotated_root = rotation @ root
    rotated_joints = joints @ rotation.T
    before = root_trunk_relative_angle_deg(root, joints, m0_heading=heading)
    after = root_trunk_relative_angle_deg(rotated_root, rotated_joints, m0_heading=heading)
    assert torch.allclose(after, before, atol=1e-5)


def test_independent_root_rotation_changes_angle_with_the_requested_sign() -> None:
    root, joints = _motion_geometry()
    _, heading, right, _ = __import__("motion_rep.pose_authority", fromlist=["_root_forward"])._root_forward(root)
    # The angle convention is trunk -> root around +M0-right.  Applying a
    # positive right-axis rotation to the root increases the measured angle.
    from motion_rep.rotation_transform import axis_angle_to_mat3x3

    baseline = root_trunk_relative_angle_deg(root, joints, m0_heading=heading)
    candidate_root = axis_angle_to_mat3x3(right[0, 0] * (10.0 * math.pi / 180.0)) @ root
    candidate = root_trunk_relative_angle_deg(candidate_root, joints, m0_heading=heading)
    assert torch.allclose(candidate - baseline, torch.full_like(baseline, 10.0), atol=1e-4)


def test_independent_trunk_rotation_has_the_opposite_sign() -> None:
    root, joints = _motion_geometry()
    _, heading, right, _ = __import__("motion_rep.pose_authority", fromlist=["_root_forward"])._root_forward(root)
    from motion_rep.rotation_transform import axis_angle_to_mat3x3

    baseline = root_trunk_relative_angle_deg(root, joints, m0_heading=heading)
    rotated_joints = joints.clone()
    trunk = rotated_joints[..., 12, :] - rotated_joints[..., 9, :]
    rotated_joints[..., 12, :] = rotated_joints[..., 9, :] + (
        axis_angle_to_mat3x3(right[0, 0] * (10.0 * math.pi / 180.0)) @ trunk.unsqueeze(-1)
    ).squeeze(-1)
    candidate = root_trunk_relative_angle_deg(root, rotated_joints, m0_heading=heading)
    assert torch.allclose(candidate - baseline, torch.full_like(baseline, -10.0), atol=1e-4)


def test_degenerate_projection_is_explicitly_rejected() -> None:
    root = _rotation_y(90.0).reshape(1, 3, 3)
    joints = torch.zeros(1, 22, 3)
    joints[..., 12, 1] = 1.0  # parallel to M0 right axis
    with pytest.raises(ValueError, match="trunk projection"):
        root_trunk_relative_angle_deg(root, joints, m0_heading=torch.tensor([[1.0, 0.0, 0.0]]))


def test_contact_evidence_separates_general_and_flat_and_excludes_first_frame() -> None:
    heel = torch.zeros(7, 3)
    toe = torch.zeros(7, 3)
    heel[:, 0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4])
    toe[:, 0] = heel[:, 0]
    heel[3, 2] = 0.03  # still general contact, but not flat
    evidence = paired_contact_evidence(heel, toe)
    assert evidence["valid_masks"]["general_contact"][0] is False
    assert evidence["contact_frames"] == 5
    assert evidence["flat_contact_frames"] == 4
    assert evidence["continuous_contact_pairs"] == 4
    assert evidence["continuous_flat_pairs"] == 2


def test_isolated_contact_does_not_create_sliding_evidence() -> None:
    heel = torch.zeros(6, 3)
    toe = torch.zeros(6, 3)
    heel[2:, 2] = 0.1
    toe[2:, 2] = 0.1
    evidence = paired_contact_evidence(heel, toe, contact_speed_m_per_frame=0.01)
    assert evidence["continuous_contact_pairs"] == 0
    assert evidence["sliding_evidence_m_per_frame"]["status"] == NOT_EVALUABLE


def test_evidence_shortage_is_not_a_pass_and_known_sliding_increment_fails() -> None:
    m0_heel = torch.zeros(6, 3)
    m0_toe = torch.zeros(6, 3)
    candidate_heel = m0_heel.clone()
    candidate_toe = m0_toe.clone()
    candidate_heel[1:, 0] = torch.arange(1, 6, dtype=torch.float32) * 0.015
    candidate_toe[1:, 0] = candidate_heel[1:, 0]
    result = evaluate_paired_foot_metrics(m0_heel, m0_toe, candidate_heel, candidate_toe)
    assert result["statuses"]["sliding_m_per_frame"] == FAIL
    short = evaluate_paired_foot_metrics(
        m0_heel[:3], m0_toe[:3], candidate_heel[:3], candidate_toe[:3]
    )
    assert short["statuses"]["sliding_m_per_frame"] == NOT_EVALUABLE


def test_candidate_equal_to_m0_is_pass_when_evidence_is_sufficient() -> None:
    foot = torch.zeros(6, 3)
    result = evaluate_paired_foot_metrics(foot, foot, foot, foot)
    assert result["status"] == PASS


def test_v2_1_config_keeps_the_frozen_source_noise_budget_and_separate_protocol() -> None:
    config = RelativeRootTrunkConfig()
    config.validate()
    assert PROTOCOL_NAME == "vimogen_relative_root_trunk_v2_1_minimal_source_noise"
    assert config.iterations == 120
    assert config.step_rms == pytest.approx(0.01)
    assert config.line_search_steps == 8
    assert config.feasible_relative_mae_deg == pytest.approx(1.0)
    assert config.feasible_relative_p95_deg == pytest.approx(2.0)
    assert DifferentiableSamplerConfig().num_inference_steps == 50
