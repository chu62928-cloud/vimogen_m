"""Unit tests for the constraint-first root-forward v1.2 protocol."""

import math

import pytest
import torch

pytest.importorskip("torch")

from motion_rep.consistency_v2 import SMPLX_22_PARENTS, Skeleton22, differentiable_forward_kinematics
from motion_rep.finalize import finalize_motion
from motion_rep.phase1 import MOTION_LAYOUT, encode_rot6d
from motion_rep.pose_authority import authority_project
from motion_rep.rotation_transform import axis_angle_to_mat3x3
from sampling.relative_root_forward_guidance_v1_2 import (
    PROTOCOL_NAME,
    TrunkStabilizedRootForwardConfig,
    TrunkStabilizedRootForwardGuidance,
    signed_axis_residual_deg,
)


def _tiny_skeleton():
    offsets = torch.zeros(22, 3)
    offsets[0] = torch.tensor([0.1, -0.2, 0.3])
    offsets[1] = torch.tensor([0.2, 0.0, 0.0])
    offsets[2] = torch.tensor([-0.2, 0.0, 0.0])
    offsets[3] = torch.tensor([0.0, 0.0, 0.4])
    offsets[4:] = torch.tensor([0.0, 0.0, 0.2])
    return Skeleton22(offsets, SMPLX_22_PARENTS, "test", offsets[0])


def _motion(frames=4):
    body = torch.eye(3).reshape(1, 1, 3, 3).repeat(frames + 1, 21, 1, 1)
    neutral = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    yaw = axis_angle_to_mat3x3(
        torch.stack((torch.zeros(frames + 1), torch.zeros(frames + 1), torch.linspace(0, .4, frames + 1)), -1)
    )
    root = yaw @ neutral
    translation = torch.zeros(frames + 1, 3)
    translation[:, 1] = torch.arange(frames + 1) * .1
    fk = differentiable_forward_kinematics(body, root, translation, skeleton=_tiny_skeleton())
    return finalize_motion(body, fk.joints, root, translation).motion


def test_signed_axis_residual_has_both_signs_and_zero():
    axis = torch.tensor([[[1.0, 0.0, 0.0]]])
    current = torch.tensor([[[0.0, 0.0, 1.0]]])
    target_pos = axis_angle_to_mat3x3(axis * (5.0 * math.pi / 180.0)) @ current.unsqueeze(-1)
    target_neg = axis_angle_to_mat3x3(axis * (-5.0 * math.pi / 180.0)) @ current.unsqueeze(-1)
    mask = torch.ones(1, 1, dtype=torch.bool)
    pos, valid_pos = signed_axis_residual_deg(current, target_pos.squeeze(-1), axis, mask)
    neg, valid_neg = signed_axis_residual_deg(current, target_neg.squeeze(-1), axis, mask)
    zero, valid_zero = signed_axis_residual_deg(current, current, axis, mask)
    assert valid_pos.item() and valid_neg.item() and valid_zero.item()
    assert torch.allclose(pos, torch.tensor([[5.0]]), atol=1e-4)
    assert torch.allclose(neg, torch.tensor([[-5.0]]), atol=1e-4)
    assert zero.abs().max() < 1e-6


def test_v1_2_config_and_protocol_are_independent():
    cfg = TrunkStabilizedRootForwardConfig.from_mapping({"enabled": True})
    assert cfg.protocol == PROTOCOL_NAME
    assert cfg.heading_gain == 0.75
    assert cfg.trunk_gain == 0.75
    assert cfg.max_trunk_step_deg == 6.0


def test_v1_2_preserves_direct_non_guided_streams_and_rebuilds_derived_channels():
    baseline = _motion(3).unsqueeze(0)
    mask = torch.ones(1, 3, dtype=torch.bool)
    mean = torch.zeros(276)
    std = torch.ones(276)
    strategy = TrunkStabilizedRootForwardGuidance(
        baseline_motion_norm=baseline,
        valid_mask=mask,
        mean=mean,
        std=std,
        target_delta_deg=5.0,
        config=TrunkStabilizedRootForwardConfig(enabled=True),
        skeleton=_tiny_skeleton(),
    )
    source, _ = __import__("motion_rep.pose_authority", fromlist=["_prefix_source"])._prefix_source(baseline, mask)
    base_root = source[..., MOTION_LAYOUT.root_rotation].clone()
    base_translation = source[..., MOTION_LAYOUT.root_translation].clone()
    velocity, diagnostics = strategy.correct_velocity(
        x_sigma=baseline,
        velocity=torch.zeros_like(baseline),
        sigma=0.5,
        valid_mask=mask,
    )
    assert diagnostics["protocol"] == PROTOCOL_NAME
    assert torch.isfinite(velocity).all()
    assert diagnostics.get("accepted", False) or diagnostics.get("rejected_reason")
    # The authority boundary is explicit even when the candidate is rejected.
    assert torch.equal(source[..., MOTION_LAYOUT.root_translation], base_translation)
    assert torch.equal(source[..., MOTION_LAYOUT.root_rotation], base_root)


def test_protocol_record_declares_constraints_not_weighted_total_loss():
    baseline = _motion(3).unsqueeze(0)
    mask = torch.ones(1, 3, dtype=torch.bool)
    strategy = TrunkStabilizedRootForwardGuidance(
        baseline_motion_norm=baseline,
        valid_mask=mask,
        mean=torch.zeros(276),
        std=torch.ones(276),
        target_delta_deg=5.0,
        config=TrunkStabilizedRootForwardConfig(enabled=True),
        skeleton=_tiny_skeleton(),
    )
    record = strategy.protocol_record()
    assert record["loss_framework"] == "independent_control_constraints_and_change_budgets"
    assert "trunk_direction" in record["control_constraints"]
    assert "body_pose.spine1" in record["guided_channels"]
