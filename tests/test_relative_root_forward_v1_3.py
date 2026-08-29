"""Unit tests for the v1.3 shadow-pose hierarchical protocol."""

import torch
import pytest

pytest.importorskip("torch")

from motion_rep.consistency_v2 import SMPLX_22_PARENTS, Skeleton22, differentiable_forward_kinematics
from motion_rep.finalize import finalize_motion
from motion_rep.phase1 import MOTION_LAYOUT, encode_rot6d
from motion_rep.rotation_transform import axis_angle_to_mat3x3
from sampling.relative_root_forward_guidance_v1_3 import (
    PROTOCOL_NAME,
    ShadowPoseHierarchicalConfig,
    ShadowPoseHierarchicalRootForwardGuidance,
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


def _strategy(target=5.0, frames=3):
    baseline = _motion(frames).unsqueeze(0)
    return ShadowPoseHierarchicalRootForwardGuidance(
        baseline_motion_norm=baseline,
        valid_mask=torch.ones(1, frames, dtype=torch.bool),
        mean=torch.zeros(276),
        std=torch.ones(276),
        target_delta_deg=target,
        config=ShadowPoseHierarchicalConfig(enabled=True, sigma_min=0.0),
        skeleton=_tiny_skeleton(),
    )


def test_v1_3_config_and_protocol_are_explicit():
    cfg = ShadowPoseHierarchicalConfig.from_mapping({"enabled": True})
    assert cfg.protocol == PROTOCOL_NAME
    assert cfg.max_solver_iterations == 4
    assert cfg.trunk_envelope_deg == 2.0
    assert cfg.jacobian_damping > 0
    assert cfg.motion_weight == 0.0


def test_only_root_and_three_spines_are_model_injection_channels():
    strategy = _strategy()
    mask = strategy._active_direct_mask(torch.device("cpu"))
    assert mask[MOTION_LAYOUT.root_rotation].all()
    for body_index in (2, 5, 8):
        assert mask[body_index * 6:(body_index + 1) * 6].all()
    assert not mask[MOTION_LAYOUT.joints].any()
    assert not mask[MOTION_LAYOUT.joints_velocity].any()
    assert not mask[MOTION_LAYOUT.root_rotation_velocity].any()
    assert not mask[MOTION_LAYOUT.root_translation].any()
    assert not mask[MOTION_LAYOUT.root_translation_velocity].any()


def test_shadow_endpoint_leaves_derived_channels_from_x0_hat_untouched():
    strategy = _strategy()
    x0_hat = strategy.baseline_motion_norm.clone()
    x0_physical = strategy._physical(x0_hat)
    mask = strategy.valid_mask
    candidate = strategy._direct_candidate_norm(
        x0_hat,
        x0_physical,
        torch.ones_like(mask, dtype=torch.float32),
        torch.zeros_like(mask, dtype=torch.float32),
        torch.zeros((*mask.shape, 3), dtype=torch.float32),
        mask,
    )
    active = strategy._active_direct_mask(candidate.device).view(1, 1, -1).expand_as(candidate)
    assert torch.equal(candidate[~active], x0_hat[~active])
    assert (candidate[active] - x0_hat[active]).abs().max() > 1e-6


def test_temporal_projection_limits_vector_step_and_masks_padding():
    strategy = _strategy()
    values = torch.zeros(1, 5, 5)
    values[0, :, 0] = torch.tensor([0., 5., 10., 2., 8.])
    mask = torch.tensor([[True, True, True, False, False]])
    projected = strategy._temporal_project(values, mask)
    assert torch.equal(projected[0, 3:], torch.zeros(2, 5))
    diffs = projected[:, 1:] - projected[:, :-1]
    pair = mask[:, 1:] & mask[:, :-1]
    assert float(torch.linalg.vector_norm(diffs[pair], dim=-1).max()) <= 2.0 + 1e-5


def test_zero_dose_returns_frozen_m0_bitwise():
    strategy = _strategy(target=0.0)
    outputs = strategy.finalize_outputs(torch.randn_like(strategy.baseline_motion_norm))
    assert torch.equal(outputs.g0, strategy.baseline_motion_norm)
    assert outputs.protocol == PROTOCOL_NAME


def test_protocol_record_declares_shadow_boundary_and_no_mixed_loss():
    record = _strategy().protocol_record()
    assert record["protocol"] == PROTOCOL_NAME
    assert record["state_boundary"] == "model_state_plus_physical_shadow_state"
    assert record["loss_framework"] == "separate_angular_constraints_with_physical_trust_regions"
    assert "dJ" in record["shadow_only_channels"]
