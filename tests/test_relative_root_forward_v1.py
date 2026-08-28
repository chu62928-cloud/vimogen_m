"""Unit coverage for the direct-pose authority root-forward protocol."""

import math

import pytest
import torch

pytest.importorskip("torch")

from motion_rep.consistency_v2 import SMPLX_22_PARENTS, differentiable_forward_kinematics
from motion_rep.finalize import finalize_motion
from motion_rep.phase1 import MOTION_LAYOUT, encode_rot6d, decode_rot6d_safe
from motion_rep.pose_authority import (
    PROTOCOL_NAME,
    authority_project,
    consistency_report,
    prepare_targets,
)
from motion_rep.rotation_transform import axis_angle_to_mat3x3
from sampling.relative_root_forward_guidance import (
    RelativeRootForwardConfig,
    RelativeRootForwardGuidance,
    apply_root_forward_tangent,
)
from sampling.relative_root_forward_guidance_v1_1 import (
    PROTOCOL_NAME as V1_1_PROTOCOL_NAME,
    ResidualAdaptiveRootForwardConfig,
    signed_root_forward_residual_deg,
)
from sampling.flow_sampler import FlowSampler


def _tiny_skeleton():
    offsets = torch.zeros(22, 3)
    offsets[0] = torch.tensor([0.1, -0.2, 0.3])
    offsets[1] = torch.tensor([0.2, 0.0, 0.0])
    offsets[2] = torch.tensor([-0.2, 0.0, 0.0])
    offsets[3] = torch.tensor([0.0, 0.0, 0.4])
    offsets[4:] = torch.tensor([0.0, 0.0, 0.2])
    from motion_rep.consistency_v2 import Skeleton22
    return Skeleton22(offsets, SMPLX_22_PARENTS, "test", offsets[0])


def _motion(frames=4):
    body = torch.eye(3).reshape(1, 1, 3, 3).repeat(frames + 1, 21, 1, 1)
    neutral = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    yaw = axis_angle_to_mat3x3(torch.stack((torch.zeros(frames + 1), torch.zeros(frames + 1), torch.linspace(0, .4, frames + 1)), -1))
    root = yaw @ neutral
    translation = torch.zeros(frames + 1, 3)
    translation[:, 1] = torch.arange(frames + 1) * .1
    fk = differentiable_forward_kinematics(body, root, translation, skeleton=_tiny_skeleton())
    return finalize_motion(body, fk.joints, root, translation).motion


def test_authority_ignores_redundant_channels_and_holds_hidden_pose():
    source = _motion(4).unsqueeze(0)
    source[..., MOTION_LAYOUT.joints] += 12.0
    source[..., MOTION_LAYOUT.joints_velocity] = 0.0
    source[..., MOTION_LAYOUT.root_rotation_velocity] = encode_rot6d(torch.eye(3))
    source[..., MOTION_LAYOUT.root_translation_velocity] = 0.0
    source[..., MOTION_LAYOUT.body_pose] += 0.001  # decode/re-encode is allowed numerical change
    mask = torch.tensor([[True, True, True, False]])
    result = authority_project(source, valid_mask=mask, skeleton=_tiny_skeleton())
    assert result.protocol == PROTOCOL_NAME
    assert result.valid_mask.tolist() == [[True, True, True, False]]
    # Last valid row uses an explicit held pose: its derived channels are 0/I/0.
    last = result.physical_motion[0, 2]
    assert torch.allclose(last[MOTION_LAYOUT.joints_velocity], torch.zeros(66), atol=1e-6)
    assert torch.allclose(
        decode_rot6d_safe(last[MOTION_LAYOUT.root_rotation_velocity]),
        torch.eye(3), atol=1e-5,
    )
    assert torch.allclose(last[MOTION_LAYOUT.root_translation_velocity], torch.zeros(3), atol=1e-6)
    assert consistency_report(result.physical_motion, mask, skeleton=_tiny_skeleton())[0]["passed"]


def test_target_forward_is_yaw_invariant_and_downward_positive():
    baseline = _motion(3).unsqueeze(0)
    mask = torch.ones(1, 3, dtype=torch.bool)
    projection = authority_project(baseline, valid_mask=mask, skeleton=_tiny_skeleton())
    targets = prepare_targets(projection.physical_motion, mask, 5.0)
    assert torch.allclose(targets.target_phi_deg, targets.phi0_deg - 5.0, atol=1e-5)
    assert torch.allclose(torch.linalg.vector_norm(targets.target_forward, dim=-1), torch.ones_like(targets.phi0_deg), atol=1e-5)


def test_zero_dose_returns_frozen_consistent_m0_exactly():
    baseline = _motion(3).unsqueeze(0)
    mask = torch.ones(1, 3, dtype=torch.bool)
    mean = torch.zeros(276)
    std = torch.ones(276)
    strategy = RelativeRootForwardGuidance(
        baseline_motion_norm=baseline,
        valid_mask=mask,
        mean=mean,
        std=std,
        target_delta_deg=0.0,
        config=RelativeRootForwardConfig(enabled=True),
        skeleton=_tiny_skeleton(),
    )
    outputs = strategy.finalize_outputs(torch.randn_like(baseline))
    assert torch.equal(outputs.g0, strategy.baseline_motion_norm)
    assert outputs.protocol == PROTOCOL_NAME
    assert float(strategy.forward_loss(strategy.baseline_motion_norm, target_forward=strategy.targets.f0)) < 1e-8
    assert float(strategy.tangent_gradient(strategy.baseline_motion_norm, target_forward=strategy.targets.f0).abs().max()) < 1e-8


def test_configuration_bounds_and_protocol_record():
    cfg = RelativeRootForwardConfig.from_mapping({"enabled": True, "target_delta_deg": 0})
    assert cfg.enabled and cfg.base_step_deg == 1.0
    with pytest.raises(ValueError):
        prepare_targets(_motion(2).unsqueeze(0), torch.ones(1, 2, dtype=torch.bool), 11.0)


def test_tangent_edit_is_left_multiplication_about_frozen_right_axis():
    root = _motion(2)[..., MOTION_LAYOUT.root_rotation].reshape(2, 6)
    matrix = decode_rot6d_safe(root)
    _, h, r, _ = __import__("motion_rep.pose_authority", fromlist=["_root_forward"])._root_forward(matrix)
    edited = apply_root_forward_tangent(matrix, r, torch.full((2,), -1.0))
    relative = edited @ matrix.transpose(-1, -2)
    axis_angle = __import__("motion_rep.rotation_transform", fromlist=["mat3x3_to_axis_angle"]).mat3x3_to_axis_angle(relative)
    cross = torch.cross(axis_angle, r, dim=-1)
    assert torch.linalg.vector_norm(cross, dim=-1).max() < 1e-5


def test_residual_adaptive_signed_residual_has_correct_sign_and_dose_scale():
    axis = torch.tensor([[[1.0, 0.0, 0.0]]])
    current = torch.tensor([[[0.0, 0.0, 1.0]]])
    target_5 = axis_angle_to_mat3x3(axis * (-5.0 * math.pi / 180.0)) @ current.unsqueeze(-1)
    target_10 = axis_angle_to_mat3x3(axis * (-10.0 * math.pi / 180.0)) @ current.unsqueeze(-1)
    target_5 = target_5.squeeze(-1)
    target_10 = target_10.squeeze(-1)
    mask = torch.ones(1, 1, dtype=torch.bool)
    residual_5, valid_5 = signed_root_forward_residual_deg(current, target_5, axis, mask)
    residual_10, valid_10 = signed_root_forward_residual_deg(current, target_10, axis, mask)
    assert valid_5.item() and valid_10.item()
    assert torch.allclose(residual_5, torch.tensor([[-5.0]]), atol=1e-4)
    assert torch.allclose(residual_10, torch.tensor([[-10.0]]), atol=1e-4)
    assert residual_10.abs().item() > residual_5.abs().item()


def test_residual_adaptive_config_isolated_protocol():
    cfg = ResidualAdaptiveRootForwardConfig.from_mapping({"enabled": True})
    assert cfg.protocol == V1_1_PROTOCOL_NAME
    assert cfg.residual_gain == 1.0
    assert cfg.max_step_deg == 8.0


def test_transfer_counterfactual_does_not_require_a_second_scheduler_step():
    x_next_guided = torch.zeros(1, 2, 276)
    velocity_model = torch.ones_like(x_next_guided)
    velocity_guided = torch.zeros_like(x_next_guided)
    recovered = FlowSampler._counterfactual_model_state(
        x_next_guided,
        velocity_model,
        velocity_guided,
        sigma=0.5,
        sigma_next=0.25,
    )
    assert recovered is not None
    assert torch.allclose(recovered, torch.full_like(recovered, -0.25))
    assert FlowSampler._counterfactual_model_state(
        x_next_guided,
        velocity_model,
        velocity_guided,
        sigma=0.5,
        sigma_next=0.25,
        stochastic_sampling=True,
    ) is None
