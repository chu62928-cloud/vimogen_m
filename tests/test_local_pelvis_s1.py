from __future__ import annotations

import math

import torch

from geometry.local_pelvis import (
    minimum_norm_relative_projection,
    relative_angle_error_deg,
    relative_pelvis_angle_curve_deg,
    target_relative_angle_curve_deg,
    target_world_pelvis_angle_curve_deg,
    world_pelvis_angle_error_deg,
)
from guidance.base import ConstraintPack, GuidanceRequest, SharedEvidence
from guidance.m7_paht_edit import M7PAHTGeometricEdit
from motion_rep.phase1 import MOTION_LAYOUT, encode_rot6d


def _rotation_x(degrees: float) -> torch.Tensor:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _identity_motion(frames: int = 4) -> torch.Tensor:
    motion = torch.zeros(1, frames, 276)
    identity = encode_rot6d(torch.eye(3))
    motion[..., MOTION_LAYOUT.body_pose] = identity.repeat(21)
    motion[..., MOTION_LAYOUT.root_rotation] = identity
    return motion


def test_relative_angle_is_invariant_to_whole_body_root_rotation() -> None:
    motion = _identity_motion()
    before = relative_pelvis_angle_curve_deg(motion)
    rotated = motion.clone()
    rotated[..., MOTION_LAYOUT.root_rotation] = encode_rot6d(_rotation_x(27.0))
    after = relative_pelvis_angle_curve_deg(rotated)
    torch.testing.assert_close(after, before, atol=1.0e-5, rtol=0.0)


def test_positive_spine_rotation_produces_positive_relative_pelvis_dose() -> None:
    motion = _identity_motion()
    spine1 = slice(12, 18)
    motion[..., spine1] = encode_rot6d(_rotation_x(5.0))
    actual = relative_pelvis_angle_curve_deg(motion)
    torch.testing.assert_close(actual, torch.full_like(actual, 5.0), atol=1.0e-4, rtol=0.0)


def test_pure_root_gradient_is_zero_but_spine_gradient_is_nonzero() -> None:
    motion = _identity_motion().requires_grad_(True)
    relative_pelvis_angle_curve_deg(motion).sum().backward()
    assert motion.grad is not None
    assert float(motion.grad[..., MOTION_LAYOUT.root_rotation].abs().max()) < 1.0e-4
    assert float(motion.grad[..., MOTION_LAYOUT.body_pose].abs().max()) > 1.0


def test_minimum_norm_projection_moves_toward_target() -> None:
    motion = _identity_motion()
    valid = torch.ones(1, motion.shape[1], dtype=torch.bool)
    target = target_relative_angle_curve_deg(motion, 5.0)
    projected, audit = minimum_norm_relative_projection(
        motion, target, valid, damping=1.0e-5, max_step_deg=2.0
    )
    actual = relative_pelvis_angle_curve_deg(projected)
    assert float(actual.mean()) > 1.5
    assert audit["jacobian_rank"] == motion.shape[1]
    assert audit["realized_step_rms_deg"] > 1.5


def test_target_curve_is_paired_to_baseline() -> None:
    motion = _identity_motion()
    motion[..., 12:18] = encode_rot6d(_rotation_x(-3.0))
    target = target_relative_angle_curve_deg(motion, 5.0)
    torch.testing.assert_close(target, torch.full_like(target, 2.0), atol=1.0e-4, rtol=0.0)


def test_s2_m7_hits_relative_and_world_targets() -> None:
    baseline = _identity_motion()
    valid = torch.ones(baseline.shape[:2], dtype=torch.bool)
    relative_target = target_relative_angle_curve_deg(baseline, 5.0)
    world_target = target_world_pelvis_angle_curve_deg(baseline, 5.0)
    request = GuidanceRequest(
        prompt_id="fixture",
        seed=0,
        target_dose_deg=5.0,
        constraint_pack=ConstraintPack.S2_RELATIVE_WORLD,
        base_noise=torch.zeros_like(baseline),
        baseline_motion=baseline,
        shared_evidence=SharedEvidence(
            valid_mask=valid,
            target_angle_curve_deg=relative_target,
            target_world_pelvis_curve_deg=world_target,
        ),
    )
    result = M7PAHTGeometricEdit().run(
        None,
        request,
        {"iterations": 20, "max_step_deg": 1.0, "damping": 1.0e-5},
    )
    assert result.status == "COMPLETED"
    assert relative_angle_error_deg(result.motion, relative_target).abs().max() < 1.0e-3
    assert world_pelvis_angle_error_deg(result.motion, world_target).abs().max() < 1.0e-3
