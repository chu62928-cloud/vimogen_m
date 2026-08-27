"""Focused tests for the opt-in v2 absolute-mean guidance boundary."""

import math

import pytest
import torch

pytest.importorskip("torch")

from motion_rep.consistency_v2 import (  # noqa: E402
    default_smplx_neutral_22_skeleton,
    differentiable_forward_kinematics,
)
from motion_rep.finalize import finalize_motion  # noqa: E402
from motion_rep.phase1 import MOTION_LAYOUT  # noqa: E402
from motion_rep.sagittal_pelvis_angle import (  # noqa: E402
    apply_person_right_axis_rotation,
    pelvis_sagittal_tilt_degrees,
)
from motion_rep.rotation_transform import axis_angle_to_mat3x3  # noqa: E402
from sampling.absolute_mean_pelvis_guidance_v2 import (  # noqa: E402
    PROTOCOL_NAME,
    AbsoluteMeanPelvisConfigV2,
    AbsoluteMeanPelvisGuidanceV2,
    pelvis_angle_curve,
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
    # SMPL-X local +z points along canonical +y at zero yaw and tilt.
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=torch.float64,
    )


def _root(yaw: float, tilt_deg: float) -> torch.Tensor:
    return _rz(yaw) @ _rx(math.radians(tilt_deg)) @ _neutral_root()


def _packed_from_roots(roots: torch.Tensor, *, fake_joints: float = 37.0) -> torch.Tensor:
    """Pack a physical stream while deliberately corrupting the joint view."""

    frames_plus_one = roots.shape[0]
    body = torch.eye(3, dtype=roots.dtype).reshape(1, 1, 3, 3).repeat(
        frames_plus_one, 21, 1, 1
    )
    joints = torch.full((frames_plus_one, 22, 3), fake_joints, dtype=roots.dtype)
    translation = torch.zeros(frames_plus_one, 3, dtype=roots.dtype)
    translation[:, 1] = torch.arange(frames_plus_one, dtype=roots.dtype) * 0.01
    return finalize_motion(body, joints, roots, translation).motion


def _strategy(
    baseline: torch.Tensor,
    target: float,
    valid_mask: torch.Tensor | None = None,
    **overrides,
) -> AbsoluteMeanPelvisGuidanceV2:
    mask = (
        torch.ones(1, baseline.shape[0], dtype=torch.bool)
        if valid_mask is None
        else valid_mask.reshape(1, -1)
    )
    config = AbsoluteMeanPelvisConfigV2(fusion_window=1, **overrides)
    return AbsoluteMeanPelvisGuidanceV2(
        baseline_motion_norm=baseline.unsqueeze(0),
        valid_mask=mask,
        mean=torch.zeros(276),
        std=torch.ones(276),
        target_mean_deg=target,
        config=config,
    )


def test_v2_reconcile_replaces_fake_joints_and_repacks_all_channels():
    roots = torch.eye(3, dtype=torch.float32).repeat(5, 1, 1)
    baseline = _packed_from_roots(roots)
    strategy = _strategy(baseline, target=0.0)
    result = strategy._reconcile(baseline.unsqueeze(0), output_standardized=False)

    body = torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3).repeat(5, 21, 1, 1)
    translation = torch.zeros(5, 3)
    translation[:, 1] = torch.arange(5) * 0.01
    skeleton = default_smplx_neutral_22_skeleton()
    expected_fk = differentiable_forward_kinematics(
        body, roots, translation, skeleton=skeleton
    )
    expected = finalize_motion(
        body, expected_fk.joints, roots, translation
    ).motion

    torch.testing.assert_close(result.motion[0], expected, atol=2e-5, rtol=0)
    assert not torch.allclose(
        result.motion[0, :, MOTION_LAYOUT.joints],
        baseline[:, MOTION_LAYOUT.joints],
    )
    joints_velocity = result.motion[0, :, MOTION_LAYOUT.joints_velocity].reshape(
        -1, 22, 3
    )
    joints = result.motion[0, :, MOTION_LAYOUT.joints].reshape(-1, 22, 3)
    torch.testing.assert_close(
        joints_velocity, expected_fk.joints[1:] - expected_fk.joints[:-1], atol=2e-5, rtol=0
    )


def test_v2_guidance_angle_and_g1_are_yaw_invariant_with_full_fk_repack():
    yaws = (0.0, math.pi / 2, math.pi, math.pi)
    roots = torch.stack([_root(yaw, 4.5) for yaw in yaws]).float()
    values = pelvis_sagittal_tilt_degrees(roots)
    torch.testing.assert_close(
        values, torch.full_like(values, 4.5), atol=2e-5, rtol=0
    )

    baseline = _packed_from_roots(roots)
    strategy = _strategy(baseline, target=5.0, terminal_enabled=True)
    outputs = strategy.finalize_outputs(baseline.unsqueeze(0))
    record = outputs.terminal_records[0]
    assert record["eligible"] and record["triggered"]
    assert record["applied_deg"] == pytest.approx(0.5, abs=0.01)
    assert outputs.g1_valid_mask[0].tolist() == [True, True, True]

    g1_result = strategy._reconcile(outputs.g1, output_standardized=False)
    g1_physical = g1_result.motion[0]
    assert g1_result.valid_mask[0].tolist() == [True, True, True]
    assert float(pelvis_angle_curve(g1_physical).mean()) == pytest.approx(5.0, abs=0.02)
    # A v2 endpoint must already be a fixed point of the complete repacker.
    torch.testing.assert_close(g1_physical, outputs.g1[0], atol=3e-4, rtol=0)


def test_v2_g1_does_not_correct_residual_over_one_degree():
    roots = torch.stack([_root(0.0, 3.5) for _ in range(5)]).float()
    baseline = _packed_from_roots(roots)
    strategy = _strategy(baseline, target=5.0, terminal_enabled=True)
    outputs = strategy.finalize_outputs(baseline.unsqueeze(0))
    record = outputs.terminal_records[0]
    assert record["failed_residual_over_limit"]
    assert not record["triggered"]
    torch.testing.assert_close(outputs.g1, outputs.g0, atol=0, rtol=0)


def test_v2_variable_length_mask_keeps_last_valid_row_and_hidden_pose():
    roots = torch.stack(
        [_root(yaw, 4.5) for yaw in (0.0, math.pi / 2, math.pi, math.pi, math.pi, math.pi, math.pi)]
    ).float()
    baseline = _packed_from_roots(roots)
    row_mask = torch.tensor([True, True, True, False, False, False])
    strategy = _strategy(baseline, target=5.0, valid_mask=row_mask)
    outputs = strategy.finalize_outputs(baseline.unsqueeze(0))

    expected_mask = [True, True, True, False, False, False]
    assert outputs.g0_valid_mask[0].tolist() == expected_mask
    assert outputs.g1_valid_mask[0].tolist() == expected_mask
    assert outputs.terminal_records[0]["triggered"]
    assert torch.isfinite(outputs.g1[0, :3]).all()


def test_v2_gradient_guidance_path_is_finite():
    roots = torch.stack([_root(0.0, tilt) for tilt in (3.0, 3.5, 4.0, 4.5, 5.0)]).float()
    baseline = _packed_from_roots(roots)
    strategy = _strategy(
        baseline,
        target=6.0,
        guidance_strength=1.0,
        max_correction_rms=0.05,
        sigma_min=0.25,
        sigma_max=0.65,
        terminal_enabled=False,
    )
    corrected, diagnostics = strategy.correct_velocity(
        x_sigma=baseline.unsqueeze(0),
        velocity=torch.zeros(1, baseline.shape[0], 276),
        sigma=torch.tensor(0.5),
        valid_mask=strategy.valid_mask,
        return_trace=True,
    )
    assert diagnostics["active"]
    assert math.isfinite(diagnostics["gradient_rms"])
    assert torch.isfinite(corrected).all()
    assert torch.isfinite(diagnostics["trace"]["x0_reconciled"]).all()


def test_v2_protocol_is_distinct_from_v1():
    assert PROTOCOL_NAME == "vimogen_absolute_mean_pelvis_v2_full_fk_sagittal"
    assert PROTOCOL_NAME.endswith("full_fk_sagittal")

