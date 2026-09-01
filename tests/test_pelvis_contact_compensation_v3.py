"""Unit tests for the frozen v3.0 pelvis/contact boundary."""

from __future__ import annotations

import math

import pytest
import torch

from evaluation.pelvis_contact_compensation_v3 import (
    FAIL,
    NOT_EVALUABLE,
    PASS,
    contact_evidence,
    evaluate_paired_foot,
    longest_true_run,
    pelvis_pitch_delta_deg,
    select_stable_window,
    target_root_rotation,
)
from motion_rep.rotation_transform import axis_angle_to_mat3x3


def _root_yaw_pitch(yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> torch.Tensor:
    base = axis_angle_to_mat3x3(torch.tensor([0.0, math.pi / 2.0, 0.0]))
    yaw = axis_angle_to_mat3x3(torch.tensor([0.0, 0.0, math.radians(yaw_deg)]))
    tilt = axis_angle_to_mat3x3(torch.tensor([math.radians(pitch_deg), 0.0, 0.0]))
    return yaw @ tilt @ base


def test_positive_dose_matches_v1_3_sign_and_exact_target() -> None:
    m0 = _root_yaw_pitch(37.0).reshape(1, 3, 3)
    candidate = target_root_rotation(m0, 10.0)
    observed = pelvis_pitch_delta_deg(m0, candidate)
    assert torch.allclose(observed, torch.full_like(observed, 10.0), atol=1e-4)


def test_fixed_m0_frame_is_invariant_to_yaw() -> None:
    m0 = _root_yaw_pitch(41.0, 8.0).reshape(1, 3, 3)
    candidate = target_root_rotation(m0, -5.0)
    assert float(pelvis_pitch_delta_deg(m0, candidate)[0]) == pytest.approx(-5.0, abs=1e-4)


def test_degenerate_vertical_forward_is_rejected() -> None:
    root = axis_angle_to_mat3x3(torch.tensor([0.0, 0.0, 0.0])).reshape(1, 3, 3)
    with pytest.raises(ValueError, match="horizontal root-forward"):
        target_root_rotation(root, 10.0)


def test_contact_confidence_excludes_first_frame_and_reports_pairs() -> None:
    heel = torch.zeros(8, 3)
    toe = torch.zeros(8, 3)
    heel[:, 0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2])
    toe[:, 0] = heel[:, 0]
    result = contact_evidence(heel, toe)
    assert result["valid_masks"]["general_contact"][0] is False
    assert result["contact_frames"] == 5
    assert result["continuous_contact_pairs"] == 4
    assert result["confidence"][0] == 0.0


def test_longest_window_ties_choose_earliest_run() -> None:
    mask = torch.tensor([False, True, True, False, True, True, False])
    assert longest_true_run(mask) == (1, 3)
    selected = select_stable_window(mask.float(), valid_mask=torch.ones(7, dtype=torch.bool), pad=1)
    assert selected["frames"] == [0, 1, 2, 3]


def test_paired_foot_equal_to_m0_passes_and_known_slide_fails() -> None:
    heel = torch.zeros(8, 3)
    toe = torch.zeros(8, 3)
    assert evaluate_paired_foot(heel, toe, heel, toe)["status"] == PASS
    candidate = heel.clone()
    candidate[1:, 0] = torch.arange(1, 8, dtype=torch.float32) * 0.02
    assert evaluate_paired_foot(heel, toe, candidate, candidate)["status"] == FAIL


def test_missing_flat_evidence_is_not_evaluable() -> None:
    heel = torch.zeros(3, 3)
    toe = heel.clone()
    result = evaluate_paired_foot(heel, toe, heel, toe)
    assert result["toe_contact"]["status"] == NOT_EVALUABLE
    assert result["status"] == NOT_EVALUABLE
