from __future__ import annotations

import math

import pytest
import torch

from evaluation.local_pelvis_metrics import evaluate_local_pelvis_metrics
from motion_rep.phase1 import MOTION_LAYOUT, encode_rot6d


def _rx(degrees: float) -> torch.Tensor:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _motion() -> torch.Tensor:
    result = torch.zeros(1, 5, 276)
    identity = encode_rot6d(torch.eye(3))
    result[..., MOTION_LAYOUT.body_pose] = identity.repeat(21)
    result[..., MOTION_LAYOUT.root_rotation] = identity
    return result


def test_metrics_separate_relative_target_from_world_pelvis_motion() -> None:
    baseline = _motion()
    candidate = baseline.clone()
    candidate[..., 12:18] = encode_rot6d(_rx(5.0))
    metrics = evaluate_local_pelvis_metrics(
        candidate, baseline, torch.ones(1, 5, dtype=torch.bool), 5.0
    )
    row = metrics["per_sequence"][0]
    assert row["target_hit"]
    assert not row["world_pelvis_gate"]
    assert row["relative_angle_delta_mean_deg"] == pytest.approx(5.0, abs=1.0e-4)
    assert row["world_pelvis_delta_mean_deg"] == pytest.approx(0.0, abs=1.0e-4)


def test_zero_dose_identity_passes_non_contact_numeric_gates() -> None:
    baseline = _motion()
    metrics = evaluate_local_pelvis_metrics(
        baseline, baseline, torch.ones(1, 5, dtype=torch.bool), 0.0
    )
    row = metrics["per_sequence"][0]
    assert row["target_hit"]
    assert row["world_pelvis_gate"]
    assert row["trunk_gate"]
    assert row["root_trajectory_gate"]
    assert row["temporal_gate"]
