from __future__ import annotations

import pytest
import torch

from motion_rep.phase1 import MOTION_LAYOUT
from scripts.compare_guided_pelvis_candidates import _difference_by_section, _physical, _ratio


def test_physical_denormalisation_respects_each_batch_row() -> None:
    normalized = torch.ones((2, 3, MOTION_LAYOUT.total_dim))
    mean = torch.stack((torch.zeros(MOTION_LAYOUT.total_dim), torch.ones(MOTION_LAYOUT.total_dim)))
    std = torch.stack((torch.ones(MOTION_LAYOUT.total_dim), torch.full((MOTION_LAYOUT.total_dim,), 2.0)))
    result = _physical(normalized, mean, std)
    assert torch.allclose(result[0], torch.ones_like(result[0]))
    assert torch.allclose(result[1], torch.full_like(result[1], 3.0))


def test_difference_report_keeps_direct_and_derived_sections_separate() -> None:
    left = torch.zeros((1, 2, MOTION_LAYOUT.total_dim))
    right = left.clone()
    right[..., MOTION_LAYOUT.root_translation] = 0.25
    report = _difference_by_section(left, right)
    assert report["root_translation"]["max_abs"] == pytest.approx(0.25)
    assert report["body_pose"]["max_abs"] == pytest.approx(0.0)
    assert report["joints"]["max_abs"] == pytest.approx(0.0)


def test_ratio_handles_zero_baseline_without_non_finite_output() -> None:
    assert _ratio(2.0, 1.0) == pytest.approx(2.0)
    assert _ratio(2.0, 0.0) is None
    assert _ratio(None, 1.0) is None
