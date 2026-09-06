"""Tests for the v0.4 terminal-projection offline ablation."""

from __future__ import annotations

import torch
import pytest

from scripts.evaluate_pelvis_guided_walk_terminal_ablation import (
    _endpoint_contact_metrics,
    _stats,
    _stats_delta,
)
from motion_rep.phase1 import MOTION_LAYOUT, encode_rot6d
from evaluation.pelvis_contact_compensation_v3 import target_root_rotation
from sampling.pelvis_contact_flow_projection_v0_1 import pelvis_target_error_summary


def _evidence() -> dict:
    return {
        "evidence": {
            "floor_height_m": 0.0,
            "valid_masks": {
                "general_contact": [False, True, True, True, True, True],
                "continuous_contact_pair": [False, True, True, True, True],
                "flat_contact": [False, True, True, True, True, True],
            },
        }
    }


def test_heel_and_toe_slip_are_reported_separately() -> None:
    vertices = torch.zeros((6, 2, 3), dtype=torch.float32)
    vertices[:, 0, 0] = torch.arange(6, dtype=torch.float32) * 0.001
    vertices[:, 1, 0] = torch.arange(6, dtype=torch.float32) * 0.003
    result = _endpoint_contact_metrics(
        vertices,
        {"heel": [0], "toe": [1], "sole": [0, 1]},
        _evidence(),
        torch.ones(6, dtype=torch.bool),
    )
    assert result["heel_slip_m_per_frame"]["p95"] == pytest.approx(1.0)
    assert result["toe_slip_m_per_frame"]["p95"] == pytest.approx(3.0)


def test_stats_delta_is_terminal_minus_pre_cast() -> None:
    pre = _stats(torch.tensor([1.0, 2.0, 3.0]))
    terminal = _stats(torch.tensor([2.0, 3.0, 4.0]))
    delta = _stats_delta(pre, terminal)
    assert delta["mean"] == 1.0
    assert delta["p95"] == 1.0
    assert delta["max"] == 1.0


def test_zero_endpoint_difference_produces_zero_delta() -> None:
    pre = _stats(torch.zeros(4))
    delta = _stats_delta(pre, pre)
    assert delta["mean"] == 0.0
    assert delta["p95"] == 0.0
    assert delta["max"] == 0.0


def test_pelvis_target_error_summary_uses_real_endpoint_root() -> None:
    horizontal_root = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    identity = horizontal_root.expand(4, -1, -1).clone()
    m0 = torch.zeros((4, MOTION_LAYOUT.total_dim), dtype=torch.float32)
    m0[:, MOTION_LAYOUT.root_rotation] = encode_rot6d(identity)
    candidate = m0.clone()
    candidate_root = target_root_rotation(identity, 2.0)
    candidate[:, MOTION_LAYOUT.root_rotation] = encode_rot6d(candidate_root)
    pre = pelvis_target_error_summary(m0, m0, torch.ones(4, dtype=torch.bool), 2.0)
    post = pelvis_target_error_summary(m0, candidate, torch.ones(4, dtype=torch.bool), 2.0)
    assert pre["mae_deg"] == pytest.approx(2.0, abs=1.0e-5)
    assert post["mae_deg"] == pytest.approx(0.0, abs=1.0e-5)
