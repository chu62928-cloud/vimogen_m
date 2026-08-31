"""Unit tests for the minimal source-noise v2 boundary.

These tests are intentionally CPU-safe and are run dynamically on the
project server, like the rest of the ViMoGen test suite.
"""

from __future__ import annotations

import torch
import pytest

from motion_rep.phase1 import MOTION_LAYOUT
from sampling.differentiable_flow_sampler import DifferentiableSamplerConfig
from sampling.relative_root_forward_guidance_v2 import (
    MinimalSourceNoiseConfig,
    _within_trust_region,
    select_source_noise_output,
)
from scripts.evaluate_relative_root_forward_v2_naturalness import (
    _allowed_increase,
    _metric_pass,
)


def test_v2_config_freezes_the_50_step_boundary_and_root_only_budget():
    config = MinimalSourceNoiseConfig()
    config.validate()
    sampler = DifferentiableSamplerConfig()
    assert sampler.num_inference_steps == 50
    assert config.feasible_pitch_mae_deg == 1.0
    assert config.feasible_forward_p95_deg == 2.0
    assert config.iterations == 120
    assert config.step_rms == 0.01
    assert config.line_search_steps == 8
    assert MOTION_LAYOUT.total_dim == 276


def test_source_delta_trust_region_uses_rms_not_l2():
    delta = torch.ones((1, 2, MOTION_LAYOUT.total_dim), dtype=torch.float32)
    projected = _within_trust_region(delta, max_rms=0.25)
    rms = torch.sqrt(projected.square().mean())
    assert torch.allclose(rms, torch.tensor(0.25))


def test_even_forward_loss_configuration_is_replaced_by_softmax_temperature():
    assert MinimalSourceNoiseConfig(forward_loss_temperature=5.0).forward_loss_temperature == 5.0


class _MotionResult:
    def __init__(self, motion_norm: torch.Tensor):
        self.motion_norm = motion_norm


def test_source_noise_result_is_not_overwritten_by_m0_fallback():
    baseline = torch.zeros((1, 2, MOTION_LAYOUT.total_dim))
    guided = torch.ones_like(baseline)
    selected = select_source_noise_output(baseline, _MotionResult(guided))
    assert torch.equal(selected, guided)


def test_infeasible_source_noise_fallback_has_zero_delta():
    baseline = torch.zeros((1, 2, MOTION_LAYOUT.total_dim))
    selected, delta = select_source_noise_output(baseline, None, return_delta=True)
    assert torch.equal(selected, baseline)
    assert torch.count_nonzero(delta) == 0


def test_source_noise_route_keeps_selected_motion_and_delta_paired():
    baseline = torch.zeros((1, 2, MOTION_LAYOUT.total_dim))
    guided = torch.ones_like(baseline)
    delta = torch.full_like(baseline, 0.25)
    result = _MotionResult(guided)
    result.source_delta = delta
    selected, selected_delta = select_source_noise_output(
        baseline, result, return_delta=True
    )
    assert torch.equal(selected, guided)
    assert torch.equal(selected_delta, delta)


def test_trust_region_masks_padded_source_frames():
    delta = torch.ones((1, 2, MOTION_LAYOUT.total_dim), dtype=torch.float32)
    mask = torch.tensor([[True, False]])
    projected = _within_trust_region(delta, max_rms=0.25, valid_mask=mask)
    assert torch.count_nonzero(projected[:, 1]) == 0
    assert torch.allclose(
        torch.sqrt(projected[:, :1].square().mean()), torch.tensor(0.25)
    )


def test_naturalness_gate_uses_five_percent_or_one_millimeter_tolerance():
    assert _allowed_increase(0.1) == pytest.approx(0.005)
    assert _allowed_increase(0.0) == pytest.approx(0.001)
    assert _metric_pass({"mean": 0.104, "p95": 0.204}, {"mean": 0.1, "p95": 0.2})
    assert not _metric_pass({"mean": 0.106, "p95": 0.2}, {"mean": 0.1, "p95": 0.2})
