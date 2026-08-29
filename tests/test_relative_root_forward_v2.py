"""Unit tests for the minimal source-noise v2 boundary.

These tests are intentionally CPU-safe and are run dynamically on the
project server, like the rest of the ViMoGen test suite.
"""

from __future__ import annotations

import torch

from motion_rep.phase1 import MOTION_LAYOUT
from sampling.differentiable_flow_sampler import DifferentiableSamplerConfig
from sampling.relative_root_forward_guidance_v2 import (
    MinimalSourceNoiseConfig,
    _within_trust_region,
)


def test_v2_config_freezes_the_50_step_boundary_and_root_only_budget():
    config = MinimalSourceNoiseConfig()
    config.validate()
    sampler = DifferentiableSamplerConfig()
    assert sampler.num_inference_steps == 50
    assert config.feasible_pitch_mae_deg == 1.0
    assert config.feasible_forward_p95_deg == 2.0
    assert MOTION_LAYOUT.total_dim == 276


def test_source_delta_trust_region_uses_rms_not_l2():
    delta = torch.ones((1, 2, MOTION_LAYOUT.total_dim), dtype=torch.float32)
    projected = _within_trust_region(delta, max_rms=0.25)
    rms = torch.sqrt(projected.square().mean())
    assert torch.allclose(rms, torch.tensor(0.25))


def test_even_forward_loss_configuration_is_replaced_by_softmax_temperature():
    assert MinimalSourceNoiseConfig(forward_loss_temperature=5.0).forward_loss_temperature == 5.0
