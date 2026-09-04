"""Tests for temporal-contact additions to the v0.2 projection protocol."""

from __future__ import annotations

import pytest
import torch

from scripts.evaluate_pelvis_contact_flow_projection_v0_1 import (
    _paired_status,
    m0_pairing_eligible,
)
from sampling.pelvis_contact_flow_projection_v0_2 import (
    KINEMATIC_TEMPORAL_METRIC,
    PelvisContactFlowProjector,
    ProjectorConfig,
    PROTOCOL_NAME,
    solve_local_projection,
    temporal_contact_residual,
)


def test_temporal_protocol_requires_velocity_weight_and_halo() -> None:
    with pytest.raises(ValueError, match="velocity weight"):
        ProjectorConfig(protocol=PROTOCOL_NAME, boundary_halo_frames=1).validate()
    with pytest.raises(ValueError, match="halo"):
        ProjectorConfig(
            protocol=PROTOCOL_NAME,
            contact_velocity_weight=1.0,
            boundary_halo_frames=0,
        ).validate()
    config = ProjectorConfig(
        protocol=PROTOCOL_NAME,
        contact_velocity_weight=1.0e6,
        boundary_halo_frames=1,
    )
    config.validate()


def test_temporal_mapping_uses_strict_defaults() -> None:
    config = ProjectorConfig.from_mapping({"protocol": PROTOCOL_NAME})
    assert config.contact_velocity_weight == pytest.approx(1.0e6)
    assert config.boundary_halo_frames == 1


def test_m0_mismatch_and_not_evaluable_cannot_unlock_formal_pass() -> None:
    assert not m0_pairing_eligible("MISMATCH_ALLOWED")
    assert not m0_pairing_eligible("INELIGIBLE_M0_MISMATCH")
    baseline = {"count": 2, "mean": 0.0, "p95": 0.0, "max": 0.0}
    assert _paired_status(baseline, baseline) == "NOT_EVALUABLE"


def test_temporal_contact_residual_is_zero_for_m0_and_detects_foot_shift() -> None:
    m0 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=torch.float64,
    )
    pairs = torch.tensor([True, True])
    assert torch.equal(temporal_contact_residual(m0, m0, pairs), torch.zeros((2, 3), dtype=torch.float64))
    candidate = m0.clone()
    candidate[2, 2] += 0.01
    residual = temporal_contact_residual(candidate, m0, pairs)
    assert float(residual[-1, 2]) == pytest.approx(0.01)


def test_merit_rejects_velocity_regression_even_when_position_improves() -> None:
    before = {
        "pelvis": 2.0,
        "heel_position": 2.0,
        "toe_position": 2.0,
        "heel_velocity": 0.8,
        "toe_velocity": 0.8,
        "penetration": 0.0,
    }
    after = {
        "pelvis": 0.5,
        "heel_position": 0.5,
        "toe_position": 0.5,
        "heel_velocity": 1.01,
        "toe_velocity": 1.01,
        "penetration": 0.0,
    }
    assert not PelvisContactFlowProjector._accept_merit(before, after)


def test_weighted_contact_rows_change_solution_without_changing_api() -> None:
    metric = torch.eye(2, dtype=torch.float64)
    empty = torch.zeros((0, 2), dtype=torch.float64)
    contact = torch.eye(2, dtype=torch.float64)
    target = torch.ones(2, dtype=torch.float64)
    unweighted, _ = solve_local_projection(
        metric,
        empty,
        torch.zeros(0, dtype=torch.float64),
        contact,
        target,
        empty,
        torch.zeros(0, dtype=torch.float64),
        contact_weight=1.0,
    )
    weighted, _ = solve_local_projection(
        metric,
        empty,
        torch.zeros(0, dtype=torch.float64),
        contact,
        target,
        empty,
        torch.zeros(0, dtype=torch.float64),
        contact_weight=1.0,
        contact_row_weights=torch.tensor([1.0, 0.01], dtype=torch.float64),
    )
    assert torch.allclose(unweighted, torch.ones(2, dtype=torch.float64) / 2.0)
    assert weighted[0] > weighted[1]
    assert not torch.allclose(unweighted, weighted)
