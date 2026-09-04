"""Tests for temporal-contact additions to the v0.2 projection protocol."""

from __future__ import annotations

import pytest
import torch

from sampling.pelvis_contact_flow_projection_v0_2 import (
    KINEMATIC_TEMPORAL_METRIC,
    PelvisContactFlowProjector,
    ProjectorConfig,
    PROTOCOL_NAME,
    solve_local_projection,
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

