"""Public seam tests for the v0.4 dose-first walk ablation."""

from __future__ import annotations

import json

import pytest
import torch

from sampling.pelvis_contact_flow_projection_v0_1 import (
    DOSE_FIRST_CONTACT_ABLATION_PROTOCOL,
    DOSE_FIRST_CONTACT_MODES,
    PelvisContactFlowProjector,
    ProjectorConfig,
    solve_local_projection,
)


def test_v04_protocol_accepts_zero_contact_weights_and_full_sequence() -> None:
    config = ProjectorConfig(
        protocol=DOSE_FIRST_CONTACT_ABLATION_PROTOCOL,
        projection_scope="full_sequence",
        contact_mode="dose_only",
        contact_position_weight=0.0,
        contact_velocity_weight=0.0,
    )
    config.validate()
    assert config.projection_scope == "full_sequence"
    assert set(DOSE_FIRST_CONTACT_MODES) == {
        "dose_only",
        "position_only_medium",
        "temporal_weak",
        "temporal_medium",
        "temporal_strong",
    }


def test_v04_separate_position_and_velocity_weights_change_normal_equations() -> None:
    metric = torch.eye(2, dtype=torch.float64)
    empty = torch.zeros((0, 2), dtype=torch.float64)
    contact = torch.eye(2, dtype=torch.float64)
    target = torch.ones(2, dtype=torch.float64)
    low_velocity, _ = solve_local_projection(
        metric,
        empty,
        torch.zeros(0, dtype=torch.float64),
        contact,
        target,
        empty,
        torch.zeros(0, dtype=torch.float64),
        contact_weight=1.0,
        contact_row_weights=torch.ones(2, dtype=torch.float64),
        contact_row_scales=torch.tensor([1.0e5, 1.0e4], dtype=torch.float64),
    )
    high_velocity, _ = solve_local_projection(
        metric,
        empty,
        torch.zeros(0, dtype=torch.float64),
        contact,
        target,
        empty,
        torch.zeros(0, dtype=torch.float64),
        contact_weight=1.0,
        contact_row_weights=torch.ones(2, dtype=torch.float64),
        contact_row_scales=torch.tensor([1.0e5, 1.0e6], dtype=torch.float64),
    )
    assert not torch.allclose(low_velocity, high_velocity)


def test_v04_dose_first_acceptance_prioritises_pelvis_and_penetration() -> None:
    before = {
        "pelvis": 2.0,
        "heel_position": 0.2,
        "toe_position": 0.2,
        "heel_velocity": 0.2,
        "toe_velocity": 0.2,
        "penetration": 0.0,
    }
    after = {
        "pelvis": 0.5,
        "heel_position": 1.5,
        "toe_position": 1.5,
        "heel_velocity": 1.5,
        "toe_velocity": 1.5,
        "penetration": 0.5,
    }
    assert PelvisContactFlowProjector._accept_dose_first(before, after)
    after_new_penetration = dict(after, penetration=1.1)
    assert not PelvisContactFlowProjector._accept_dose_first(
        before, after_new_penetration
    )


def test_v04_zero_contact_rows_match_empty_contact_system() -> None:
    metric = torch.eye(2, dtype=torch.float64)
    empty = torch.zeros((0, 2), dtype=torch.float64)
    zero_rows, _ = solve_local_projection(
        metric,
        empty,
        torch.zeros(0, dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        empty,
        torch.zeros(0, dtype=torch.float64),
        contact_weight=1.0,
        contact_row_scales=torch.zeros(2, dtype=torch.float64),
    )
    no_rows, _ = solve_local_projection(
        metric,
        empty,
        torch.zeros(0, dtype=torch.float64),
        empty,
        torch.zeros(0, dtype=torch.float64),
        empty,
        torch.zeros(0, dtype=torch.float64),
        contact_weight=1.0,
    )
    assert torch.allclose(zero_rows, no_rows, atol=1.0e-10)


def test_v04_strict_json_rejects_non_finite_values(tmp_path) -> None:
    from sampling.pelvis_contact_flow_projection_v0_1 import write_strict_json

    output = tmp_path / "record.json"
    write_strict_json(output, {"status": "PASS", "value": 1.0})
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"
    with pytest.raises(ValueError):
        write_strict_json(output, {"value": float("inf")})

