"""Tests for the v0.4.2 lower-body terminal prototype."""

from __future__ import annotations

import pytest
import torch

from sampling.lower_body_terminal_compensation import LowerBodyDofMap, LowerBodySolverConfig
from sampling.lower_body_terminal_compensation import (
    _finite_difference_jacobian,
    _marker_residual_stats,
    _position_marker_indices,
)


def test_default_lower_body_map_uses_anatomical_subspace() -> None:
    mapping = LowerBodyDofMap.default()
    mapping.validate()
    assert len(mapping.dofs) == 14
    assert sum(item.anatomical_role == "knee_flexion" for item in mapping.dofs) == 2
    assert sum(item.anatomical_role == "hip_rotation" for item in mapping.dofs) == 2


def test_full_so3_map_is_explicitly_diagnostic() -> None:
    mapping = LowerBodyDofMap.full_so3_diagnostic()
    mapping.validate()
    assert len(mapping.dofs) == 24
    assert mapping.source.endswith("diagnostic_only")


def test_lower_body_map_rejects_unknown_joint() -> None:
    mapping = LowerBodyDofMap(LowerBodyDofMap.default().dofs[:-1] + (mapping_dof("bad_joint"),))
    with pytest.raises(ValueError):
        mapping.validate()


def mapping_dof(joint: str):
    from sampling.lower_body_terminal_compensation import LowerBodyDof
    return LowerBodyDof(joint, (1.0, 0.0, 0.0), "test")


def test_solver_config_rejects_zero_damping() -> None:
    with pytest.raises(ValueError):
        LowerBodySolverConfig(damping=0.0).validate()


def test_solver_config_rejects_nonpositive_position_tolerance() -> None:
    with pytest.raises(ValueError):
        LowerBodySolverConfig(position_tolerance_m=0.0).validate()


def test_position_rows_follow_frozen_per_side_flat_contact() -> None:
    evidence = {
        "left": {"valid_masks": {"flat_contact": [False, True, False]}},
        "right": {"valid_masks": {"flat_contact": [True, False, False]}},
    }
    assert _position_marker_indices(evidence, 0, 3, torch.device("cpu")) == [2, 3]
    assert _position_marker_indices(evidence, 1, 3, torch.device("cpu")) == [0, 1]
    assert _position_marker_indices(evidence, 2, 3, torch.device("cpu")) == []


def test_marker_residual_stats_are_per_marker_and_use_max_marker_for_gate() -> None:
    error = torch.tensor(
        [[0.0006, 0.0008, 0.0], [0.0, 0.0, 0.0004], [0.02, 0.0, 0.0], [0.03, 0.0, 0.0]],
        dtype=torch.float32,
    )
    total, largest, per_marker = _marker_residual_stats(error, [0, 1])
    assert total == pytest.approx(1.077033, abs=1.0e-5)
    assert largest == pytest.approx(1.0, abs=1.0e-5)
    assert per_marker["left_heel"] == pytest.approx(1.0, abs=1.0e-5)
    assert per_marker["left_toe"] == pytest.approx(0.4, abs=1.0e-5)


def test_finite_difference_jacobian_matches_linear_fixture() -> None:
    point = torch.tensor([0.2, -0.4], dtype=torch.float32)
    matrix = torch.tensor([[2.0, 1.0], [-3.0, 0.5]], dtype=torch.float32)
    result = _finite_difference_jacobian(lambda value: matrix @ value, point, 1.0e-4)
    assert torch.allclose(result, matrix, atol=2.0e-3)
