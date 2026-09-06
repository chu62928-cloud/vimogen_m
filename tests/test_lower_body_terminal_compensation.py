"""Tests for the v0.4.2 lower-body terminal prototype."""

from __future__ import annotations

import pytest

from sampling.lower_body_terminal_compensation import LowerBodyDofMap, LowerBodySolverConfig


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
