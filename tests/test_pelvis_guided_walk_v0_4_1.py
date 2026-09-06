"""Unit tests for the v0.4.1 terminal policy seam."""

from __future__ import annotations

import pytest
import torch

from evaluation.pelvis_contact_compensation_v3 import pelvis_pitch_delta_deg, target_root_rotation
from sampling.terminal_projection_policy import TerminalProjectionPolicy, build_terminal_root_target


def _roots(frames: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    m0 = torch.eye(3).repeat(frames, 1, 1)
    source = target_root_rotation(m0, torch.full((frames,), 1.0))
    return m0, source


def test_dead_zone_leaves_requested_signed_residual() -> None:
    m0, source = _roots()
    target, error, active = build_terminal_root_target(
        m0,
        source,
        2.0,
        valid_mask=torch.ones(4, dtype=torch.bool),
        policy=TerminalProjectionPolicy(dead_zone_deg=0.5),
    )
    actual = pelvis_pitch_delta_deg(m0, target)
    assert torch.all(active)
    assert torch.allclose(error, torch.full((4,), 1.0), atol=1e-5)
    assert torch.allclose(actual, torch.full((4,), 1.5), atol=1e-5)


def test_inside_dead_zone_does_not_edit_root() -> None:
    m0, source = _roots()
    target, _, active = build_terminal_root_target(
        m0,
        source,
        2.0,
        valid_mask=torch.ones(4, dtype=torch.bool),
        policy=TerminalProjectionPolicy(dead_zone_deg=1.0),
    )
    assert not bool(active.any())
    assert torch.equal(target, source)


def test_disabled_pelvis_keeps_source_even_with_zero_dead_zone() -> None:
    m0, source = _roots()
    target, _, active = build_terminal_root_target(
        m0,
        source,
        2.0,
        valid_mask=torch.ones(4, dtype=torch.bool),
        policy=TerminalProjectionPolicy(dead_zone_deg=0.0, pelvis_enabled=False),
    )
    assert not bool(active.any())
    assert torch.equal(target, source)


@pytest.mark.parametrize("value", [-1.0, 180.0, 360.0])
def test_dead_zone_range_is_validated(value: float) -> None:
    with pytest.raises(ValueError):
        TerminalProjectionPolicy(dead_zone_deg=value).validate()

