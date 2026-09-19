"""Boundary tests for the v2 unified evaluator and optimizer fallbacks."""

from __future__ import annotations

import json

import pytest
import torch

from evaluation.relative_root_forward_v2 import (
    FAIL,
    NOT_EVALUABLE,
    PASS,
    Gate,
    build_v2_gates,
    combine_gate_statuses,
    contact_evidence,
    write_json_strict,
)
from motion_rep.phase1 import MOTION_LAYOUT
from motion_rep.phase1 import encode_rot6d
from sampling.differentiable_flow_sampler import _subspace_response_vector
from sampling.relative_root_forward_guidance_v2 import _normalized_negative_gradient


def test_gate_precedence_keeps_explicit_failure_over_missing_evidence() -> None:
    assert combine_gate_statuses([Gate("missing", NOT_EVALUABLE, 1, None, 0, "short"), Gate("bad", FAIL, 1, 2, 1, "over")]) == FAIL
    assert combine_gate_statuses([Gate("missing", NOT_EVALUABLE, 1, None, 0, "short")]) == NOT_EVALUABLE
    assert combine_gate_statuses([Gate("ok", PASS, 1, 1, 1, "equal")]) == PASS


def test_contact_evidence_excludes_first_frame_and_reports_not_evaluable() -> None:
    heel = torch.zeros(2, 3)
    toe = torch.zeros(2, 3)
    evidence = contact_evidence(heel, toe)
    assert evidence["contact_frames"] == 1
    assert evidence["valid_masks"]["contact"][0] is False
    assert evidence["height_evidence"]["status"] == NOT_EVALUABLE
    assert evidence["sliding_evidence_m_per_frame"]["status"] == NOT_EVALUABLE


def test_contact_evidence_requires_continuous_pairs_for_sliding() -> None:
    heel = torch.zeros(6, 3)
    toe = torch.zeros(6, 3)
    heel[:, 0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
    toe[:, 0] = heel[:, 0]
    evidence = contact_evidence(heel, toe, contact_speed_m_per_frame=0.01)
    assert evidence["continuous_contact_pairs"] == 3
    assert evidence["sliding_evidence_m_per_frame"]["status"] == PASS


def test_negative_gradient_fallback_is_masked_and_rms_normalized() -> None:
    gradient = torch.ones((1, 3, MOTION_LAYOUT.total_dim))
    mask = torch.tensor([[True, False, True]])
    direction = _normalized_negative_gradient(gradient, 0.25, mask)
    assert direction is not None
    assert torch.count_nonzero(direction[:, 1]) == 0
    assert torch.sqrt(direction.square().sum() / (2 * MOTION_LAYOUT.total_dim)) == pytest.approx(0.25)
    assert _normalized_negative_gradient(torch.zeros_like(gradient), 0.25, mask) is None


def test_unified_gates_do_not_call_missing_tail_or_trunk_evidence_a_pass() -> None:
    gates = build_v2_gates({"per_sample": [], "tail_safety": {}})
    assert gates[0].status == NOT_EVALUABLE


def test_strict_json_writer_rejects_nonfinite_values(tmp_path) -> None:
    with pytest.raises(ValueError):
        write_json_strict(tmp_path / "bad.json", {"x": float("nan")})
    target = tmp_path / "good.json"
    write_json_strict(target, {"x": [1, 2], "ok": True})
    assert json.loads(target.read_text(encoding="utf-8"))["ok"] is True


def test_subspace_response_vector_has_fixed_root_body_foot_feature_layout() -> None:
    motion = torch.zeros((1, 3, MOTION_LAYOUT.total_dim))
    identity = encode_rot6d(torch.eye(3)).reshape(6)
    root_forward_horizontal = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    motion[..., MOTION_LAYOUT.body_pose] = identity.repeat(21)
    motion[..., MOTION_LAYOUT.root_rotation] = encode_rot6d(root_forward_horizontal)
    motion[..., MOTION_LAYOUT.joints] = 0.0
    motion[..., MOTION_LAYOUT.root_translation] = 0.0
    values = _subspace_response_vector(
        motion,
        torch.zeros(MOTION_LAYOUT.total_dim),
        torch.ones(MOTION_LAYOUT.total_dim),
        torch.tensor([[True, True, False]]),
    )
    assert values.shape == (2, 16)
    assert torch.isfinite(values).all()
