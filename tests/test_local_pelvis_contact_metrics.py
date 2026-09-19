import torch

from evaluation.local_pelvis_contact_metrics import (
    evaluate_frozen_contact,
    frozen_contact_reference,
)


def _markers(frames: int = 12) -> torch.Tensor:
    positions = torch.zeros(4, frames, 3)
    positions[1, :, 0] = 0.2
    positions[2:, :, 1] = 0.3
    positions[3, :, 0] = 0.2
    return positions


def test_identity_and_frozen_contact_displacement():
    baseline = _markers()
    valid = torch.ones(baseline.shape[1], dtype=torch.bool)
    frozen = frozen_contact_reference(baseline, valid)
    identity = evaluate_frozen_contact(baseline.clone(), frozen)
    assert identity["status"] == "EVALUATED"
    assert identity["full_contact_pass"]
    assert identity["contact_displacement_p95_mm"] == 0.0
    candidate = baseline.clone()
    candidate[..., 2] += 0.020
    lifted = evaluate_frozen_contact(candidate, frozen)
    assert lifted["status"] == "EVALUATED"
    assert not lifted["gates"]["contact_displacement"]
    assert lifted["contact_displacement_p95_mm"] > 19.0
    assert frozen["contact_mask"].any()


def test_frozen_contact_detects_sliding_and_missing_observation():
    baseline = _markers()
    valid = torch.ones(baseline.shape[1], dtype=torch.bool)
    frozen = frozen_contact_reference(baseline, valid)
    sliding = baseline.clone()
    sliding[:, :, 0] += torch.arange(baseline.shape[1]) * 0.010
    measured = evaluate_frozen_contact(sliding, frozen)
    assert not measured["gates"]["sliding_increment"]
    moving_m0 = baseline.clone()
    moving_m0[:, :, 0] += torch.arange(baseline.shape[1]) * 0.100
    no_contact = frozen_contact_reference(moving_m0, valid)
    result = evaluate_frozen_contact(moving_m0, no_contact)
    assert result["status"] == "NOT_EVALUABLE"
    assert not result["full_contact_pass"]
