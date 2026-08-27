import torch

from scripts.diagnose_m1_devset_failures import _summarise_curve


def test_failure_summary_reports_target_and_valid_frames():
    baseline = torch.zeros(100, 276)
    candidate = baseline.clone()
    summary = _summarise_curve(baseline, candidate, 0.0)
    assert summary["valid_frame_count"] == 100
    assert summary["absolute_error_median_degrees"] == 0.0


def test_failure_summary_preserves_signed_target_error():
    baseline = torch.zeros(100, 276)
    candidate = baseline.clone()
    # canonical-y heading and the candidate +z axis are already upright; this
    # sanity test ensures the diagnostic remains finite on the neutral pose.
    summary = _summarise_curve(baseline, candidate, 5.0)
    assert summary["target_delta_degrees"] == 5.0
    assert summary["absolute_error_median_degrees"] == 5.0
