import pytest

from evaluation.m1_paired_metrics import paired_excess, summarize_paired


def _metrics(joint=0.003, trans=0.002, rot=0.137):
    return {
        "joint_position_velocity_median_m": joint,
        "root_translation_velocity_median_m": trans,
        "root_rotation_velocity_median_degrees": rot,
    }


def test_paired_excess_uses_m0_as_reference_and_fixed_limits():
    result = paired_excess(_metrics(), _metrics(joint=0.0035, trans=0.0029, rot=0.20))
    assert result["excess"]["joint_position_velocity_median_m"] == pytest.approx(0.0005)
    assert result["excess"]["root_translation_velocity_median_m"] == pytest.approx(0.0009)
    assert result["excess"]["root_rotation_velocity_median_degrees"] == pytest.approx(0.063)
    assert result["non_degradation_pass"] is True


def test_paired_excess_fails_only_the_channel_over_its_limit():
    result = paired_excess(_metrics(), _metrics(joint=0.0041, trans=0.0021, rot=0.137))
    assert result["channel_pass"] == {
        "joint_position_velocity_median_m": False,
        "root_translation_velocity_median_m": True,
        "root_rotation_velocity_median_degrees": True,
    }
    assert result["non_degradation_pass"] is False


def test_summary_keeps_each_pair_and_does_not_average_before_the_gate():
    records = [
        {"m0": _metrics(), "method": _metrics(joint=0.0035)},
        {"m0": _metrics(), "method": _metrics(joint=0.0041)},
    ]
    result = summarize_paired(records)
    assert result["count"] == 2
    assert result["all_samples_pass"] is False
    assert len(result["records"]) == 2
