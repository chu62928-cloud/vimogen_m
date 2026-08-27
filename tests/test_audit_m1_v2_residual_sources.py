from scripts.audit_m1_v2_residual_sources import summarize_source_contribution


def test_summary_identifies_official_smoothing_reduction_without_overclaiming():
    raw = {
        "joint_position_velocity_median_m": 0.010,
        "root_translation_velocity_median_m": 0.008,
        "root_rotation_velocity_median_degrees": 0.40,
    }
    official = {
        "joint_position_velocity_median_m": 0.004,
        "root_translation_velocity_median_m": 0.003,
        "root_rotation_velocity_median_degrees": 0.20,
    }
    result = summarize_source_contribution(raw, official)
    assert result["official_smoothing_reduces_all_three_medians"] is True
    assert result["euler_drift_attribution"] == "UNKNOWN_NO_STEP_TRACE"
    assert result["model_reprediction_attribution"] == "UNKNOWN_NO_STEP_TRACE"


def test_summary_does_not_call_smoothing_a_fix_when_residual_increases():
    raw = {
        "joint_position_velocity_median_m": 0.004,
        "root_translation_velocity_median_m": 0.003,
        "root_rotation_velocity_median_degrees": 0.20,
    }
    official = {
        "joint_position_velocity_median_m": 0.006,
        "root_translation_velocity_median_m": 0.004,
        "root_rotation_velocity_median_degrees": 0.30,
    }
    result = summarize_source_contribution(raw, official)
    assert result["official_smoothing_reduces_all_three_medians"] is False
    assert result["interpretation"] == "SMOOTHING_NOT_REDUCING_ALL_CHANNELS"
