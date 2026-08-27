import torch

from motion_rep.phase1 import MOTION_LAYOUT
from scripts.audit_turning_m1_angles import (
    SPEED_THRESHOLD_M_PER_FRAME,
    _heading_summary,
    _speed_summary,
)


def test_speed_summary_uses_position_difference_and_hidden_last_boundary():
    motion = torch.zeros(4, 276)
    motion[:, MOTION_LAYOUT.root_translation] = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.02, 0.0], [0.0, 0.03, 0.0]]
    )
    motion[:, MOTION_LAYOUT.root_translation_velocity] = torch.tensor(
        [[0.0, 0.01, 0.0], [0.0, 0.01, 0.0], [0.0, 0.01, 0.0], [0.0, 0.02, 0.0]]
    )
    summary = _speed_summary(motion)
    assert summary["travel_valid_frame_count"] == 4
    assert summary["stored_vs_visible_position_difference_max_abs_m"] == 0.0
    assert summary["last_stored_velocity_is_hidden_boundary"] is True
    assert summary["threshold_m_per_frame"] == SPEED_THRESHOLD_M_PER_FRAME


def test_heading_summary_reports_low_speed_frames_as_invalid():
    motion = torch.zeros(4, 276)
    motion[:, MOTION_LAYOUT.root_translation] = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.011, 0.0], [0.0, 0.021, 0.0]]
    )
    summary = _heading_summary(motion)
    assert summary["valid_frame_count"] == 3
    assert summary["heading_unwrapped_range_degrees"] == 0.0
