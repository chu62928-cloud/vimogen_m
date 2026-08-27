import pytest
import torch

pytest.importorskip("motion_rep.phase1")

from evaluation.representation_recovery import (  # noqa: E402
    CorruptionConfig,
    calibrate_corruption,
    corrupt_motion,
    evaluate_one,
    summarize,
)
from motion_rep.phase1 import MOTION_LAYOUT, encode_rot6d  # noqa: E402


def _motion(frames: int = 6) -> torch.Tensor:
    motion = torch.zeros(frames, 276)
    identity = encode_rot6d(torch.eye(3).reshape(1, 3, 3))[0]
    motion[:, MOTION_LAYOUT.body_pose] = identity.repeat(21)
    motion[:, MOTION_LAYOUT.root_rotation] = identity
    motion[:, MOTION_LAYOUT.root_rotation_velocity] = identity
    joints = torch.zeros(frames, 22, 3)
    joints[:, :, 1] = torch.arange(frames, dtype=torch.float32)[:, None]
    motion[:, MOTION_LAYOUT.joints] = joints.reshape(frames, 66)
    motion[:, MOTION_LAYOUT.joints_velocity] = torch.cat((joints[1:] - joints[:-1], joints[-1:] - joints[-2:-1])).reshape(frames, 66)
    translation = torch.zeros(frames, 3)
    translation[:, 1] = torch.arange(frames, dtype=torch.float32)
    motion[:, MOTION_LAYOUT.root_translation] = translation
    motion[:, MOTION_LAYOUT.root_translation_velocity] = torch.cat((translation[1:] - translation[:-1], translation[-1:] - translation[-2:-1]))
    return motion


def test_corruption_is_deterministic():
    source = _motion()
    config = CorruptionConfig()
    first = corrupt_motion(source, sample_key="a", config=config)
    second = corrupt_motion(source, sample_key="a", config=config)
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_dev_calibration_and_paired_summary():
    config = calibrate_corruption([_motion(), _motion(7)])
    record = evaluate_one(_motion(), sample_key="a", corruption=config)
    summary = summarize([record])
    assert summary["record_count"] == 1
    assert set(summary["methods"]) == {"absolute_position", "velocity_integral", "reconciled"}
    assert "paired_bootstrap_effects" in summary
