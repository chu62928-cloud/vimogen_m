import torch

from motion_rep.phase1 import MOTION_LAYOUT, encode_rot6d
from scripts.evaluate_m1_v2_real_pilot import channel_consistency


def _consistent_motion(frames=4):
    motion = torch.zeros(frames, 276, dtype=torch.float32)
    identity6d = encode_rot6d(torch.eye(3).reshape(1, 3, 3))[0]
    motion[:, MOTION_LAYOUT.body_pose] = identity6d.repeat(21)
    motion[:, MOTION_LAYOUT.root_rotation] = identity6d
    motion[:, MOTION_LAYOUT.root_rotation_velocity] = identity6d
    joints = torch.zeros(frames, 22, 3)
    joints[:, :, 1] = torch.arange(frames, dtype=torch.float32)[:, None]
    motion[:, MOTION_LAYOUT.joints] = joints.reshape(frames, 66)
    velocity = torch.zeros_like(joints)
    velocity[:, :, 1] = 1.0
    motion[:, MOTION_LAYOUT.joints_velocity] = velocity.reshape(frames, 66)
    motion[:, MOTION_LAYOUT.root_translation] = joints[:, 0]
    motion[:, MOTION_LAYOUT.root_translation_velocity] = torch.tensor([0.0, 1.0, 0.0])
    return motion


def test_channel_consistency_is_zero_for_recomputed_velocity_channels():
    result = channel_consistency(_consistent_motion())
    assert result["joint_position_velocity_max_m"] == 0.0
    assert result["root_translation_velocity_max_m"] == 0.0
    assert result["root_rotation_velocity_max_degrees"] == 0.0
    assert result["last_row_finite"] is True


def test_channel_consistency_detects_corrupted_velocity_channel():
    motion = _consistent_motion()
    motion[1, MOTION_LAYOUT.root_translation_velocity.start] += 0.25
    result = channel_consistency(motion)
    assert result["root_translation_velocity_max_m"] >= 0.25
