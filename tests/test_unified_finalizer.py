import math

import pytest
import torch

from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe, encode_rot6d
from motion_rep.unified_finalizer import finalize_motion_tensor


def _identity_motion(frames: int = 4) -> torch.Tensor:
    motion = torch.zeros(frames, 276, dtype=torch.float32)
    identity6d = encode_rot6d(torch.eye(3).reshape(1, 3, 3))[0]
    motion[:, MOTION_LAYOUT.body_pose] = identity6d.repeat(21)
    motion[:, MOTION_LAYOUT.root_rotation] = identity6d
    motion[:, MOTION_LAYOUT.root_rotation_velocity] = identity6d
    positions = torch.zeros(frames, 22, 3)
    positions[:, :, 1] = torch.arange(frames, dtype=torch.float32)[:, None]
    motion[:, MOTION_LAYOUT.joints] = positions.reshape(frames, 66)
    motion[:, MOTION_LAYOUT.joints_velocity] = torch.ones(frames, 66)
    motion[:, MOTION_LAYOUT.root_translation] = positions[:, 0]
    motion[:, MOTION_LAYOUT.root_translation_velocity] = torch.tensor([0.0, 1.0, 0.0])
    return motion


def test_unified_finalizer_recovers_t_plus_one_and_recomputes_velocity():
    motion = _identity_motion()
    result = finalize_motion_tensor(motion)
    assert result.motion.shape == motion.shape
    assert result.valid_mask.tolist() == [True] * motion.shape[0]
    joints = result.motion[:, MOTION_LAYOUT.joints].reshape(-1, 22, 3)
    velocities = result.motion[:, MOTION_LAYOUT.joints_velocity].reshape(-1, 22, 3)
    torch.testing.assert_close(velocities, joints[1:] - joints[:-1])
    translation = result.motion[:, MOTION_LAYOUT.root_translation]
    translation_velocity = result.motion[:, MOTION_LAYOUT.root_translation_velocity]
    torch.testing.assert_close(translation_velocity, translation[1:] - translation[:-1])


def test_unified_finalizer_produces_valid_rot6d_and_root_velocity():
    motion = _identity_motion()
    angle = math.radians(15.0)
    motion[-1, MOTION_LAYOUT.root_rotation] = encode_rot6d(
        torch.tensor([[[math.cos(angle), -math.sin(angle), 0.0],
                       [math.sin(angle), math.cos(angle), 0.0],
                       [0.0, 0.0, 1.0]]])
    )[0]
    result = finalize_motion_tensor(motion)
    rotations = decode_rot6d_safe(result.motion[:, MOTION_LAYOUT.root_rotation])
    gram = rotations.transpose(-1, -2) @ rotations
    torch.testing.assert_close(gram, torch.eye(3).expand_as(gram), atol=1e-5, rtol=0)
    assert torch.linalg.det(rotations).min().item() > 0.999


def test_unified_finalizer_mask_and_standardization_boundaries_are_explicit():
    physical = _identity_motion()
    mean = torch.zeros(276)
    std = torch.full((276,), 2.0)
    normalized = (physical - mean) / std
    mask = torch.tensor([True, True, True, False])
    result = finalize_motion_tensor(
        normalized,
        valid_mask=mask,
        mean=mean,
        std=std,
        input_standardized=True,
        output_standardized=True,
    )
    assert result.valid_mask.tolist() == [True, True, False, False]
    assert torch.count_nonzero(result.motion[2:]).item() == 0
    torch.testing.assert_close(
        result.motion[:2, MOTION_LAYOUT.joints],
        physical[:2, MOTION_LAYOUT.joints],
        atol=1e-5,
        rtol=0,
    )
    with pytest.raises(ValueError, match="mean and std"):
        finalize_motion_tensor(normalized, input_standardized=True)


def test_b0_uses_the_same_adapter_and_zero_delta_is_an_exact_noop():
    from motion_rep.baselines import build_b0

    motion = _identity_motion()
    expected = finalize_motion_tensor(motion)
    actual = build_b0(motion)
    torch.testing.assert_close(actual.motion, expected.motion, atol=0, rtol=0)
    assert actual.valid_mask.equal(expected.valid_mask)


def test_batch_and_single_sample_replay_are_bitwise_equal():
    first = _identity_motion()
    second = _identity_motion() * 0.5
    single = finalize_motion_tensor(first)
    batch = finalize_motion_tensor(torch.stack((first, second)))
    torch.testing.assert_close(batch.motion[0], single.motion, atol=0, rtol=0)
