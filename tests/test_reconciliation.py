import pytest
import torch

pytest.importorskip("motion_rep.phase1")

from motion_rep.phase1 import MOTION_LAYOUT, encode_rot6d, decode_rot6d_safe  # noqa: E402
from motion_rep.reconciliation import ReconciliationConfig, reconcile_motion_tensor  # noqa: E402


def _motion(frames: int = 6) -> torch.Tensor:
    motion = torch.zeros(frames, 276)
    identity = encode_rot6d(torch.eye(3).reshape(1, 3, 3))[0]
    motion[:, MOTION_LAYOUT.body_pose] = identity.repeat(21)
    motion[:, MOTION_LAYOUT.root_rotation] = identity
    motion[:, MOTION_LAYOUT.root_rotation_velocity] = identity
    direct = torch.zeros(frames, 22, 3)
    direct[:, :, 1] = torch.arange(frames, dtype=torch.float32)[:, None]
    motion[:, MOTION_LAYOUT.joints] = direct.reshape(frames, 66)
    # The velocity view is deliberately too slow, so reconciliation has a
    # measurable long-term correction to recover.
    velocity = torch.zeros_like(direct)
    velocity[:, :, 1] = 0.5
    motion[:, MOTION_LAYOUT.joints_velocity] = velocity.reshape(frames, 66)
    translation = torch.zeros(frames, 3)
    translation[:, 1] = torch.arange(frames, dtype=torch.float32)
    motion[:, MOTION_LAYOUT.root_translation] = translation
    motion[:, MOTION_LAYOUT.root_translation_velocity] = torch.tensor([0.0, 0.5, 0.0])
    return motion


def test_zero_anchor_weight_is_velocity_authoritative():
    source = _motion()
    result = reconcile_motion_tensor(
        source,
        config=ReconciliationConfig(correction_window=3, anchor_weight=0.0, root_rotation_anchor_weight=0.0),
    ).motion
    joints = result[:, MOTION_LAYOUT.joints].reshape(-1, 22, 3)
    assert joints[-1, 0, 1] == pytest.approx(2.5)


def test_unit_anchor_weight_and_window_one_preserve_direct_pose():
    source = _motion()
    result = reconcile_motion_tensor(
        source,
        config=ReconciliationConfig(correction_window=1, anchor_weight=1.0, root_rotation_anchor_weight=1.0),
    ).motion
    joints = result[:, MOTION_LAYOUT.joints].reshape(-1, 22, 3)
    torch.testing.assert_close(joints, source[:, MOTION_LAYOUT.joints].reshape(-1, 22, 3))
    velocity = result[:, MOTION_LAYOUT.joints_velocity].reshape(-1, 22, 3)
    torch.testing.assert_close(velocity[:-1], joints[1:] - joints[:-1])


def test_reconciliation_returns_valid_rotation_and_recomputed_channels():
    result = reconcile_motion_tensor(_motion()).motion
    root = decode_rot6d_safe(result[:, MOTION_LAYOUT.root_rotation])
    torch.testing.assert_close(
        root.transpose(-1, -2) @ root,
        torch.eye(3).expand_as(root),
        atol=1e-5,
        rtol=0,
    )
    assert torch.isfinite(result).all()


def test_batched_per_sample_statistics_are_broadcast_explicitly():
    source = _motion()
    batch = torch.stack((source, source), dim=0)
    mean = torch.zeros(2, 276)
    std = torch.ones(2, 276)
    result = reconcile_motion_tensor(
        batch,
        config=ReconciliationConfig(correction_window=1),
        valid_mask=torch.ones(2, source.shape[0], dtype=torch.bool),
        mean=mean,
        std=std,
        input_standardized=True,
        output_standardized=True,
        output_dtype=torch.float32,
    ).motion
    assert result.shape == batch.shape
    assert torch.isfinite(result).all()
