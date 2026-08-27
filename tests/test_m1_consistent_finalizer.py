import torch

from motion_rep.consistent_finalizer import finalize_consistent_motion_tensor
from motion_rep.phase1 import MOTION_LAYOUT, encode_rot6d, decode_rot6d_safe


def _z_rotation(angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    out = torch.zeros((*angle.shape, 3, 3), dtype=angle.dtype)
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    out[..., 2, 2] = 1
    return out


def _packed_inconsistent(frames: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    body = torch.eye(3).repeat(frames, 21, 1, 1)
    root = _z_rotation(torch.linspace(0.0, 0.4, frames))
    joints = torch.zeros(frames, 22, 3)
    joints[:, :, 1] = torch.arange(frames).float().unsqueeze(-1)
    translation = torch.zeros(frames, 3)
    translation[:, 1] = torch.arange(frames).float()
    packed = torch.zeros(frames, 276)
    packed[:, MOTION_LAYOUT.body_pose] = encode_rot6d(body).reshape(frames, 126)
    packed[:, MOTION_LAYOUT.joints] = joints.reshape(frames, 66)
    packed[:, MOTION_LAYOUT.root_rotation] = encode_rot6d(root)
    packed[:, MOTION_LAYOUT.root_translation] = translation
    # Deliberately stale redundant channels: the direct pose channels are the authority.
    packed[:, MOTION_LAYOUT.joints_velocity] = 0
    packed[:, MOTION_LAYOUT.root_rotation_velocity] = encode_rot6d(torch.eye(3).repeat(frames, 1, 1))
    packed[:, MOTION_LAYOUT.root_translation_velocity] = 0
    return packed, joints, root


def test_pose_authority_preserves_direct_positions_and_recomputes_velocity():
    packed, joints, root = _packed_inconsistent()
    result = finalize_consistent_motion_tensor(packed).motion
    assert torch.allclose(result[:, MOTION_LAYOUT.joints].reshape(-1, 22, 3), joints, atol=1e-6)
    recovered_v = result[:, MOTION_LAYOUT.joints_velocity].reshape(-1, 22, 3)
    assert torch.allclose(recovered_v[:-1], joints[1:] - joints[:-1], atol=1e-6)
    assert torch.allclose(recovered_v[-1], joints[-1] - joints[-2], atol=1e-6)


def test_pose_authority_recomputes_root_rotation_velocity():
    packed, _, root = _packed_inconsistent()
    result = finalize_consistent_motion_tensor(packed).motion
    decoded_v = decode_rot6d_safe(result[:, MOTION_LAYOUT.root_rotation_velocity])
    expected = root[1:] @ root[:-1].transpose(-1, -2)
    expected_last = root[-1:] @ root[-2:-1].transpose(-1, -2)
    assert torch.allclose(decoded_v[:-1], expected, atol=1e-5)
    assert torch.allclose(decoded_v[-1], expected_last, atol=1e-5)


def test_standardized_boundary_is_explicit():
    packed, _, _ = _packed_inconsistent()
    mean = torch.zeros(276)
    std = torch.ones(276)
    result = finalize_consistent_motion_tensor(
        packed, mean=mean, std=std, input_standardized=True, output_standardized=True
    )
    assert torch.allclose(result.motion, finalize_consistent_motion_tensor(packed).motion, atol=1e-6)
