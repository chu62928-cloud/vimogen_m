"""Focused tests for the opt-in full-FK v2 representation boundary."""

from pathlib import Path

import pytest
import torch

pytest.importorskip("torch")

from motion_rep.consistency_v2 import (  # noqa: E402
    PROTOCOL_NAME,
    SMPLX_22_PARENTS,
    default_smplx_neutral_22_skeleton,
    differentiable_forward_kinematics,
    reconcile_motion_tensor_v2,
)
from motion_rep.finalize import finalize_motion  # noqa: E402
from motion_rep.phase1 import MOTION_LAYOUT, encode_rot6d  # noqa: E402
from motion_rep.rotation_transform import axis_angle_to_mat3x3  # noqa: E402


def _tiny_skeleton() -> tuple[torch.Tensor, tuple[int, ...]]:
    offsets = torch.zeros(22, 3)
    offsets[0] = torch.tensor([0.25, -0.1, 0.4])
    offsets[1] = torch.tensor([0.2, 0.0, 0.0])
    offsets[2] = torch.tensor([-0.2, 0.0, 0.0])
    offsets[3] = torch.tensor([0.0, 0.0, 1.0])
    for index in range(4, 22):
        offsets[index] = torch.tensor([0.0, 0.0, 0.2])
    return offsets, SMPLX_22_PARENTS


def _packed(frames: int = 4, *, wrong_joints: bool = True) -> torch.Tensor:
    identity = torch.eye(3)
    local = identity.reshape(1, 1, 3, 3).repeat(frames + 1, 21, 1, 1)
    root = axis_angle_to_mat3x3(torch.zeros(frames + 1, 3))
    translation = torch.zeros(frames + 1, 3)
    translation[:, 1] = torch.arange(frames + 1) * 0.1
    joints = torch.zeros(frames + 1, 22, 3)
    if wrong_joints:
        joints += 10.0
    return finalize_motion(local, joints, root, translation).motion


def test_v2_uses_fk_joints_and_recomputes_all_redundant_channels():
    offsets, parents = _tiny_skeleton()
    motion = _packed()
    result = reconcile_motion_tensor_v2(
        motion,
        fusion_window=1,
        anchor_weight=1.0,
        root_rotation_anchor_weight=1.0,
        rest_offsets=offsets,
        parents=parents,
    )
    root = result.motion[:, MOTION_LAYOUT.root_rotation]
    del root  # root remains available below through the packed channels
    body = torch.eye(3).reshape(1, 1, 3, 3).repeat(5, 21, 1, 1)
    roots = torch.eye(3).reshape(1, 3, 3).repeat(5, 1, 1)
    trans = torch.zeros(5, 3)
    trans[:, 1] = torch.arange(5) * 0.1
    expected = differentiable_forward_kinematics(
        body,
        roots,
        trans,
        rest_offsets=offsets,
        parents=parents,
    ).joints
    actual = result.motion[:, MOTION_LAYOUT.joints].reshape(4, 22, 3)
    torch.testing.assert_close(actual, expected[:-1], atol=1e-6, rtol=0)
    velocity = result.motion[:, MOTION_LAYOUT.joints_velocity].reshape(4, 22, 3)
    torch.testing.assert_close(velocity, expected[1:] - expected[:-1], atol=1e-6, rtol=0)
    translation = result.motion[:, MOTION_LAYOUT.root_translation]
    translation_velocity = result.motion[:, MOTION_LAYOUT.root_translation_velocity]
    torch.testing.assert_close(translation, trans[:-1], atol=1e-6, rtol=0)
    torch.testing.assert_close(translation_velocity, trans[1:] - trans[:-1], atol=1e-6, rtol=0)
    root_velocity = result.motion[:, MOTION_LAYOUT.root_rotation_velocity]
    identity_rot6d = encode_rot6d(torch.eye(3)).expand_as(root_velocity)
    torch.testing.assert_close(root_velocity, identity_rot6d, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        actual[:, 0] - translation,
        offsets[0].expand_as(translation),
        atol=1e-6,
        rtol=0,
    )
    assert result.protocol == PROTOCOL_NAME


def test_fk_does_not_rotate_neutral_root_offset_and_is_differentiable():
    offsets, parents = _tiny_skeleton()
    local = torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 21, 1, 1)
    root = axis_angle_to_mat3x3(torch.tensor([[0.0, 0.0, 1.5707963], [0.0, 0.0, 1.5707963]]))
    translation = torch.zeros(2, 3, requires_grad=True)
    fk = differentiable_forward_kinematics(local, root, translation, rest_offsets=offsets, parents=parents)
    torch.testing.assert_close(fk.joints[0, 0], offsets[0], atol=1e-6, rtol=0)
    fk.joints.square().sum().backward()
    assert translation.grad is not None and torch.isfinite(translation.grad).all()


def test_v2_tail_padding_keeps_last_valid_output_and_ignores_tail_values():
    offsets, parents = _tiny_skeleton()
    first = _packed(frames=4)
    second = first.clone()
    second[2:] = torch.randn_like(second[2:])
    mask = torch.tensor([True, True, False, False])
    left = reconcile_motion_tensor_v2(first, fusion_window=1, valid_mask=mask, rest_offsets=offsets, parents=parents)
    right = reconcile_motion_tensor_v2(second, fusion_window=1, valid_mask=mask, rest_offsets=offsets, parents=parents)
    assert left.valid_mask.tolist() == [True, True, False, False]
    torch.testing.assert_close(left.motion[:2], right.motion[:2], atol=1e-6, rtol=0)


def test_v2_rejects_noncontiguous_valid_mask():
    offsets, parents = _tiny_skeleton()
    with pytest.raises(ValueError, match="contiguous valid prefix"):
        reconcile_motion_tensor_v2(
            _packed(frames=4),
            fusion_window=1,
            valid_mask=torch.tensor([True, False, True, False]),
            rest_offsets=offsets,
            parents=parents,
        )


def test_default_parent_table_is_frozen():
    assert list(SMPLX_22_PARENTS) == [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def test_default_asset_contains_neutral_root_and_offsets():
    skeleton = default_smplx_neutral_22_skeleton()
    assert skeleton.rest_offsets.shape == (22, 3)
    assert skeleton.root_offset.tolist() == pytest.approx(
        [0.00312325498, -0.3514074385, 0.01203655079], abs=1e-8
    )
    assert skeleton.rest_offsets[0].tolist() == pytest.approx([0.0, 0.0, 0.0], abs=0.0)


def test_lightweight_fk_matches_real_neutral_smplx_joints():
    smplx = pytest.importorskip("smplx")
    body_model_root = Path(__file__).resolve().parents[1] / "data/body_models"
    if not (body_model_root / "smplx/SMPLX_NEUTRAL.npz").is_file():
        pytest.skip("neutral SMPL-X asset is not mounted")
    model = smplx.create(
        str(body_model_root),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        batch_size=1,
    )
    with torch.no_grad():
        reference = model(return_verts=False).joints[0, :22]
    identity = torch.eye(3).reshape(1, 1, 3, 3)
    fk = differentiable_forward_kinematics(
        identity.expand(1, 21, 3, 3),
        identity[:, 0],
        torch.zeros(1, 3),
    )
    torch.testing.assert_close(fk.joints[0], reference, atol=3e-7, rtol=0)
