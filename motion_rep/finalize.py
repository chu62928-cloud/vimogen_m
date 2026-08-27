"""Canonical ``T+1`` pose stream to ViMoGen ``T x 276`` finalisation.

The interface intentionally accepts physical-space pose data.  All geometry,
relative rotations and forward differences are computed before optional
standardisation.  The final output has one fewer frame because each output
row needs the following pose to define its velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .phase1 import (
    MOTION_LAYOUT,
    encode_rot6d,
    standardize_motion,
    validate_motion_tensor,
)


@dataclass(frozen=True)
class FinalizedMotion:
    """Finalised motion and the mask of rows with a valid next-pose boundary."""

    motion: torch.Tensor
    valid_mask: torch.Tensor


def _check_stream_shapes(
    local_rotations: torch.Tensor,
    joint_positions: torch.Tensor,
    root_rotation: torch.Tensor,
    root_translation: torch.Tensor,
) -> tuple[bool, int]:
    if local_rotations.ndim not in (4, 5) or local_rotations.shape[-3:] != (21, 3, 3):
        raise ValueError(
            "local_rotations must have shape [T+1,21,3,3] or [B,T+1,21,3,3]"
        )
    batched = local_rotations.ndim == 5
    time_axis = 1 if batched else 0
    batch_prefix = local_rotations.shape[:time_axis]
    frame_count = local_rotations.shape[time_axis]

    expected_joint = batch_prefix + (frame_count, 22, 3)
    expected_root_rotation = batch_prefix + (frame_count, 3, 3)
    expected_translation = batch_prefix + (frame_count, 3)
    if tuple(joint_positions.shape) != expected_joint:
        raise ValueError(f"joint_positions must have shape {expected_joint}, got {tuple(joint_positions.shape)}")
    if tuple(root_rotation.shape) != expected_root_rotation:
        raise ValueError(f"root_rotation must have shape {expected_root_rotation}, got {tuple(root_rotation.shape)}")
    if tuple(root_translation.shape) != expected_translation:
        raise ValueError(
            f"root_translation must have shape {expected_translation}, got {tuple(root_translation.shape)}"
        )
    if frame_count < 2:
        raise ValueError("finalization requires T+1 frames with T >= 1")
    if not all(torch.is_floating_point(value) for value in (local_rotations, joint_positions, root_rotation, root_translation)):
        raise TypeError("pose streams must use floating point dtypes")
    return batched, frame_count


def _time_slice(value: torch.Tensor, batched: bool, start: int, stop: int | None = None) -> torch.Tensor:
    index = [slice(None)] * value.ndim
    index[1 if batched else 0] = slice(start, stop)
    return value[tuple(index)]


def _relative_rotation(rotation: torch.Tensor, batched: bool) -> torch.Tensor:
    current = _time_slice(rotation, batched, 1, None)
    previous = _time_slice(rotation, batched, 0, -1)
    return current @ previous.transpose(-1, -2)


def _check_rotation_matrices(rotation: torch.Tensor, name: str) -> None:
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    gram = rotation.transpose(-1, -2) @ rotation
    if not torch.isfinite(rotation).all() or not torch.isfinite(gram).all():
        raise ValueError(f"{name} contains non-finite values")
    if not torch.allclose(gram, identity, atol=1e-4, rtol=1e-4):
        raise ValueError(f"{name} is not a valid SO(3) rotation")
    determinant = torch.linalg.det(rotation)
    if torch.any(determinant <= 0):
        raise ValueError(f"{name} contains reflections or singular matrices")


def _canonical_valid_mask(
    valid_mask: torch.Tensor | None,
    *,
    batched: bool,
    frame_count: int,
    batch_size: int | None,
    device: torch.device,
) -> torch.Tensor:
    output_frames = frame_count - 1
    if valid_mask is None:
        shape = (batch_size, output_frames) if batched else (output_frames,)
        return torch.ones(shape, dtype=torch.bool, device=device)
    expected = (batch_size, frame_count) if batched else (frame_count,)
    if tuple(valid_mask.shape) != expected or valid_mask.dtype != torch.bool:
        raise ValueError(f"valid_mask must have shape {expected} and dtype bool")
    return valid_mask[..., :-1] & valid_mask[..., 1:]


def finalize_motion(
    local_rotations: torch.Tensor,
    joint_positions: torch.Tensor,
    root_rotation: torch.Tensor,
    root_translation: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> FinalizedMotion:
    """Convert physical ``T+1`` pose streams into a canonical ``T x 276`` tensor.

    The last pose is not copied or padded: it supplies the forward-difference
    boundary for the final output row.  If either pose of a pair is masked,
    that output row is marked invalid and zeroed after optional
    standardisation.  This makes the boundary policy explicit and prevents
    padded data from entering later angle or contact metrics.
    """

    batched, frame_count = _check_stream_shapes(
        local_rotations, joint_positions, root_rotation, root_translation
    )
    for value, name in (
        (local_rotations, "local_rotations"),
        (root_rotation, "root_rotation"),
    ):
        _check_rotation_matrices(value, name)

    body = encode_rot6d(_time_slice(local_rotations, batched, 0, -1)).reshape(
        (*local_rotations.shape[: (1 if batched else 0)], frame_count - 1, 126)
    )
    joints = _time_slice(joint_positions, batched, 0, -1).reshape(
        (*joint_positions.shape[: (1 if batched else 0)], frame_count - 1, 66)
    )
    joints_velocity = (
        _time_slice(joint_positions, batched, 1, None)
        - _time_slice(joint_positions, batched, 0, -1)
    ).reshape(*joints.shape[:-1], 66)
    root = encode_rot6d(_time_slice(root_rotation, batched, 0, -1))
    root_velocity = encode_rot6d(_relative_rotation(root_rotation, batched))
    translation = _time_slice(root_translation, batched, 0, -1)
    translation_velocity = _time_slice(root_translation, batched, 1, None) - translation

    motion = torch.cat(
        (
            body,
            joints,
            joints_velocity,
            root,
            root_velocity,
            translation,
            translation_velocity,
        ),
        dim=-1,
    )
    validate_motion_tensor(motion)
    mask = _canonical_valid_mask(
        valid_mask,
        batched=batched,
        frame_count=frame_count,
        batch_size=local_rotations.shape[0] if batched else None,
        device=motion.device,
    )

    if (mean is None) != (std is None):
        raise ValueError("mean and std must be supplied together")
    if mean is not None and std is not None:
        motion = standardize_motion(motion, mean, std)

    if output_dtype is not None:
        motion = motion.to(dtype=output_dtype)
    # Invalid rows are deliberately zero, independent of the standardisation
    # offset, so downstream code cannot accidentally consume them.
    motion = motion.masked_fill(~mask.unsqueeze(-1), 0)
    return FinalizedMotion(motion=motion, valid_mask=mask)

