"""Common finalization adapter for physical ViMoGen ``T x 276`` motions.

The model stores the first ``T`` poses and one forward-difference row per
frame.  This module makes the representation boundary explicit: recover the
``T+1`` pose stream from the first pose plus the stored velocities, then call
the single physical-space finalizer in :mod:`motion_rep.finalize`.  It is the
adapter used by B0 and by the revised M1 evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .finalize import FinalizedMotion, finalize_motion
from .phase1 import (
    MOTION_LAYOUT,
    decode_rot6d_safe,
    destandardize_motion,
    standardize_motion,
    validate_motion_tensor,
)


@dataclass(frozen=True)
class UnifiedFinalizedMotion(FinalizedMotion):
    """Result plus a machine-readable record of the representation boundary."""

    boundary: str = "physical_Tplus1_to_Tx276"
    input_was_standardized: bool = False
    output_is_standardized: bool = False


def _recover_single_pose_stream(motion: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Recover local rotations, joints, root rotations and translations.

    The first pose is the root anchor.  Root rotation and all position streams
    are then advanced with the stored forward-difference rows.  Body-local
    rotations have no separate velocity channels in ViMoGen, so the final
    local pose is held at the last stored body pose, matching the legacy
    ``equal_length=True`` reconstruction.
    """

    frames = motion.shape[0]
    if frames < 1:
        raise ValueError("motion must contain at least one frame")
    body = decode_rot6d_safe(
        motion[:, MOTION_LAYOUT.body_pose].reshape(frames, 21, 6)
    )
    body_stream = torch.cat((body, body[-1:]), dim=0)

    joints = motion[:, MOTION_LAYOUT.joints].reshape(frames, 22, 3)
    joints_velocity = motion[:, MOTION_LAYOUT.joints_velocity].reshape(frames, 22, 3)
    joints_stream = torch.cat(
        (joints[:1], joints[:1] + torch.cumsum(joints_velocity, dim=0)), dim=0
    )

    root = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation])
    root_velocity = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation_velocity])
    root_frames = [root[:1]]
    for index in range(frames):
        root_frames.append(root_velocity[index:index + 1] @ root_frames[-1])
    root_stream = torch.cat(root_frames, dim=0)

    translation = motion[:, MOTION_LAYOUT.root_translation]
    translation_velocity = motion[:, MOTION_LAYOUT.root_translation_velocity]
    translation_stream = torch.cat(
        (
            translation[:1],
            translation[:1] + torch.cumsum(translation_velocity, dim=0),
        ),
        dim=0,
    )
    return body_stream, joints_stream, root_stream, translation_stream


def _single_valid_pose_mask(
    valid_mask: torch.Tensor | None, frames: int, device: torch.device
) -> torch.Tensor:
    if valid_mask is None:
        row_mask = torch.ones(frames, dtype=torch.bool, device=device)
    else:
        if valid_mask.shape != (frames,) or valid_mask.dtype is not torch.bool:
            raise ValueError(f"valid_mask must have shape {(frames,)} and dtype bool")
        row_mask = valid_mask
    # The final recovered pose is supplied by the last velocity row.  It is
    # valid only when the last stored row is valid; pair masking in
    # finalize_motion then makes the final output row explicit.
    return torch.cat((row_mask, row_mask[-1:]), dim=0)


def _single_finalize(
    motion: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> FinalizedMotion:
    body, joints, root, translation = _recover_single_pose_stream(motion)
    pose_mask = _single_valid_pose_mask(valid_mask, motion.shape[0], motion.device)
    return finalize_motion(
        body,
        joints,
        root,
        translation,
        valid_mask=pose_mask,
    )


def finalize_motion_tensor(
    motion: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    input_standardized: bool = False,
    output_standardized: bool = False,
    output_dtype: torch.dtype | None = None,
) -> UnifiedFinalizedMotion:
    """Apply the common finalizer to physical or explicitly normalized motion.

    Parameters
    ----------
    motion:
        ``[T,276]`` or ``[B,T,276]``.  The default boundary is physical space.
    valid_mask:
        Row mask with shape ``[T]`` or ``[B,T]``.  The recovered last pose is
        given the last row's mask; the low-level finalizer then masks pairs.
    mean, std:
        The same 276-channel statistics used by ViMoGen.  They are mandatory
        whenever either standardization flag is true.
    input_standardized/output_standardized:
        Declare the normalization boundary instead of silently guessing it.
    """

    validate_motion_tensor(motion)
    if (input_standardized or output_standardized) and (mean is None or std is None):
        raise ValueError("mean and std must be supplied for standardized input/output")
    if not (input_standardized or output_standardized) and (mean is not None or std is not None):
        raise ValueError("mean and std require an explicit standardization flag")
    physical = (
        destandardize_motion(motion, mean, std) if input_standardized else motion
    )
    batched = physical.ndim == 3
    if valid_mask is not None:
        expected = physical.shape[:-1]
        if tuple(valid_mask.shape) != tuple(expected) or valid_mask.dtype is not torch.bool:
            raise ValueError(f"valid_mask must have shape {tuple(expected)} and dtype bool")
    if batched:
        finalized = [
            _single_finalize(physical[index], None if valid_mask is None else valid_mask[index])
            for index in range(physical.shape[0])
        ]
        result_motion = torch.stack([item.motion for item in finalized], dim=0)
        result_mask = torch.stack([item.valid_mask for item in finalized], dim=0)
    else:
        result = _single_finalize(physical, valid_mask)
        result_motion, result_mask = result.motion, result.valid_mask
    if output_standardized:
        result_motion = standardize_motion(result_motion, mean, std)
        result_motion = result_motion.masked_fill(~result_mask.unsqueeze(-1), 0)
    if output_dtype is not None:
        result_motion = result_motion.to(output_dtype)
    return UnifiedFinalizedMotion(
        motion=result_motion,
        valid_mask=result_mask,
        input_was_standardized=input_standardized,
        output_is_standardized=output_standardized,
    )

