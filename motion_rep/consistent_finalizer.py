"""Pose-authoritative finalization for M1 outputs.

The packed 276-D representation contains both pose samples and redundant
forward-difference channels.  M1 can leave those two views inconsistent.  The
pose-authoritative boundary keeps the explicit first ``T`` poses, creates one
transparent constant-velocity endpoint, and recomputes every redundant
velocity channel before returning the packed motion.
"""

from __future__ import annotations

import torch

from .finalize import FinalizedMotion, finalize_motion
from .phase1 import MOTION_LAYOUT, decode_rot6d_safe, validate_motion_tensor
from .unified_finalizer import _broadcast_statistics, _physical_motion, _standardized_motion


def _stream_from_direct_pose(motion: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Recover a ``T+1`` stream using direct pose channels as authority."""

    validate_motion_tensor(motion)
    if motion.ndim != 2:
        raise ValueError("pose-authoritative finalizer expects one [T,276] motion")
    frames = motion.shape[0]
    if frames < 1:
        raise ValueError("motion must contain at least one frame")
    body = decode_rot6d_safe(motion[:, MOTION_LAYOUT.body_pose].reshape(frames, 21, 6))
    body_stream = torch.cat((body, body[-1:]), dim=0)

    joints = motion[:, MOTION_LAYOUT.joints].reshape(frames, 22, 3)
    translation = motion[:, MOTION_LAYOUT.root_translation]
    root = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation])
    if frames == 1:
        joints_next = joints[-1:]
        translation_next = translation[-1:]
        root_next = root[-1:]
    else:
        joints_next = joints[-1:] + (joints[-1:] - joints[-2:-1])
        translation_next = translation[-1:] + (translation[-1:] - translation[-2:-1])
        last_increment = root[-1:] @ root[-2:-1].transpose(-1, -2)
        root_next = last_increment @ root[-1:]
    return (
        body_stream,
        torch.cat((joints, joints_next), dim=0),
        torch.cat((root, root_next), dim=0),
        torch.cat((translation, translation_next), dim=0),
    )


def _single_mask(valid_mask: torch.Tensor | None, frames: int, device: torch.device) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones(frames, dtype=torch.bool, device=device)
    if valid_mask.shape != (frames,) or valid_mask.dtype is not torch.bool:
        raise ValueError(f"valid_mask must have shape {(frames,)} and dtype bool")
    return valid_mask


def _single_finalize(motion: torch.Tensor, valid_mask: torch.Tensor | None) -> FinalizedMotion:
    body, joints, root, translation = _stream_from_direct_pose(motion)
    pose_mask = torch.cat((_single_mask(valid_mask, motion.shape[0], motion.device),
                          _single_mask(valid_mask, motion.shape[0], motion.device)[-1:]))
    return finalize_motion(body, joints, root, translation, valid_mask=pose_mask)


def finalize_consistent_motion_tensor(
    motion: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    input_standardized: bool = False,
    output_standardized: bool = False,
    output_dtype: torch.dtype | None = None,
    protocol: str = "pose_authoritative_v1",
    skeleton=None,
) -> FinalizedMotion:
    """Finalize physical or explicitly normalized packed motion.

    The function is deliberately separate from the v1 velocity-authoritative
    finalizer so old M0/M1 results remain reproducible and auditable.
    """

    if protocol in {"fk_v2", "v2", "vimogen_276d_consistency_v2"}:
        from .consistency_v2 import reconcile_motion_tensor_v2

        return reconcile_motion_tensor_v2(
            motion,
            valid_mask=valid_mask,
            mean=mean,
            std=std,
            input_standardized=input_standardized,
            output_standardized=output_standardized,
            output_dtype=output_dtype,
            skeleton=skeleton,
        )
    if protocol not in {"pose_authoritative_v1", "v1", "legacy"}:
        raise ValueError(f"unknown finalization protocol: {protocol!r}")
    validate_motion_tensor(motion)
    if (input_standardized or output_standardized) and (mean is None or std is None):
        raise ValueError("mean and std must be supplied for standardized input/output")
    if not (input_standardized or output_standardized) and (mean is not None or std is not None):
        raise ValueError("mean and std require an explicit standardization flag")
    physical = _physical_motion(motion, mean, std) if input_standardized else motion
    if valid_mask is not None:
        expected = physical.shape[:-1]
        if tuple(valid_mask.shape) != tuple(expected) or valid_mask.dtype is not torch.bool:
            raise ValueError(f"valid_mask must have shape {tuple(expected)} and dtype bool")
    if physical.ndim == 3:
        finalized = [
            _single_finalize(physical[i], None if valid_mask is None else valid_mask[i])
            for i in range(physical.shape[0])
        ]
        result_motion = torch.stack([item.motion for item in finalized], dim=0)
        result_mask = torch.stack([item.valid_mask for item in finalized], dim=0)
    else:
        result = _single_finalize(physical, valid_mask)
        result_motion, result_mask = result.motion, result.valid_mask
    if output_standardized:
        result_motion = _standardized_motion(result_motion, mean, std)
        result_motion = result_motion.masked_fill(~result_mask.unsqueeze(-1), 0)
    if output_dtype is not None:
        result_motion = result_motion.to(output_dtype)
    return FinalizedMotion(motion=result_motion, valid_mask=result_mask)


def finalize_fk_consistent_motion_tensor(motion: torch.Tensor, **kwargs) -> FinalizedMotion:
    """Explicit v2 name; retained alongside the pose-authoritative v1 API."""

    kwargs["protocol"] = "vimogen_276d_consistency_v2"
    return finalize_consistent_motion_tensor(motion, **kwargs)


__all__ = ["finalize_consistent_motion_tensor", "finalize_fk_consistent_motion_tensor"]
