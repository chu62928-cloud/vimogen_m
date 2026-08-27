"""Tail-safe full 276-D consistency projection (protocol v3).

This module keeps the frozen v2 boundary intact and changes only the
end-of-sequence treatment.  The T physical output poses are fused and
smoothed without a synthetic T+1 pose entering the smoothing window.  The
hidden pose required by the forward-difference packer is held at the last
valid fused pose, so the final output frame cannot receive an extrapolation
gradient through a padded endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .consistency_v2 import (
    ConsistencyV2Config,
    FKResult,
    Skeleton22,
    _align_statistic,
    _resolve_skeleton,
    default_smplx_neutral_22_skeleton,
    differentiable_forward_kinematics,
    load_smplx_neutral_22_skeleton,
)
from .finalize import FinalizedMotion, finalize_motion
from .phase1 import MOTION_LAYOUT, decode_rot6d_safe, validate_motion_tensor
from .rotation_transform import axis_angle_to_mat3x3, mat3x3_to_axis_angle


PROTOCOL_NAME = "vimogen_276d_consistency_v3_tail_safe"


def _moving_average_truncated(values: torch.Tensor, window: int) -> torch.Tensor:
    """Centered moving average with a valid-window denominator at both ends."""

    if window == 1 or values.shape[-2] == 0:
        return values
    radius = window // 2
    pieces = []
    length = values.shape[-2]
    for index in range(length):
        start = max(0, index - radius)
        end = min(length, index + radius + 1)
        pieces.append(values[..., start:end, :].mean(dim=-2, keepdim=True))
    return torch.cat(pieces, dim=-2)


def _fuse_translation_tail_safe(
    direct: torch.Tensor, velocity: torch.Tensor, window: int, weight: float
) -> torch.Tensor:
    """Fuse only physical output poses, then hold the hidden endpoint."""

    velocity_stream = torch.cat(
        (direct[:1], direct[:1] + torch.cumsum(velocity, dim=0)), dim=0
    )
    correction = direct - velocity_stream[:-1]
    smoothed = _moving_average_truncated(correction, window)
    fused_output = velocity_stream[:-1] + float(weight) * smoothed
    return torch.cat((fused_output, fused_output[-1:]), dim=0)


def fuse_root_rotation_tail_safe(
    direct: torch.Tensor, velocity: torch.Tensor, window: int, weight: float
) -> torch.Tensor:
    """Fuse root SO(3) streams without allowing hidden T+1 feedback.

    ``direct`` contains T authoritative output rotations and ``velocity``
    contains T forward-difference rows.  The returned stream has T+1 poses;
    its final hidden pose is exactly the final output pose.
    """

    velocity_matrix = decode_rot6d_safe(velocity)
    frames = [direct[:1]]
    for index in range(velocity_matrix.shape[0]):
        frames.append(velocity_matrix[index : index + 1] @ frames[-1])
    velocity_stream = torch.cat(frames, dim=0)
    correction = direct @ velocity_stream[:-1].transpose(-1, -2)
    correction_axis = mat3x3_to_axis_angle(correction)
    smoothed = _moving_average_truncated(correction_axis, window)
    fused_output = axis_angle_to_mat3x3(float(weight) * smoothed) @ velocity_stream[:-1]
    return torch.cat((fused_output, fused_output[-1:]), dim=0)


@dataclass(frozen=True)
class FKConsistentMotionV3(FinalizedMotion):
    """Packed v3 result and protocol metadata."""

    protocol: str = PROTOCOL_NAME
    skeleton_source: str = "data/body_models/smplx/SMPLX_NEUTRAL.npz"


def _single_consistency_tail_safe(
    motion: torch.Tensor,
    *,
    tree: Skeleton22,
    fusion_window: int,
    anchor_weight: float,
    root_rotation_anchor_weight: float,
    valid_mask: torch.Tensor | None,
) -> FKConsistentMotionV3:
    validate_motion_tensor(motion)
    if motion.ndim != 2 or motion.shape[0] < 1:
        raise ValueError("motion must have shape [T,276] with T >= 1")
    frames = motion.shape[0]
    source = motion
    if valid_mask is None:
        pose_mask = torch.ones(frames + 1, dtype=torch.bool, device=motion.device)
    else:
        if valid_mask.shape != (frames,) or valid_mask.dtype is not torch.bool:
            raise ValueError(f"valid_mask must have shape {(frames,)} and dtype bool")
        if torch.any(valid_mask[1:] & ~valid_mask[:-1]):
            raise ValueError("valid_mask must be a contiguous valid prefix followed by tail padding")
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        pose_mask = torch.zeros(frames + 1, dtype=torch.bool, device=motion.device)
        pose_mask[:frames] = valid_mask
        if valid_indices.numel():
            last = int(valid_indices[-1].item())
            if last + 1 < frames:
                source = motion.clone()
                source[last + 1 :] = source[last : last + 1]
            pose_mask[last + 1] = True

    body = decode_rot6d_safe(source[:, MOTION_LAYOUT.body_pose].reshape(frames, 21, 6)).float()
    body_stream = torch.cat((body, body[-1:]), dim=0)
    direct_root = decode_rot6d_safe(source[:, MOTION_LAYOUT.root_rotation]).float()
    fused_root = fuse_root_rotation_tail_safe(
        direct_root,
        source[:, MOTION_LAYOUT.root_rotation_velocity].float(),
        fusion_window,
        root_rotation_anchor_weight,
    )
    fused_translation = _fuse_translation_tail_safe(
        source[:, MOTION_LAYOUT.root_translation].float(),
        source[:, MOTION_LAYOUT.root_translation_velocity].float(),
        fusion_window,
        anchor_weight,
    )
    fk = differentiable_forward_kinematics(
        body_stream,
        fused_root,
        fused_translation,
        skeleton=tree,
    )
    finalized = finalize_motion(
        body_stream,
        fk.joints,
        fused_root,
        fused_translation,
        valid_mask=pose_mask,
    )
    return FKConsistentMotionV3(
        motion=finalized.motion,
        valid_mask=finalized.valid_mask,
        protocol=PROTOCOL_NAME,
        skeleton_source=tree.source,
    )


def reconcile_motion_tensor_v3(
    motion: torch.Tensor,
    *,
    config: ConsistencyV2Config | None = None,
    fusion_window: int = 9,
    anchor_weight: float = 1.0,
    root_rotation_anchor_weight: float | None = None,
    valid_mask: torch.Tensor | None = None,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    input_standardized: bool = False,
    output_standardized: bool = False,
    output_dtype: torch.dtype | None = None,
    skeleton: Skeleton22 | dict | None = None,
    rest_offsets: torch.Tensor | None = None,
    parents: tuple[int, ...] | list[int] | torch.Tensor | None = None,
) -> FKConsistentMotionV3:
    """Project physical or standardized [T,276]/[B,T,276] data with v3."""

    if config is not None:
        fusion_window = config.fusion_window
        anchor_weight = config.anchor_weight
        root_rotation_anchor_weight = config.root_rotation_anchor_weight
        if skeleton is None and config.skeleton_path is not None:
            skeleton = load_smplx_neutral_22_skeleton(config.skeleton_path)
    validate_motion_tensor(motion)
    if fusion_window < 1 or fusion_window % 2 == 0:
        raise ValueError("fusion_window must be a positive odd integer")
    if not 0 <= anchor_weight <= 1:
        raise ValueError("anchor_weight must lie in [0,1]")
    root_weight = anchor_weight if root_rotation_anchor_weight is None else float(root_rotation_anchor_weight)
    if not 0 <= root_weight <= 1:
        raise ValueError("root_rotation_anchor_weight must lie in [0,1]")
    if (input_standardized or output_standardized) and (mean is None or std is None):
        raise ValueError("mean and std must be supplied for standardized input/output")
    if not (input_standardized or output_standardized) and (mean is not None or std is not None):
        raise ValueError("mean and std require an explicit standardization flag")
    if input_standardized:
        mean_aligned = _align_statistic(mean, motion, "mean")
        std_aligned = _align_statistic(std, motion, "std")
        if not torch.isfinite(mean_aligned).all() or not torch.isfinite(std_aligned).all() or torch.any(std_aligned <= 0):
            raise ValueError("mean/std must be finite and std strictly positive")
        physical = (motion * std_aligned + mean_aligned).float()
    else:
        physical = motion.float()
    if valid_mask is not None and (tuple(valid_mask.shape) != tuple(physical.shape[:-1]) or valid_mask.dtype is not torch.bool):
        raise ValueError(f"valid_mask must have shape {tuple(physical.shape[:-1])} and dtype bool")
    tree = _resolve_skeleton(skeleton, rest_offsets, parents)
    if physical.ndim == 3:
        values = [
            _single_consistency_tail_safe(
                physical[index],
                tree=tree,
                fusion_window=fusion_window,
                anchor_weight=float(anchor_weight),
                root_rotation_anchor_weight=root_weight,
                valid_mask=None if valid_mask is None else valid_mask[index],
            )
            for index in range(physical.shape[0])
        ]
        result_motion = torch.stack([item.motion for item in values], dim=0)
        result_mask = torch.stack([item.valid_mask for item in values], dim=0)
    else:
        value = _single_consistency_tail_safe(
            physical,
            tree=tree,
            fusion_window=fusion_window,
            anchor_weight=float(anchor_weight),
            root_rotation_anchor_weight=root_weight,
            valid_mask=valid_mask,
        )
        result_motion, result_mask = value.motion, value.valid_mask
    if output_standardized:
        mean_aligned = _align_statistic(mean, result_motion, "mean")
        std_aligned = _align_statistic(std, result_motion, "std")
        result_motion = (result_motion - mean_aligned) / std_aligned
        result_motion = result_motion.masked_fill(~result_mask.unsqueeze(-1), 0)
    if output_dtype is not None:
        result_motion = result_motion.to(output_dtype)
    return FKConsistentMotionV3(
        motion=result_motion,
        valid_mask=result_mask,
        protocol=PROTOCOL_NAME,
        skeleton_source=tree.source,
    )


full_fk_consistency_v3 = reconcile_motion_tensor_v3


__all__ = [
    "ConsistencyV2Config",
    "FKConsistentMotionV3",
    "FKResult",
    "PROTOCOL_NAME",
    "Skeleton22",
    "default_smplx_neutral_22_skeleton",
    "differentiable_forward_kinematics",
    "fuse_root_rotation_tail_safe",
    "full_fk_consistency_v3",
    "load_smplx_neutral_22_skeleton",
    "reconcile_motion_tensor_v3",
]
