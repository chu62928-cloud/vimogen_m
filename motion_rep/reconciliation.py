"""Control-aware reconciliation for ViMoGen's physical 276-D motion.

The packed representation contains a direct pose view and a redundant
forward-velocity view.  This module makes the intended authority explicit:
the velocity reconstruction supplies short-term dynamics, while a smoothed
direct-minus-velocity correction restores the long-term direct anchor.  The
correction, rather than the raw trajectory, is smoothed.  A final canonical
``T+1 -> T x 276`` conversion then recomputes every velocity channel from the
single reconciled pose stream.

The function is deliberately offline and deterministic.  It does not alter
the legacy sampler unless a caller explicitly invokes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .finalize import FinalizedMotion, finalize_motion
from .phase1 import (
    MOTION_LAYOUT,
    decode_rot6d_safe,
    destandardize_motion,
    standardize_motion,
    validate_motion_tensor,
)
from .rotation_transform import axis_angle_to_mat3x3, mat3x3_to_axis_angle


@dataclass(frozen=True)
class ReconciliationConfig:
    """Frozen numerical choices for one reconciliation protocol."""

    correction_window: int = 9
    anchor_weight: float = 1.0
    root_rotation_anchor_weight: float = 1.0
    # Explicit version selector.  The default remains byte-for-byte v1; v2
    # routes to the separate full-FK implementation and never silently
    # changes historical reconciliation semantics.
    protocol: str = "vimogen_276d_control_aware_reconciliation_v1"

    def __post_init__(self) -> None:
        if self.correction_window < 1 or self.correction_window % 2 == 0:
            raise ValueError("correction_window must be a positive odd integer")
        if not 0.0 <= self.anchor_weight <= 1.0:
            raise ValueError("anchor_weight must lie in [0,1]")
        if not 0.0 <= self.root_rotation_anchor_weight <= 1.0:
            raise ValueError("root_rotation_anchor_weight must lie in [0,1]")
        if self.protocol not in {
            "vimogen_276d_control_aware_reconciliation_v1",
            "legacy_v1",
            "vimogen_276d_consistency_v2",
            "fk_v2",
            "v2",
        }:
            raise ValueError(f"unknown reconciliation protocol: {self.protocol!r}")


@dataclass(frozen=True)
class ReconciledMotion(FinalizedMotion):
    """Canonical motion plus an auditable reconciliation description."""

    protocol: str = "vimogen_276d_control_aware_reconciliation_v1"
    correction_window: int = 9
    anchor_weight: float = 1.0
    root_rotation_anchor_weight: float = 1.0
    input_was_standardized: bool = False
    output_is_standardized: bool = False


def _moving_average(values: torch.Tensor, window: int) -> torch.Tensor:
    if values.shape[-2] == 0:
        return values
    if window == 1:
        return values
    radius = window // 2
    left = values[..., :1, :].expand(*values.shape[:-2], radius, values.shape[-1])
    right = values[..., -1:, :].expand(*values.shape[:-2], radius, values.shape[-1])
    padded = torch.cat((left, values, right), dim=-2)
    kernel = padded.unfold(-2, window, 1).mean(dim=-1)
    return kernel


def _extrapolate_last(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-2] == 1:
        return values[-1:]
    return values[-1:] + (values[-1:] - values[-2:-1])


def _statistics_for_motion(
    statistic: torch.Tensor, motion: torch.Tensor, *, name: str
) -> torch.Tensor:
    """Broadcast channel statistics for single or batched motion tensors.

    The training/evaluation path stores one ``[B,276]`` mean/std row per
    sample, whereas the geometry helpers operate on ``[T,276]`` streams.
    Make that boundary explicit instead of relying on PyTorch's ambiguous
    trailing-dimension broadcasting (which would compare ``T`` with ``B``).
    """

    if statistic.ndim == 1:
        if statistic.shape[0] != MOTION_LAYOUT.total_dim:
            raise ValueError(f"{name} must have 276 channels")
        return statistic
    if statistic.ndim == 2 and statistic.shape[-1] == MOTION_LAYOUT.total_dim:
        if motion.ndim == 3 and statistic.shape[0] == motion.shape[0]:
            return statistic.unsqueeze(1)
        if motion.ndim == 2 and statistic.shape[0] == 1:
            return statistic[0]
    raise ValueError(
        f"{name} must have shape [276] or one [B,276] row per batched motion; "
        f"got {tuple(statistic.shape)} for motion {tuple(motion.shape)}"
    )


def _position_velocity_stream(
    direct: torch.Tensor, velocity: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return direct ``T+1`` and velocity-integrated ``T+1`` positions."""

    direct_stream = torch.cat((direct, _extrapolate_last(direct)), dim=0)
    velocity_stream = torch.cat((
        direct[:1],
        direct[:1] + torch.cumsum(velocity, dim=0),
    ), dim=0)
    return direct_stream, velocity_stream


def _reconcile_positions(
    direct: torch.Tensor,
    velocity: torch.Tensor,
    *,
    config: ReconciliationConfig,
    weight: float,
) -> torch.Tensor:
    direct_stream, velocity_stream = _position_velocity_stream(direct, velocity)
    correction = direct_stream - velocity_stream
    smoothed = _moving_average(correction, config.correction_window)
    return velocity_stream + float(weight) * smoothed


def _root_velocity_stream(
    direct: torch.Tensor, velocity_rot6d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    velocity = decode_rot6d_safe(velocity_rot6d)
    if direct.shape[0] > 1:
        last_increment = direct[-1:] @ direct[-2:-1].transpose(-1, -2)
        direct_next = last_increment @ direct[-1:]
    else:
        direct_next = direct[-1:]
    direct_stream = torch.cat((direct, direct_next), dim=0)
    frames = [direct[:1]]
    for index in range(velocity.shape[0]):
        frames.append(velocity[index:index + 1] @ frames[-1])
    return direct_stream, torch.cat(frames, dim=0)


def _reconcile_root_rotation(
    direct: torch.Tensor,
    velocity_rot6d: torch.Tensor,
    *,
    config: ReconciliationConfig,
    weight: float,
) -> torch.Tensor:
    direct_stream, velocity_stream = _root_velocity_stream(direct, velocity_rot6d)
    correction = direct_stream @ velocity_stream.transpose(-1, -2)
    correction_axis = mat3x3_to_axis_angle(correction)
    smoothed_axis = _moving_average(correction_axis, config.correction_window)
    # The correction is left-multiplied in the world/canonical frame.  This
    # keeps the root control convention explicit and avoids component-wise
    # interpolation of rotation matrices.
    return axis_angle_to_mat3x3(float(weight) * smoothed_axis) @ velocity_stream


def _single_reconcile(
    motion: torch.Tensor,
    *,
    config: ReconciliationConfig,
    component_weights: Mapping[str, float] | None,
    valid_mask: torch.Tensor | None,
) -> FinalizedMotion:
    frames = motion.shape[0]
    if frames < 1:
        raise ValueError("motion must contain at least one frame")
    body = decode_rot6d_safe(motion[:, MOTION_LAYOUT.body_pose].reshape(frames, 21, 6))
    body_stream = torch.cat((body, body[-1:]), dim=0)
    direct_joints = motion[:, MOTION_LAYOUT.joints].reshape(frames, 22, 3)
    joints_velocity = motion[:, MOTION_LAYOUT.joints_velocity].reshape(frames, 22, 3)
    direct_translation = motion[:, MOTION_LAYOUT.root_translation]
    translation_velocity = motion[:, MOTION_LAYOUT.root_translation_velocity]
    direct_root = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation])
    root_velocity = motion[:, MOTION_LAYOUT.root_rotation_velocity]
    weights = dict(component_weights or {})

    def component_weight(name: str, default: float) -> float:
        value = float(weights.get(name, default))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"component weight {name!r} must lie in [0,1]")
        return value

    joints = _reconcile_positions(
        direct_joints.reshape(frames, -1),
        joints_velocity.reshape(frames, -1),
        config=config,
        weight=component_weight("joint_positions", config.anchor_weight),
    ).reshape(frames + 1, 22, 3)
    translation = _reconcile_positions(
        direct_translation,
        translation_velocity,
        config=config,
        weight=component_weight("root_translation", config.anchor_weight),
    )
    root = _reconcile_root_rotation(
        direct_root,
        root_velocity,
        config=config,
        weight=component_weight("root_rotation", config.root_rotation_anchor_weight),
    )
    if valid_mask is None:
        pose_mask = torch.ones(frames + 1, dtype=torch.bool, device=motion.device)
    else:
        if valid_mask.shape != (frames,) or valid_mask.dtype is not torch.bool:
            raise ValueError(f"valid_mask must have shape {(frames,)} and dtype bool")
        pose_mask = torch.cat((valid_mask, valid_mask[-1:]))
    return finalize_motion(body_stream, joints, root, translation, valid_mask=pose_mask)


def reconcile_motion_tensor(
    motion: torch.Tensor,
    *,
    config: ReconciliationConfig | None = None,
    component_weights: Mapping[str, float] | None = None,
    valid_mask: torch.Tensor | None = None,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    input_standardized: bool = False,
    output_standardized: bool = False,
    output_dtype: torch.dtype | None = None,
) -> ReconciledMotion:
    """Reconcile physical or explicitly standardized ``[T,276]`` motion."""

    validate_motion_tensor(motion)
    if (input_standardized or output_standardized) and (mean is None or std is None):
        raise ValueError("mean and std must be supplied for standardized input/output")
    if not (input_standardized or output_standardized) and (mean is not None or std is not None):
        raise ValueError("mean and std require an explicit standardization flag")
    config = config or ReconciliationConfig()
    if config.protocol in {"vimogen_276d_consistency_v2", "fk_v2", "v2"}:
        from .consistency_v2 import reconcile_motion_tensor_v2

        if component_weights:
            raise ValueError("component_weights are not supported by FK consistency v2")
        return reconcile_motion_tensor_v2(
            motion,
            fusion_window=config.correction_window,
            anchor_weight=config.anchor_weight,
            root_rotation_anchor_weight=config.root_rotation_anchor_weight,
            valid_mask=valid_mask,
            mean=mean,
            std=std,
            input_standardized=input_standardized,
            output_standardized=output_standardized,
            output_dtype=output_dtype,
        )
    if input_standardized:
        mean_for_motion = _statistics_for_motion(mean, motion, name="mean")
        std_for_motion = _statistics_for_motion(std, motion, name="std")
        physical = destandardize_motion(motion, mean_for_motion, std_for_motion)
    else:
        physical = motion
    if valid_mask is not None:
        if tuple(valid_mask.shape) != tuple(physical.shape[:-1]) or valid_mask.dtype is not torch.bool:
            raise ValueError(f"valid_mask must have shape {tuple(physical.shape[:-1])} and dtype bool")
    if physical.ndim == 3:
        outputs = [
            _single_reconcile(
                physical[index],
                config=config,
                component_weights=component_weights,
                valid_mask=None if valid_mask is None else valid_mask[index],
            )
            for index in range(physical.shape[0])
        ]
        result_motion = torch.stack([item.motion for item in outputs], dim=0)
        result_mask = torch.stack([item.valid_mask for item in outputs], dim=0)
    else:
        result = _single_reconcile(
            physical,
            config=config,
            component_weights=component_weights,
            valid_mask=valid_mask,
        )
        result_motion, result_mask = result.motion, result.valid_mask
    if output_standardized:
        mean_for_motion = _statistics_for_motion(mean, result_motion, name="mean")
        std_for_motion = _statistics_for_motion(std, result_motion, name="std")
        result_motion = standardize_motion(result_motion, mean_for_motion, std_for_motion)
        result_motion = result_motion.masked_fill(~result_mask.unsqueeze(-1), 0)
    if output_dtype is not None:
        result_motion = result_motion.to(output_dtype)
    return ReconciledMotion(
        motion=result_motion,
        valid_mask=result_mask,
        correction_window=config.correction_window,
        anchor_weight=config.anchor_weight,
        root_rotation_anchor_weight=config.root_rotation_anchor_weight,
        input_was_standardized=input_standardized,
        output_is_standardized=output_standardized,
    )


__all__ = ["ReconciledMotion", "ReconciliationConfig", "reconcile_motion_tensor"]
