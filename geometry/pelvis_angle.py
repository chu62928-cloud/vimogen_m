"""Frozen pelvis-angle semantics shared by all M1--M7 methods."""

from __future__ import annotations

import torch

from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe
from motion_rep.sagittal_pelvis_angle import pelvis_sagittal_tilt_degrees


EVALUATOR_VERSION = "pelvis_angle_local_sagittal_v1"


def pelvis_angle_curve_deg(motion: torch.Tensor) -> torch.Tensor:
    if motion.ndim != 3 or motion.shape[-1] != MOTION_LAYOUT.total_dim:
        raise ValueError("motion must have shape [B,T,276]")
    root = decode_rot6d_safe(motion[..., MOTION_LAYOUT.root_rotation])
    return pelvis_sagittal_tilt_degrees(root)


def target_angle_curve_deg(
    baseline_motion: torch.Tensor,
    target_dose_deg: float | torch.Tensor,
) -> torch.Tensor:
    baseline = pelvis_angle_curve_deg(baseline_motion)
    dose = torch.as_tensor(
        target_dose_deg, dtype=baseline.dtype, device=baseline.device
    )
    try:
        dose = torch.broadcast_to(dose, baseline.shape)
    except RuntimeError as error:
        raise ValueError("target dose is not broadcastable to [B,T]") from error
    return baseline + dose


def wrap_angle_deg(value: torch.Tensor) -> torch.Tensor:
    return torch.remainder(value + 180.0, 360.0) - 180.0


def angle_error_deg(
    motion: torch.Tensor,
    target_curve_deg: torch.Tensor,
) -> torch.Tensor:
    actual = pelvis_angle_curve_deg(motion)
    if target_curve_deg.shape != actual.shape:
        raise ValueError("target curve must match motion [B,T]")
    return wrap_angle_deg(actual - target_curve_deg)
