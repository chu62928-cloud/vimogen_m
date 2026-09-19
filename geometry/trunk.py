"""Shared trunk and head-direction diagnostics."""

from __future__ import annotations

import torch


TRUNK_VERSION = "m1_m7_trunk_direction_v1"


def _angle_deg(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    a = a / torch.linalg.vector_norm(a, dim=-1, keepdim=True).clamp_min(eps)
    b = b / torch.linalg.vector_norm(b, dim=-1, keepdim=True).clamp_min(eps)
    cross = torch.linalg.vector_norm(torch.cross(a, b, dim=-1), dim=-1)
    dot = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.atan2(cross, dot))


def trunk_direction_deviation_deg(
    joints: torch.Tensor,
    baseline_joints: torch.Tensor,
    *,
    spine1_index: int = 3,
    neck_index: int = 12,
) -> torch.Tensor:
    if joints.shape != baseline_joints.shape or joints.shape[-2:] != (22, 3):
        raise ValueError("joints and baseline_joints must share shape [B,T,22,3]")
    axis = joints[..., neck_index, :] - joints[..., spine1_index, :]
    baseline_axis = baseline_joints[..., neck_index, :] - baseline_joints[..., spine1_index, :]
    return _angle_deg(axis, baseline_axis)
