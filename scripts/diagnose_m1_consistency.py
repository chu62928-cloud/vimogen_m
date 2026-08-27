"""Small reusable summaries for M1 redundant-channel diagnostics."""

from __future__ import annotations

import math

import torch

from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe


def pearson_correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    a, b = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) == 0.0:
        return None
    value = float((a @ b / denom).item())
    if abs(value - 1.0) < 1e-12:
        return 1.0
    if abs(value + 1.0) < 1e-12:
        return -1.0
    return value


def _rotation_angle_degrees(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    rel = a @ b.transpose(-1, -2)
    cosine = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def summarize_source_consistency(motion: torch.Tensor) -> dict[str, object]:
    if motion.ndim != 2 or motion.shape[-1] != MOTION_LAYOUT.total_dim:
        raise ValueError("expected physical [T,276] motion")
    joints = motion[:, MOTION_LAYOUT.joints].reshape(-1, 22, 3)
    joints_v = motion[:, MOTION_LAYOUT.joints_velocity].reshape(-1, 22, 3)
    direct_joints_v = joints[1:] - joints[:-1]
    joints_residual = (direct_joints_v - joints_v[:-1]).norm(dim=-1)
    integrated_joints = joints[:1] + torch.cumsum(joints_v, dim=0)
    joints_position_error = (joints[1:] - integrated_joints[:-1]).norm(dim=-1)

    translation = motion[:, MOTION_LAYOUT.root_translation]
    translation_v = motion[:, MOTION_LAYOUT.root_translation_velocity]
    direct_translation_v = translation[1:] - translation[:-1]
    translation_residual = (direct_translation_v - translation_v[:-1]).norm(dim=-1)
    integrated_translation = translation[:1] + torch.cumsum(translation_v, dim=0)
    translation_position_error = (translation[1:] - integrated_translation[:-1]).norm(dim=-1)

    root = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation])
    root_v = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation_velocity])
    integrated_root = [root[:1]]
    for i in range(motion.shape[0]):
        integrated_root.append(root_v[i:i + 1] @ integrated_root[-1])
    integrated_root = torch.cat(integrated_root, dim=0)
    root_error = _rotation_angle_degrees(root[1:], integrated_root[1:-1])

    def mean(value: torch.Tensor) -> float:
        return float(value.mean().item())

    return {
        "joint_velocity_residual": {"mean_m_per_frame": mean(joints_residual)},
        "root_translation_velocity_residual": {"mean_m_per_frame": mean(translation_residual)},
        "joint_position_integrated_error": {"mean_m": mean(joints_position_error)},
        "root_translation_position_integrated_error": {"mean_m": mean(translation_position_error)},
        "root_rotation_direct_vs_integrated": {
            "mean_degrees": mean(root_error),
            "max_degrees": float(root_error.max().item()),
        },
    }
