"""Content-preservation and temporal diagnostics relative to paired M0."""

from __future__ import annotations

from typing import Any

import torch

from geometry.trunk import trunk_direction_deviation_deg
from motion_rep.phase1 import MOTION_LAYOUT


EVALUATOR_VERSION = "m1_m7_content_metrics_v1"


def _p95(value: torch.Tensor) -> float:
    return float(torch.quantile(value, 0.95).detach().cpu()) if value.numel() else 0.0


def _difference(value: torch.Tensor, order: int) -> torch.Tensor:
    result = value
    for _ in range(order):
        result = result[:, 1:] - result[:, :-1]
    return result


def evaluate_content_metrics(
    motion: torch.Tensor,
    baseline_motion: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, Any]:
    if motion.shape != baseline_motion.shape or motion.ndim != 3:
        raise ValueError("motion and baseline_motion must share [B,T,276]")
    joints = motion[..., MOTION_LAYOUT.joints].reshape(*motion.shape[:2], 22, 3)
    baseline_joints = baseline_motion[..., MOTION_LAYOUT.joints].reshape(
        *motion.shape[:2], 22, 3
    )
    root = motion[..., MOTION_LAYOUT.root_translation]
    baseline_root = baseline_motion[..., MOTION_LAYOUT.root_translation]
    trunk = trunk_direction_deviation_deg(joints, baseline_joints)
    rows = []
    for batch in range(motion.shape[0]):
        length = int(valid_mask[batch].sum().item())
        joint_delta = torch.linalg.vector_norm(
            joints[batch, :length] - baseline_joints[batch, :length], dim=-1
        )
        root_delta = torch.linalg.vector_norm(
            root[batch, :length] - baseline_root[batch, :length], dim=-1
        )
        temporal: dict[str, float] = {}
        for order, name in ((1, "velocity"), (2, "acceleration"), (3, "jerk")):
            if length <= order:
                temporal[f"joint_{name}_deviation_p95_mm"] = 0.0
                continue
            candidate_diff = _difference(joints[batch : batch + 1, :length], order)
            baseline_diff = _difference(
                baseline_joints[batch : batch + 1, :length], order
            )
            difference = torch.linalg.vector_norm(
                candidate_diff - baseline_diff, dim=-1
            ).reshape(-1)
            temporal[f"joint_{name}_deviation_p95_mm"] = _p95(difference) * 1000.0
        rows.append(
            {
                "sequence_index": batch,
                "mpjpe_vs_m0_mm": float(joint_delta.mean().cpu()) * 1000.0,
                "mpjpe_vs_m0_p95_mm": _p95(joint_delta.reshape(-1)) * 1000.0,
                "root_translation_deviation_p95_mm": _p95(root_delta) * 1000.0,
                "trunk_direction_deviation_p95_deg": _p95(trunk[batch, :length]),
                "end_frame_joint_jump_deviation_p95_mm": (
                    _p95(
                        torch.linalg.vector_norm(
                            (joints[batch, length - 1] - joints[batch, length - 2])
                            - (
                                baseline_joints[batch, length - 1]
                                - baseline_joints[batch, length - 2]
                            ),
                            dim=-1,
                        )
                    )
                    * 1000.0
                    if length > 1
                    else 0.0
                ),
                **temporal,
            }
        )
    return {"evaluator_version": EVALUATOR_VERSION, "per_sequence": rows}
