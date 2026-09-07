"""Shared sequence-first ground and contact metrics."""

from __future__ import annotations

from typing import Any, Mapping

import torch


EVALUATOR_VERSION = "m1_m7_physical_metrics_v1"


def _summary(values: torch.Tensor, scale: float = 1.0) -> dict[str, float | None]:
    values = values[torch.isfinite(values)]
    if not values.numel():
        return {"mean": None, "p95": None, "max": None}
    values = values * scale
    return {
        "mean": float(values.mean().detach().cpu()),
        "p95": float(torch.quantile(values, 0.95).detach().cpu()),
        "max": float(values.max().detach().cpu()),
    }


def evaluate_physical_metrics(
    marker_positions: Mapping[str, Mapping[str, torch.Tensor]],
    baseline_markers: Mapping[str, Mapping[str, torch.Tensor]],
    valid_mask: torch.Tensor,
    contact_masks: Mapping[str, torch.Tensor],
    ground_height_m: torch.Tensor,
    *,
    up_axis: int = 2,
    penetration_tolerance_m: float = 0.001,
) -> dict[str, Any]:
    """Evaluate candidate markers using contact labels frozen from M0."""

    if valid_mask.dtype is not torch.bool or valid_mask.ndim != 2:
        raise ValueError("valid_mask must be bool[B,T]")
    rows: list[dict[str, Any]] = []
    for batch in range(valid_mask.shape[0]):
        penetration_values: list[torch.Tensor] = []
        contact_speeds: list[torch.Tensor] = []
        slide_distances: list[torch.Tensor] = []
        floating_values: list[torch.Tensor] = []
        support_errors: list[torch.Tensor] = []
        for side in ("left", "right"):
            mask = contact_masks[side][batch] & valid_mask[batch]
            pair = mask[1:] & mask[:-1]
            for marker in ("heel", "toe"):
                candidate = marker_positions[side][marker][batch]
                baseline = baseline_markers[side][marker][batch]
                floor = ground_height_m[batch]
                penetration = (floor - candidate[..., up_axis]).clamp_min(0.0)
                penetration_values.append(penetration[valid_mask[batch]])
                if mask.any():
                    candidate_height = candidate[..., up_axis] - floor
                    baseline_height = baseline[..., up_axis] - floor
                    floating_values.append((candidate_height - baseline_height).clamp_min(0.0)[mask])
                    support_errors.append((candidate_height - baseline_height).abs()[mask])
                if pair.any():
                    delta = candidate[1:] - candidate[:-1]
                    delta[..., up_axis] = 0.0
                    speed = torch.linalg.vector_norm(delta, dim=-1)[pair]
                    contact_speeds.append(speed)
                    slide_distances.append(speed.sum().reshape(1))
        penetration = torch.cat(penetration_values)
        speed = torch.cat(contact_speeds) if contact_speeds else torch.empty(0)
        slide = torch.cat(slide_distances) if slide_distances else torch.empty(0)
        floating = torch.cat(floating_values) if floating_values else torch.empty(0)
        support = torch.cat(support_errors) if support_errors else torch.empty(0)
        rows.append(
            {
                "sequence_index": batch,
                "penetration_mm": _summary(penetration, 1000.0),
                "penetration_frame_rate": float(
                    (penetration > float(penetration_tolerance_m)).float().mean().cpu()
                ),
                "contact_tangential_speed_mm_per_frame": _summary(speed, 1000.0),
                "contact_slide_distance_mm": _summary(slide, 1000.0),
                "foot_floating_mm": _summary(floating, 1000.0),
                "support_height_error_mm": _summary(support, 1000.0),
                "contact_evaluable": bool(speed.numel()),
            }
        )
    return {"evaluator_version": EVALUATOR_VERSION, "per_sequence": rows}
