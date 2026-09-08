"""Shared sequence-first ground and contact metrics."""

from __future__ import annotations

from typing import Any, Mapping

import torch


EVALUATOR_VERSION = "m1_m7_physical_metrics_v1"

NOT_EVALUATED = "NOT_EVALUATED"
REFERENCE_MISSING = "REFERENCE_MISSING"
EVAL_ERROR = "EVAL_ERROR"
EVALUATED_FAIL = "EVALUATED_FAIL"
EVALUATED_PASS = "EVALUATED_PASS"


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


def _empty_summary() -> dict[str, float | None]:
    return {"mean": None, "p95": None, "max": None}


def _scalar_summary(values: list[torch.Tensor], *, scale: float = 1.0) -> dict[str, float | None]:
    if not values:
        return _empty_summary()
    flattened = torch.cat([value.reshape(-1) for value in values])
    flattened = flattened[torch.isfinite(flattened)]
    if not flattened.numel():
        return _empty_summary()
    flattened = flattened * scale
    return {
        "mean": float(flattened.mean().detach().cpu()),
        "p95": float(torch.quantile(flattened, 0.95).detach().cpu()),
        "max": float(flattened.max().detach().cpu()),
    }


def _marker_key(side: str, marker: str) -> str:
    return f"{side}_{marker}"


def _validate_marker_mapping(
    value: Mapping[str, Mapping[str, torch.Tensor]],
    valid_mask: torch.Tensor,
    name: str,
) -> None:
    if set(value) != {"left", "right"}:
        raise ValueError(f"{name} must contain left and right")
    expected = (*valid_mask.shape, 3)
    for side in ("left", "right"):
        if set(value[side]) != {"heel", "toe"}:
            raise ValueError(f"{name}[{side!r}] must contain heel and toe")
        for marker in ("heel", "toe"):
            tensor = value[side][marker]
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name}[{side!r}][{marker!r}] must have shape {expected}")
            if not torch.is_floating_point(tensor):
                raise TypeError(f"{name}[{side!r}][{marker!r}] must be floating point")


def _longest_segment_distance(
    contact: torch.Tensor,
    speed: torch.Tensor,
) -> torch.Tensor:
    """Return the longest sum of pair distances in a contact run."""

    if contact.ndim != 1 or speed.ndim != 1 or speed.shape[0] != max(contact.shape[0] - 1, 0):
        raise ValueError("contact/speed must describe one sequence")
    best = speed.new_zeros(())
    current = speed.new_zeros(())
    for index in range(speed.shape[0]):
        if bool(contact[index] & contact[index + 1]):
            current = current + speed[index]
            best = torch.maximum(best, current)
        else:
            current = speed.new_zeros(())
    return best


def _threshold_decision(
    row: Mapping[str, Any], thresholds: Mapping[str, float] | None
) -> tuple[str, bool | None, list[str]]:
    if thresholds is None:
        return NOT_EVALUATED, None, ["THRESHOLDS_NOT_FROZEN"]
    if not row["contact_evaluable"]:
        return NOT_EVALUATED, None, ["NO_CONTINUOUS_CONTACT_EVIDENCE"]
    failures: list[str] = []
    for name, limit in thresholds.items():
        value = row.get(name)
        if value is None or not torch.isfinite(torch.as_tensor(value)):
            return EVAL_ERROR, False, [f"MISSING_OR_NONFINITE_{name}"]
        if float(value) > float(limit):
            failures.append(f"{name}_fail")
    if failures:
        return EVALUATED_FAIL, False, failures
    return EVALUATED_PASS, True, []


def evaluate_physical_metrics_v2(
    marker_positions: Mapping[str, Mapping[str, torch.Tensor]],
    baseline_markers: Mapping[str, Mapping[str, torch.Tensor]],
    valid_mask: torch.Tensor,
    contact_masks: Mapping[str, torch.Tensor],
    ground_height_m: torch.Tensor,
    *,
    contact_pair_masks: Mapping[str, torch.Tensor] | None = None,
    up_axis: int = 2,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate frozen-M0 heel/toe evidence without letting candidates relabel it.

    This is the versioned S0-closure evaluator.  It reports raw metrics even
    when physical thresholds are not yet frozen.  In that case the state is
    ``NOT_EVALUATED`` and never a pass.  ``contact_masks`` and
    ``contact_pair_masks`` are always supplied by the paired M0 reference.
    """

    if valid_mask.dtype is not torch.bool or valid_mask.ndim != 2:
        raise ValueError("valid_mask must be bool[B,T]")
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be 0, 1, or 2")
    _validate_marker_mapping(marker_positions, valid_mask, "marker_positions")
    _validate_marker_mapping(baseline_markers, valid_mask, "baseline_markers")
    batch, frames = valid_mask.shape
    if tuple(ground_height_m.shape) != (batch,):
        raise ValueError("ground_height_m must have shape [B]")
    if not torch.isfinite(ground_height_m).all():
        raise ValueError("ground_height_m must be finite")
    if set(contact_masks) != {"left", "right"}:
        raise ValueError("contact_masks must contain left and right")
    for side in ("left", "right"):
        if contact_masks[side].dtype is not torch.bool or tuple(contact_masks[side].shape) != (batch, frames):
            raise ValueError(f"contact_masks[{side!r}] must be bool[B,T]")
    if contact_pair_masks is None:
        contact_pair_masks = {
            side: contact_masks[side][:, 1:] & contact_masks[side][:, :-1]
            for side in ("left", "right")
        }
    if set(contact_pair_masks) != {"left", "right"}:
        raise ValueError("contact_pair_masks must contain left and right")
    for side in ("left", "right"):
        if contact_pair_masks[side].dtype is not torch.bool or tuple(contact_pair_masks[side].shape) != (batch, max(frames - 1, 0)):
            raise ValueError(f"contact_pair_masks[{side!r}] must be bool[B,T-1]")

    rows: list[dict[str, Any]] = []
    for index in range(batch):
        valid = valid_mask[index]
        floor = ground_height_m[index]
        penetration_values: list[torch.Tensor] = []
        speed_values: list[torch.Tensor] = []
        height_values: list[torch.Tensor] = []
        floating_values: list[torch.Tensor] = []
        support_values: list[torch.Tensor] = []
        marker_rows: dict[str, dict[str, Any]] = {}
        total_slide = torch.zeros((), dtype=ground_height_m.dtype, device=ground_height_m.device)
        longest_slide = torch.zeros_like(total_slide)
        contact_count = 0

        for side in ("left", "right"):
            contact = contact_masks[side][index] & valid
            pairs = contact_pair_masks[side][index] & valid[1:] & valid[:-1]
            for marker in ("heel", "toe"):
                key = _marker_key(side, marker)
                candidate = marker_positions[side][marker][index]
                baseline = baseline_markers[side][marker][index]
                if not torch.isfinite(candidate).all() or not torch.isfinite(baseline).all():
                    raise FloatingPointError(f"non-finite {key} marker")
                penetration = (floor - candidate[:, up_axis]).clamp_min(0.0)
                penetration_values.append(penetration[valid])
                delta = candidate[1:] - candidate[:-1]
                delta = delta.clone()
                delta[:, up_axis] = 0.0
                speed = torch.linalg.vector_norm(delta, dim=-1)
                speed_on_pairs = speed[pairs]
                speed_values.append(speed_on_pairs)
                slide = speed_on_pairs.sum() if speed_on_pairs.numel() else speed.new_zeros(())
                segment_slide = _longest_segment_distance(contact, speed)
                total_slide = total_slide + slide
                longest_slide = torch.maximum(longest_slide, segment_slide)
                marker_contact = int(contact.sum().item())
                contact_count += marker_contact
                if marker_contact:
                    candidate_height = (candidate[:, up_axis] - floor).clamp_min(0.0)
                    baseline_height = (baseline[:, up_axis] - floor).clamp_min(0.0)
                    contact_height = candidate_height[contact]
                    floating = (candidate_height[contact] > 0.0).to(candidate.dtype)
                    support = (candidate_height - baseline_height).abs()[contact]
                    height_values.append(contact_height)
                    floating_values.append(floating)
                    support_values.append(support)
                marker_rows[key] = {
                    "penetration_mm": _scalar_summary([penetration[valid]], scale=1000.0),
                    "contact_tangent_speed_mm_per_frame": _scalar_summary([speed_on_pairs], scale=1000.0),
                    "contact_height_mm": _scalar_summary(
                        [height_values[-1]] if marker_contact else [], scale=1000.0
                    ),
                    "support_height_error_mm": _scalar_summary(
                        [support_values[-1]] if marker_contact else [], scale=1000.0
                    ),
                    "contact_frames": marker_contact,
                    "continuous_contact_pairs": int(pairs.sum().item()),
                    "slide_distance_mm": float(slide.detach().cpu() * 1000.0),
                    "longest_contact_segment_slide_mm": float(
                        segment_slide.detach().cpu() * 1000.0
                    ),
                }

        penetration = _scalar_summary(penetration_values, scale=1000.0)
        speed = _scalar_summary(speed_values, scale=1000.0)
        height = _scalar_summary(height_values, scale=1000.0)
        floating_rate = (
            float(torch.cat(floating_values).mean().detach().cpu())
            if floating_values
            else None
        )
        support = _scalar_summary(support_values, scale=1000.0)
        row: dict[str, Any] = {
            "sequence_index": index,
            "marker_metrics": marker_rows,
            "penetration_p95_mm": penetration["p95"],
            "penetration_max_mm": penetration["max"],
            "penetration_frame_rate": float(
                torch.cat(penetration_values).gt(0.0).to(torch.float32).mean().cpu()
            ) if penetration_values else None,
            "contact_tangent_speed_p95_mm_per_frame": speed["p95"],
            "contact_tangent_speed_max_mm_per_frame": speed["max"],
            "total_slide_distance_mm": float(total_slide.detach().cpu() * 1000.0),
            "max_segment_slide_distance_mm": float(longest_slide.detach().cpu() * 1000.0),
            "contact_height_p95_mm": height["p95"],
            "contact_height_max_mm": height["max"],
            "floating_frame_rate": floating_rate,
            "support_height_error_p95_mm": support["p95"],
            "contact_evaluable": bool(speed_values and any(value.numel() for value in speed_values)),
            "valid_frames": int(valid.sum().item()),
            "contact_frames": contact_count,
        }
        status, physical_pass, reasons = _threshold_decision(row, thresholds)
        row.update(
            {
                "status": status,
                "physical_pass": physical_pass,
                "physical_fail_reasons": reasons,
            }
        )
        rows.append(row)

    statuses = [row["status"] for row in rows]
    if any(status == EVAL_ERROR for status in statuses):
        overall = EVAL_ERROR
    elif any(status == EVALUATED_FAIL for status in statuses):
        overall = EVALUATED_FAIL
    elif statuses and all(status == EVALUATED_PASS for status in statuses):
        overall = EVALUATED_PASS
    else:
        overall = NOT_EVALUATED
    physical_pass: bool | None
    if thresholds is None:
        physical_pass = None
    else:
        physical_pass = bool(rows) and all(row["physical_pass"] is True for row in rows)
    reasons = sorted(
        {
            reason
            for row in rows
            for reason in row.get("physical_fail_reasons", [])
        }
    )
    return {
        "evaluator_version": "m1_m7_physical_metrics_v2",
        "status": overall,
        "reason": reasons[0] if len(reasons) == 1 else reasons,
        "physical_pass": physical_pass,
        "thresholds": None if thresholds is None else dict(thresholds),
        "per_sequence": rows,
    }
