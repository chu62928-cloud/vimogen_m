"""Sequence-first pelvis control metrics for the scale study."""

from __future__ import annotations

import math
from typing import Any

import torch

from geometry.pelvis_angle import angle_error_deg, pelvis_angle_curve_deg


EVALUATOR_VERSION = "m1_m7_control_metrics_v1"


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def evaluate_control_metrics(
    motion: torch.Tensor,
    baseline_motion: torch.Tensor,
    target_angle_curve_deg: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    target_dose_deg: float,
    frame_tolerance_deg: float = 2.0,
) -> dict[str, Any]:
    if motion.shape != baseline_motion.shape:
        raise ValueError("motion and baseline_motion must have identical shape")
    if valid_mask.dtype is not torch.bool or valid_mask.shape != motion.shape[:2]:
        raise ValueError("valid_mask must be bool[B,T]")
    error = angle_error_deg(motion, target_angle_curve_deg)
    actual_delta = pelvis_angle_curve_deg(motion) - pelvis_angle_curve_deg(
        baseline_motion
    )
    rows: list[dict[str, Any]] = []
    for index in range(motion.shape[0]):
        selected_error = error[index][valid_mask[index]]
        selected_delta = actual_delta[index][valid_mask[index]]
        if not selected_error.numel():
            raise ValueError("every sequence needs at least one valid frame")
        absolute = selected_error.abs()
        mean_delta = selected_delta.mean()
        if abs(float(target_dose_deg)) <= 1.0e-8:
            sign_correct = None
        else:
            sign_correct = bool(
                torch.sign(mean_delta)
                == torch.sign(
                    torch.as_tensor(target_dose_deg, device=mean_delta.device)
                )
            )
        row = {
            "sequence_index": index,
            "valid_frames": int(selected_error.numel()),
            "angle_mae_deg": _scalar(absolute.mean()),
            "angle_p95_deg": _scalar(torch.quantile(absolute, 0.95)),
            "angle_max_deg": _scalar(absolute.max()),
            "actual_mean_dose_deg": _scalar(mean_delta),
            "sign_correct": sign_correct,
            "frame_pass_rate": _scalar(
                (absolute <= float(frame_tolerance_deg)).float().mean()
            ),
            "sequence_angle_pass": bool(
                absolute.mean() <= 1.0
                and torch.quantile(absolute, 0.95) <= float(frame_tolerance_deg)
                and (sign_correct is not False)
            ),
            "zero_dose_drift_mae_deg": (
                _scalar(selected_delta.abs().mean())
                if abs(float(target_dose_deg)) <= 1.0e-8
                else None
            ),
        }
        rows.append(row)
    mae = torch.tensor([row["angle_mae_deg"] for row in rows])
    p95 = torch.tensor([row["angle_p95_deg"] for row in rows])
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "target_dose_deg": float(target_dose_deg),
        "frame_tolerance_deg": float(frame_tolerance_deg),
        "per_sequence": rows,
        "summary": {
            "sequence_count": len(rows),
            "angle_mae_median_deg": _scalar(mae.median()),
            "angle_mae_mean_deg": _scalar(mae.mean()),
            "angle_p95_median_deg": _scalar(p95.median()),
            "sequence_angle_pass_rate": sum(
                int(row["sequence_angle_pass"]) for row in rows
            )
            / len(rows),
            "nonfinite_count": int(
                sum(
                    not math.isfinite(float(row[key]))
                    for row in rows
                    for key in ("angle_mae_deg", "angle_p95_deg", "angle_max_deg")
                )
            ),
        },
    }


def dose_response_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Fit response slope and paired sign symmetry from sequence rows."""

    nonzero = [row for row in rows if abs(float(row["target_dose_deg"])) > 1.0e-8]
    if len(nonzero) < 2:
        return {"dose_response_slope": None, "sign_symmetry_gap_deg": None}
    target = torch.tensor([float(row["target_dose_deg"]) for row in nonzero])
    actual = torch.tensor([float(row["actual_mean_dose_deg"]) for row in nonzero])
    centred_target = target - target.mean()
    denominator = centred_target.square().sum()
    slope = (
        ((actual - actual.mean()) * centred_target).sum() / denominator
        if denominator > 0
        else torch.tensor(float("nan"))
    )
    gaps: list[float] = []
    by_dose = {float(row["target_dose_deg"]): row for row in nonzero}
    for dose, row in by_dose.items():
        if dose > 0 and -dose in by_dose:
            gaps.append(
                abs(
                    float(row["actual_mean_dose_deg"])
                    + float(by_dose[-dose]["actual_mean_dose_deg"])
                )
            )
    return {
        "dose_response_slope": float(slope) if torch.isfinite(slope) else None,
        "sign_symmetry_gap_deg": sum(gaps) / len(gaps) if gaps else None,
    }
