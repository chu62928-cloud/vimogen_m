"""Evaluation and acceptance gates for anatomical-local pelvis protocol v4."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import torch

from motion_rep.anatomical_pelvis import anti_cheat_metrics, load_pelvis_calibration
from motion_rep.consistency_v3 import differentiable_forward_kinematics
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe
from evaluation.absolute_mean_pelvis_v3 import audit_motion_consistency, _correlation
from sampling.absolute_mean_pelvis_guidance_v4 import anatomical_angle_curves


v4_PROTOCOL = "vimogen_absolute_mean_pelvis_v4_anatomical_local"
DEFAULT_ANTI_CHEAT_LIMITS = {"soft_limit_deg": 2.0, "p95_limit_deg": 3.0, "low_signal_deg": 0.5, "ratio_floor_deg": 0.25}


def _validate_motion(motion: torch.Tensor, name: str) -> None:
    if not isinstance(motion, torch.Tensor) or motion.ndim != 2 or motion.shape[-1] != 276:
        raise ValueError(f"{name} must have shape [T,276]")
    if motion.shape[0] < 1 or not torch.isfinite(motion).all():
        raise ValueError(f"{name} must contain finite frames")


def _validate_mask(mask: torch.Tensor, frames: int) -> torch.Tensor:
    if mask.shape != (frames,) or mask.dtype is not torch.bool:
        raise ValueError(f"valid_mask must be bool[{frames}]")
    if not bool(mask.any()):
        raise ValueError("valid_mask must contain at least one valid frame")
    count = int(mask.sum().detach().cpu())
    expected = torch.zeros_like(mask)
    expected[:count] = True
    if not torch.equal(mask, expected):
        raise ValueError("valid_mask must be a contiguous valid prefix")
    return mask


def evaluate_single(
    *,
    sample_id: str,
    method: str,
    seed: int,
    target_mean_deg: float,
    baseline_phys: torch.Tensor,
    candidate_phys: torch.Tensor,
    valid_mask: torch.Tensor,
    calibration: Any,
    terminal_record: dict[str, Any] | None = None,
    skeleton: Any = None,
    rest_offsets: torch.Tensor | None = None,
    parents: tuple[int, ...] | list[int] | torch.Tensor | None = None,
) -> dict[str, Any]:
    """Evaluate target, curve preservation, local dominance, and consistency."""

    _validate_motion(baseline_phys, "baseline_phys")
    _validate_motion(candidate_phys, "candidate_phys")
    if baseline_phys.shape != candidate_phys.shape:
        raise ValueError("baseline_phys and candidate_phys must have the same shape")
    mask = _validate_mask(valid_mask, candidate_phys.shape[0])
    if isinstance(calibration, (str, bytes)):
        calibration = load_pelvis_calibration(calibration)
    baseline = anatomical_angle_curves(baseline_phys.float(), calibration)
    candidate = anatomical_angle_curves(candidate_phys.float(), calibration)
    valid = mask & baseline["pelvis_valid"] & candidate["pelvis_valid"]
    if not bool(valid.any()):
        raise ValueError("no valid anatomical pelvis frames")
    baseline_curve = baseline["pelvis_deg"][valid]
    candidate_curve = candidate["pelvis_deg"][valid]
    baseline_centered = baseline_curve - baseline_curve.mean()
    candidate_centered = candidate_curve - candidate_curve.mean()
    baseline_std = baseline_centered.std(unbiased=False)
    candidate_std = candidate_centered.std(unbiased=False)
    anti = anti_cheat_metrics(
        baseline["pelvis_deg"], candidate["pelvis_deg"],
        baseline["trunk_deg"], candidate["trunk_deg"],
        baseline["thigh_left_deg"], candidate["thigh_left_deg"],
        baseline["thigh_right_deg"], candidate["thigh_right_deg"], valid,
    )
    p95_gate = all(float(anti[key]) <= 3.0 for key in ("trunk_abs_p95_deg", "thigh_left_abs_p95_deg", "thigh_right_abs_p95_deg"))
    local_gate = bool(anti["local_change_low_signal"]) or (bool(anti["local_change_same_sign"]) and float(anti["local_change_share"]) > 0.5)
    consistency = audit_motion_consistency(candidate_phys, mask, skeleton=skeleton, rest_offsets=rest_offsets, parents=parents)
    row: dict[str, Any] = {
        "protocol": v4_PROTOCOL,
        "angle_authority": "calibrated_ASIS_PSIS_anatomical_local_sagittal_v4",
        "sample_id": sample_id,
        "method": method,
        "seed": int(seed),
        "target_mean_deg": float(target_mean_deg),
        "achieved_mean_deg": float(candidate_curve.mean().detach().cpu()),
        "absolute_mean_error_deg": abs(float(candidate_curve.mean().detach().cpu()) - float(target_mean_deg)),
        "centered_curve_rmse_deg": float(torch.sqrt((candidate_centered - baseline_centered).square().mean()).detach().cpu()),
        "centered_curve_correlation": _correlation(baseline_centered, candidate_centered),
        "fluctuation_std_ratio": float((candidate_std / baseline_std.clamp_min(1e-8)).detach().cpu()),
        "motion_rms_from_m0": float(torch.sqrt((candidate_phys[valid].float() - baseline_phys[valid].float()).square().mean()).detach().cpu()),
        "valid_frames": int(valid.sum().detach().cpu()),
        "delta_pelvis_mean_deg": float(anti["delta_pelvis_mean_deg"]),
        "delta_trunk_mean_deg": float(anti["delta_trunk_mean_deg"]),
        "delta_pelvis_trunk_mean_deg": float(anti["delta_local_mean_deg"]),
        "local_change_share": float(anti["local_change_share"]),
        "local_change_same_sign": bool(anti["local_change_same_sign"]),
        "local_change_low_signal": bool(anti["local_change_low_signal"]),
        "trunk_delta_p95_deg": float(anti["trunk_abs_p95_deg"]),
        "thigh_left_delta_p95_deg": float(anti["thigh_left_abs_p95_deg"]),
        "thigh_right_delta_p95_deg": float(anti["thigh_right_abs_p95_deg"]),
        "trunk_delta_median_deg": float(anti["delta_trunk_deg"][valid].abs().median()),
        "thigh_left_delta_median_deg": float(anti["delta_thigh_left_deg"][valid].abs().median()),
        "thigh_right_delta_median_deg": float(anti["delta_thigh_right_deg"][valid].abs().median()),
        "trunk_delta_max_deg": float(anti["delta_trunk_deg"][valid].abs().max()),
        "thigh_left_delta_max_deg": float(anti["delta_thigh_left_deg"][valid].abs().max()),
        "thigh_right_delta_max_deg": float(anti["delta_thigh_right_deg"][valid].abs().max()),
        "over_2deg_trunk_frame_rate": float((anti["delta_trunk_deg"][valid].abs() > 2).float().mean()),
        "over_2deg_thigh_left_frame_rate": float((anti["delta_thigh_left_deg"][valid].abs() > 2).float().mean()),
        "over_2deg_thigh_right_frame_rate": float((anti["delta_thigh_right_deg"][valid].abs() > 2).float().mean()),
        "ratio_t_median": float(anti["ratio_median"]),
        "ratio_t_p05": float(anti["ratio_p05"]),
        "ratio_t_p95": float(anti["ratio_p95"]),
        "ratio_action": float(anti["ratio_action"]),
        "ratio_low_signal_frame_rate": float(anti["low_signal_frame_rate"]),
        "anti_cheat_pass": bool(p95_gate and local_gate),
        "anti_cheat_p95_pass": bool(p95_gate),
    }
    row.update({key: value for key, value in consistency.items() if key != "consistency"})
    row["consistency"] = consistency["consistency"]
    if terminal_record is not None:
        row.update({"terminal_triggered": bool(terminal_record.get("triggered", False)), "terminal_correction_deg": float(terminal_record.get("applied_deg", 0.0)), "terminal_skipped": bool(terminal_record.get("terminal_skipped", False))})
    return row


def summarize_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("at least one metric row is required")
    def median(name: str) -> float:
        return float(torch.tensor([float(row[name]) for row in rows]).median())
    def maximum(name: str) -> float:
        return float(torch.tensor([float(row[name]) for row in rows]).max())
    return {
        "unit_count": len(rows),
        "absolute_mean_error_median_deg": median("absolute_mean_error_deg"),
        "absolute_mean_error_le_2deg_rate": sum(float(row["absolute_mean_error_deg"]) <= 2 for row in rows) / len(rows),
        "centered_curve_correlation_median": median("centered_curve_correlation"),
        "centered_curve_rmse_median_deg": median("centered_curve_rmse_deg"),
        "fluctuation_std_ratio_median": median("fluctuation_std_ratio"),
        "trunk_delta_p95_max_deg": maximum("trunk_delta_p95_deg"),
        "thigh_left_delta_p95_max_deg": maximum("thigh_left_delta_p95_deg"),
        "thigh_right_delta_p95_max_deg": maximum("thigh_right_delta_p95_deg"),
        "local_change_share_median": median("local_change_share"),
        "ratio_t_median": median("ratio_t_median"),
        "ratio_t_p05_median": median("ratio_t_p05"),
        "ratio_t_p95_median": median("ratio_t_p95"),
        "ratio_action_median": median("ratio_action"),
        "ratio_low_signal_frame_rate_median": median("ratio_low_signal_frame_rate"),
        "anti_cheat_pass_rate": sum(bool(row.get("anti_cheat_pass", False)) for row in rows) / len(rows),
        "consistency": {name: maximum(name) for name in (
            "joint_fk_residual_max_m", "joint_velocity_residual_max_m",
            "root_rotation_velocity_residual_max_deg", "root_translation_velocity_residual_max_m",
        )},
    }


def control_success_gate(summary_by_target: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    for target, summary in summary_by_target.items():
        if float(summary["absolute_mean_error_median_deg"]) > 2:
            failures.append(f"{target}: median absolute mean error > 2 deg")
        if float(summary["centered_curve_correlation_median"]) < 0.9:
            failures.append(f"{target}: centered-curve correlation < 0.90")
        for key in ("trunk_delta_p95_max_deg", "thigh_left_delta_p95_max_deg", "thigh_right_delta_p95_max_deg"):
            if float(summary[key]) > 3:
                failures.append(f"{target}: {key} > 3 deg")
        if float(summary["anti_cheat_pass_rate"]) < 0.9:
            failures.append(f"{target}: anti-cheat pass rate < 90%")
    return {"passed": not failures, "failures": failures}


evaluate_absolute_mean_pelvis = evaluate_single
evaluate_one = evaluate_single


__all__ = ["DEFAULT_ANTI_CHEAT_LIMITS", "control_success_gate", "evaluate_absolute_mean_pelvis", "evaluate_one", "evaluate_single", "summarize_rows", "v4_PROTOCOL"]
