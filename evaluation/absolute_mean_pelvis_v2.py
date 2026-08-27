"""Evaluation and gates for the root-rotation sagittal pelvis-angle v2.

This module is deliberately independent of ``absolute_mean_pelvis.py``.  The
v1 evaluator remains the historical evaluator and keeps its old angle
authority.  v2 consumes physical-space, already-consistent G0/G1 motion and
uses :func:`motion_rep.sagittal_pelvis_angle.pelvis_sagittal_tilt_degrees` as
the only angle authority.

In addition to the v1 control metrics, each row audits all observable
redundant channels:

* direct ``J`` versus differentiable FK;
* stored ``dJ`` versus direct forward differences;
* stored ``dR`` versus direct adjacent SO(3) increments; and
* stored ``dT`` versus direct forward differences.

The final velocity row points to a hidden T+1 pose in the packed protocol, so
velocity residuals are intentionally evaluated on the valid adjacent pairs
only.  A boolean mask must be a contiguous valid prefix; this prevents
padding or a gap from being mistaken for a physical discontinuity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import torch

from motion_rep.consistency_v2 import differentiable_forward_kinematics
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe
from motion_rep.sagittal_pelvis_angle import pelvis_sagittal_tilt_degrees


V2_PROTOCOL = "vimogen_absolute_mean_pelvis_v2_full_fk_sagittal"
DEFAULT_CONSISTENCY_TOLERANCES = {
    "joint_fk_residual_max_m": 1e-5,
    "joint_velocity_residual_max_m": 1e-6,
    "root_rotation_velocity_residual_max_deg": 1e-4,
    "root_translation_velocity_residual_max_m": 1e-6,
}


def _validate_motion(motion: torch.Tensor, name: str) -> None:
    if not isinstance(motion, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if motion.ndim != 2 or motion.shape[-1] != MOTION_LAYOUT.total_dim:
        raise ValueError(f"{name} must have shape [T,276], got {tuple(motion.shape)}")
    if not torch.is_floating_point(motion):
        raise TypeError(f"{name} must be floating point")
    if motion.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one frame")
    if not torch.isfinite(motion).all():
        raise ValueError(f"{name} contains non-finite values")


def _validate_valid_prefix(valid_mask: torch.Tensor | None, frame_count: int, device: torch.device) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones(frame_count, dtype=torch.bool, device=device)
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != (frame_count,) or valid_mask.dtype is not torch.bool:
        raise ValueError(f"valid_mask must be bool[{frame_count}]")
    if valid_mask.device != device:
        raise ValueError("valid_mask must be on the same device as motion")
    if not bool(valid_mask.any()):
        raise ValueError("valid_mask must contain at least one valid frame")
    valid_count = int(valid_mask.sum().detach().cpu())
    expected = torch.zeros_like(valid_mask)
    expected[:valid_count] = True
    if not torch.equal(valid_mask, expected):
        raise ValueError("valid_mask must be a contiguous valid prefix")
    return valid_mask


def _geodesic_degrees(rotation: torch.Tensor) -> torch.Tensor:
    """Return the unsigned SO(3) geodesic angle for each matrix."""

    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    skew = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    return torch.atan2(sine, cosine) * (180.0 / math.pi)


def _max_or_zero(values: torch.Tensor) -> float:
    return float(values.max().detach().cpu()) if values.numel() else 0.0


def _rms_or_zero(values: torch.Tensor) -> float:
    return float(torch.sqrt(values.square().mean()).detach().cpu()) if values.numel() else 0.0


def _fk_joints(
    motion_phys: torch.Tensor,
    *,
    skeleton: Any = None,
    rest_offsets: torch.Tensor | None = None,
    parents: tuple[int, ...] | list[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode one physical packed motion and run the differentiable FK."""

    body = decode_rot6d_safe(
        motion_phys[:, MOTION_LAYOUT.body_pose].reshape(-1, 21, 6)
    ).float()
    root = decode_rot6d_safe(motion_phys[:, MOTION_LAYOUT.root_rotation]).float()
    translation = motion_phys[:, MOTION_LAYOUT.root_translation].float()
    if skeleton is not None:
        if rest_offsets is not None or parents is not None:
            raise ValueError("provide skeleton or rest_offsets/parents, not both")
        fk = differentiable_forward_kinematics(
            body, root, translation, skeleton=skeleton
        )
    else:
        fk = differentiable_forward_kinematics(
            body,
            root,
            translation,
            rest_offsets=rest_offsets,
            parents=parents,
        )
    return fk.joints


def audit_motion_consistency(
    motion_phys: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    skeleton: Any = None,
    rest_offsets: torch.Tensor | None = None,
    parents: tuple[int, ...] | list[int] | torch.Tensor | None = None,
) -> dict[str, float | int | dict[str, float | int]]:
    """Audit all observable direct/velocity/FK channels of one motion.

    Residual magnitudes are Euclidean metres for ``J``, ``dJ`` and ``dT``;
    root-rotation residuals are unsigned SO(3) degrees.  The returned flat
    names are stable for tabular reports, while ``consistency`` provides the
    same values under explicit ``J_fk``, ``dJ``, ``dR`` and ``dT`` labels.
    """

    _validate_motion(motion_phys, "motion_phys")
    mask = _validate_valid_prefix(valid_mask, motion_phys.shape[0], motion_phys.device)
    physical = motion_phys.float()
    direct_joints = physical[:, MOTION_LAYOUT.joints].reshape(-1, 22, 3)
    fk_joints = _fk_joints(
        physical,
        skeleton=skeleton,
        rest_offsets=rest_offsets,
        parents=parents,
    )
    joint_frame_error = torch.linalg.vector_norm(direct_joints - fk_joints, dim=-1)
    joint_frame_error = joint_frame_error[mask]

    pair_mask = mask[:-1] & mask[1:]
    direct_dj = direct_joints[1:] - direct_joints[:-1]
    stored_dj = physical[:-1, MOTION_LAYOUT.joints_velocity].reshape(-1, 22, 3)
    dj_error = torch.linalg.vector_norm(direct_dj - stored_dj, dim=-1)[pair_mask]

    direct_root = decode_rot6d_safe(physical[:, MOTION_LAYOUT.root_rotation])
    stored_dr = decode_rot6d_safe(
        physical[:-1, MOTION_LAYOUT.root_rotation_velocity]
    )
    expected_dr = direct_root[1:] @ direct_root[:-1].transpose(-1, -2)
    dr_error = _geodesic_degrees(expected_dr @ stored_dr.transpose(-1, -2))[pair_mask]

    direct_dt = physical[1:, MOTION_LAYOUT.root_translation] - physical[:-1, MOTION_LAYOUT.root_translation]
    stored_dt = physical[:-1, MOTION_LAYOUT.root_translation_velocity]
    dt_error = torch.linalg.vector_norm(direct_dt - stored_dt, dim=-1)[pair_mask]

    values: dict[str, float | int] = {
        "joint_fk_residual_max_m": _max_or_zero(joint_frame_error),
        "joint_fk_residual_rms_m": _rms_or_zero(joint_frame_error),
        "joint_velocity_residual_max_m": _max_or_zero(dj_error),
        "joint_velocity_residual_rms_m": _rms_or_zero(dj_error),
        "root_rotation_velocity_residual_max_deg": _max_or_zero(dr_error),
        "root_rotation_velocity_residual_rms_deg": _rms_or_zero(dr_error),
        "root_translation_velocity_residual_max_m": _max_or_zero(dt_error),
        "root_translation_velocity_residual_rms_m": _rms_or_zero(dt_error),
        "joint_fk_residual_count": int(joint_frame_error.numel()),
        "joint_velocity_residual_count": int(dj_error.numel()),
        "root_rotation_velocity_residual_count": int(dr_error.numel()),
        "root_translation_velocity_residual_count": int(dt_error.numel()),
    }
    # Descriptive aliases make the audit self-documenting in CSV/JSON exports.
    values.update(
        {
            "J_fk_residual_max_m": values["joint_fk_residual_max_m"],
            "dJ_forward_difference_residual_max_m": values["joint_velocity_residual_max_m"],
            "dR_so3_residual_max_deg": values["root_rotation_velocity_residual_max_deg"],
            "dT_forward_difference_residual_max_m": values["root_translation_velocity_residual_max_m"],
        }
    )
    values["consistency"] = {
        "J_fk_max_m": float(values["joint_fk_residual_max_m"]),
        "J_fk_rms_m": float(values["joint_fk_residual_rms_m"]),
        "dJ_max_m": float(values["joint_velocity_residual_max_m"]),
        "dJ_rms_m": float(values["joint_velocity_residual_rms_m"]),
        "dR_max_deg": float(values["root_rotation_velocity_residual_max_deg"]),
        "dR_rms_deg": float(values["root_rotation_velocity_residual_rms_deg"]),
        "dT_max_m": float(values["root_translation_velocity_residual_max_m"]),
        "dT_rms_m": float(values["root_translation_velocity_residual_rms_m"]),
        "valid_frames": int(mask.sum().detach().cpu()),
        "valid_pairs": int(pair_mask.sum().detach().cpu()),
    }
    return values


def _correlation(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-8) -> float:
    if left.numel() != right.numel() or left.numel() == 0:
        return 0.0
    if left.numel() < 2:
        return 1.0 if torch.allclose(left, right, atol=eps, rtol=0) else 0.0
    left = left - left.mean()
    right = right - right.mean()
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denom.detach().cpu()) <= eps:
        return 1.0 if torch.allclose(left, right, atol=eps, rtol=0) else 0.0
    return float(((left * right).sum() / denom).detach().cpu())


def evaluate_single(
    *,
    sample_id: str,
    method: str,
    seed: int,
    target_mean_deg: float,
    baseline_phys: torch.Tensor,
    candidate_phys: torch.Tensor,
    valid_mask: torch.Tensor,
    terminal_record: dict[str, Any] | None = None,
    skeleton: Any = None,
    rest_offsets: torch.Tensor | None = None,
    parents: tuple[int, ...] | list[int] | torch.Tensor | None = None,
) -> dict[str, Any]:
    """Evaluate one physical-space G0/G1 pair under the v2 protocol."""

    _validate_motion(baseline_phys, "baseline_phys")
    _validate_motion(candidate_phys, "candidate_phys")
    if candidate_phys.shape != baseline_phys.shape:
        raise ValueError("baseline_phys and candidate_phys must have the same shape")
    if baseline_phys.device != candidate_phys.device:
        raise ValueError("baseline_phys and candidate_phys must be on the same device")
    mask = _validate_valid_prefix(valid_mask, candidate_phys.shape[0], candidate_phys.device)

    baseline_root = decode_rot6d_safe(
        baseline_phys[:, MOTION_LAYOUT.root_rotation].float()
    )
    candidate_root = decode_rot6d_safe(
        candidate_phys[:, MOTION_LAYOUT.root_rotation].float()
    )
    baseline_curve = pelvis_sagittal_tilt_degrees(baseline_root)[mask]
    candidate_curve = pelvis_sagittal_tilt_degrees(candidate_root)[mask]
    baseline_centered = baseline_curve - baseline_curve.mean()
    candidate_centered = candidate_curve - candidate_curve.mean()
    baseline_std = baseline_centered.std(unbiased=False)
    candidate_std = candidate_centered.std(unbiased=False)
    consistency = audit_motion_consistency(
        candidate_phys,
        mask,
        skeleton=skeleton,
        rest_offsets=rest_offsets,
        parents=parents,
    )
    target_mean_deg = float(target_mean_deg)
    row: dict[str, Any] = {
        "protocol": V2_PROTOCOL,
        "angle_authority": "root_rotation_local_sagittal_tilt_v2",
        "sample_id": sample_id,
        "method": method,
        "seed": int(seed),
        "target_mean_deg": target_mean_deg,
        "achieved_mean_deg": float(candidate_curve.mean().detach().cpu()),
        "absolute_mean_error_deg": float(
            abs(candidate_curve.mean().detach().cpu().item() - target_mean_deg)
        ),
        "centered_curve_rmse_deg": float(
            torch.sqrt((candidate_centered - baseline_centered).square().mean()).detach().cpu()
        ),
        "centered_curve_correlation": _correlation(baseline_centered, candidate_centered),
        "fluctuation_std_ratio": float(
            (candidate_std / baseline_std.clamp_min(1e-8)).detach().cpu()
        ),
        "motion_rms_from_m0": float(
            torch.sqrt(
                (candidate_phys[mask].float() - baseline_phys[mask].float()).square().mean()
            ).detach().cpu()
        ),
        "valid_frames": int(mask.sum().detach().cpu()),
    }
    row.update({key: value for key, value in consistency.items() if key != "consistency"})
    row["consistency"] = consistency["consistency"]
    if terminal_record is not None:
        row.update(
            {
                "terminal_triggered": bool(terminal_record.get("triggered", False)),
                "terminal_failed_residual_over_limit": bool(
                    terminal_record.get("failed_residual_over_limit", False)
                ),
                "terminal_correction_deg": float(terminal_record.get("applied_deg", 0.0)),
            }
        )
    return row


def summarize_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate v1-compatible control metrics and v2 consistency maxima."""

    rows = list(rows)
    if not rows:
        raise ValueError("at least one metric row is required")
    errors = torch.tensor([float(row["absolute_mean_error_deg"]) for row in rows])
    correlations = torch.tensor([float(row["centered_curve_correlation"]) for row in rows])
    std_ratios = torch.tensor([float(row["fluctuation_std_ratio"]) for row in rows])
    curve_rmse = torch.tensor([float(row["centered_curve_rmse_deg"]) for row in rows])
    summary: dict[str, Any] = {
        "unit_count": len(rows),
        "absolute_mean_error_median_deg": float(errors.median()),
        "absolute_mean_error_le_2deg_rate": float((errors <= 2.0).float().mean()),
        "centered_curve_correlation_median": float(correlations.median()),
        "centered_curve_rmse_median_deg": float(curve_rmse.median()),
        "fluctuation_std_ratio_median": float(std_ratios.median()),
        "terminal_trigger_rate": float(
            sum(bool(row.get("terminal_triggered", False)) for row in rows) / len(rows)
        ),
    }
    for key in DEFAULT_CONSISTENCY_TOLERANCES:
        values = torch.tensor([float(row[key]) for row in rows])
        summary[key] = float(values.max())
        summary[key.replace("_max_", "_median_")] = float(values.median())
    summary.update(
        {
            "J_fk_residual_max_m": summary["joint_fk_residual_max_m"],
            "dJ_forward_difference_residual_max_m": summary["joint_velocity_residual_max_m"],
            "dR_so3_residual_max_deg": summary["root_rotation_velocity_residual_max_deg"],
            "dT_forward_difference_residual_max_m": summary["root_translation_velocity_residual_max_m"],
        }
    )
    return summary


def control_success_gate(
    summary_by_target: Mapping[str, Mapping[str, Any]],
    *,
    consistency_tolerances: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Apply the v1 control gate plus the complete v2 consistency gate."""

    tolerances = dict(DEFAULT_CONSISTENCY_TOLERANCES)
    if consistency_tolerances is not None:
        tolerances.update({str(key): float(value) for key, value in consistency_tolerances.items()})
    failures: list[str] = []
    for target, summary in summary_by_target.items():
        if summary["absolute_mean_error_median_deg"] > 2.0:
            failures.append(f"{target}: median absolute mean error > 2 deg")
        if summary["absolute_mean_error_le_2deg_rate"] < 0.90:
            failures.append(f"{target}: fewer than 90% units have <= 2 deg error")
        if summary["centered_curve_correlation_median"] < 0.90:
            failures.append(f"{target}: median centered-curve correlation < 0.90")
        ratio = summary["fluctuation_std_ratio_median"]
        if ratio < 0.8 or ratio > 1.2:
            failures.append(f"{target}: median fluctuation std ratio outside [0.8,1.2]")
        for key, tolerance in tolerances.items():
            if key not in summary:
                failures.append(f"{target}: missing consistency metric {key}")
            elif float(summary[key]) > tolerance:
                failures.append(
                    f"{target}: {key} > {tolerance:g}"
                )
    return {"passed": not failures, "failures": failures}


def g1_promotion_gate(
    *,
    g0_summary: Mapping[str, Any],
    g1_summary: Mapping[str, Any],
    naturalness_no_degradation: bool,
) -> dict[str, Any]:
    """Retain the v1 G1 promotion rules for v2 summaries."""

    failures: list[str] = []
    if g1_summary["absolute_mean_error_median_deg"] >= g0_summary["absolute_mean_error_median_deg"]:
        failures.append("G1 did not improve median absolute mean error")
    if g1_summary.get("terminal_trigger_rate", 0.0) > 0.05:
        failures.append("G1 terminal trigger rate exceeds 5%")
    if g1_summary.get("centered_curve_rmse_median_deg", 0.0) > g0_summary.get(
        "centered_curve_rmse_median_deg", 0.0
    ) + 0.1:
        failures.append("G1 increased median centered-curve RMSE by more than 0.1 deg")
    if not naturalness_no_degradation:
        failures.append("naturalness non-degradation gate failed")
    return {"promote_g1": not failures, "failures": failures}


# Names used in reports and small downstream adapters.
evaluate_absolute_mean_pelvis = evaluate_single
evaluate_one = evaluate_single
consistency_audit = audit_motion_consistency
evaluate_motion_consistency = audit_motion_consistency
compute_consistency_metrics = audit_motion_consistency


__all__ = [
    "DEFAULT_CONSISTENCY_TOLERANCES",
    "V2_PROTOCOL",
    "audit_motion_consistency",
    "consistency_audit",
    "compute_consistency_metrics",
    "control_success_gate",
    "evaluate_absolute_mean_pelvis",
    "evaluate_one",
    "evaluate_motion_consistency",
    "evaluate_single",
    "g1_promotion_gate",
    "summarize_rows",
]
