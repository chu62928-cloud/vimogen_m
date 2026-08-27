"""Metrics and decision gates for absolute-mean pelvis guidance v1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch

from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe
from motion_rep.pelvis_angle import pelvis_pitch_curve_v2
from motion_rep.consistency_v2 import reconcile_motion_tensor_v2
from motion_rep.reconciliation import ReconciliationConfig, reconcile_motion_tensor
from sampling.absolute_mean_pelvis_guidance import pelvis_angle_curve


def _geodesic_degrees(rotation: torch.Tensor) -> torch.Tensor:
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


def root_rotation_velocity_residual_degrees(
    motion_phys: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Return adjacent direct-root versus stored-velocity residuals.

    The final velocity row points to the hidden T+1 pose and has no second
    direct root in the packed tensor, so this public residual contains T-1
    auditable pairs.
    """

    if motion_phys.ndim != 2 or motion_phys.shape[-1] != 276:
        raise ValueError("motion_phys must have shape [T,276]")
    if valid_mask.shape != motion_phys.shape[:1] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool[T]")
    if motion_phys.shape[0] < 2:
        return torch.empty(0, dtype=motion_phys.dtype, device=motion_phys.device)
    root = decode_rot6d_safe(motion_phys[:, MOTION_LAYOUT.root_rotation])
    velocity = decode_rot6d_safe(
        motion_phys[:-1, MOTION_LAYOUT.root_rotation_velocity]
    )
    predicted = velocity @ root[:-1]
    residual = root[1:] @ predicted.transpose(-1, -2)
    pair_mask = valid_mask[:-1] & valid_mask[1:]
    return _geodesic_degrees(residual)[pair_mask]


def _masked_curve(motion_phys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return pelvis_angle_curve(motion_phys)[mask]


def _correlation(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-8) -> float:
    if left.numel() < 2 or right.numel() != left.numel():
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denom) <= eps:
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
    terminal_record: dict | None = None,
    angle_version: str = "legacy_v1",
    skeleton=None,
) -> dict:
    if angle_version in {"fk_v2", "v2", "vimogen_276d_consistency_v2"}:
        baseline_curve = pelvis_pitch_curve_v2(baseline_phys, skeleton=skeleton)[valid_mask]
        candidate_curve = pelvis_pitch_curve_v2(candidate_phys, skeleton=skeleton)[valid_mask]
    elif angle_version in {"legacy_v1", "v1"}:
        baseline_curve = _masked_curve(baseline_phys, valid_mask)
        candidate_curve = _masked_curve(candidate_phys, valid_mask)
    else:
        raise ValueError(f"unknown angle_version: {angle_version!r}")
    baseline_centered = baseline_curve - baseline_curve.mean()
    candidate_centered = candidate_curve - candidate_curve.mean()
    baseline_std = baseline_centered.std(unbiased=False)
    candidate_std = candidate_centered.std(unbiased=False)
    std_ratio = float(
        (candidate_std / baseline_std.clamp_min(1e-8)).detach().cpu()
    )
    residual = root_rotation_velocity_residual_degrees(candidate_phys, valid_mask)
    row = {
        "sample_id": sample_id,
        "method": method,
        "seed": int(seed),
        "target_mean_deg": float(target_mean_deg),
        "achieved_mean_deg": float(candidate_curve.mean().detach().cpu()),
        "absolute_mean_error_deg": float(
            abs(candidate_curve.mean().detach().cpu().item() - target_mean_deg)
        ),
        "centered_curve_rmse_deg": float(
            torch.sqrt((candidate_centered - baseline_centered).square().mean()).detach().cpu()
        ),
        "centered_curve_correlation": _correlation(
            baseline_centered, candidate_centered
        ),
        "fluctuation_std_ratio": std_ratio,
        "motion_rms_from_m0": float(
            torch.sqrt((candidate_phys[valid_mask] - baseline_phys[valid_mask]).square().mean()).detach().cpu()
        ),
        "root_rotation_velocity_residual_max_deg": (
            float(residual.max().detach().cpu()) if residual.numel() else 0.0
        ),
        "valid_frames": int(valid_mask.sum().detach().cpu()),
    }
    if terminal_record is not None:
        row.update(
            {
                "terminal_triggered": bool(terminal_record.get("triggered", False)),
                "terminal_failed_residual_over_limit": bool(
                    terminal_record.get("failed_residual_over_limit", False)
                ),
                "terminal_correction_deg": float(
                    terminal_record.get("applied_deg", 0.0)
                ),
            }
        )
    return row


def physical_authority(
    motion_norm: torch.Tensor,
    valid_mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    consistency_mode: str = "legacy_v1",
    skeleton=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if consistency_mode in {"fk_v2", "v2", "vimogen_276d_consistency_v2"}:
        result = reconcile_motion_tensor_v2(
            motion_norm.float(),
            fusion_window=9,
            anchor_weight=1.0,
            root_rotation_anchor_weight=1.0,
            valid_mask=valid_mask,
            mean=mean,
            std=std,
            input_standardized=True,
            output_standardized=False,
            output_dtype=torch.float32,
            skeleton=skeleton,
        )
        return result.motion, result.valid_mask
    result = reconcile_motion_tensor(
        motion_norm.float(),
        config=ReconciliationConfig(
            correction_window=9,
            anchor_weight=1.0,
            root_rotation_anchor_weight=1.0,
        ),
        valid_mask=valid_mask,
        mean=mean,
        std=std,
        input_standardized=True,
        output_standardized=False,
        output_dtype=torch.float32,
    )
    return result.motion, result.valid_mask


def summarize_rows(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    if not rows:
        raise ValueError("at least one metric row is required")
    errors = torch.tensor([row["absolute_mean_error_deg"] for row in rows])
    correlations = torch.tensor(
        [row["centered_curve_correlation"] for row in rows]
    )
    std_ratios = torch.tensor([row["fluctuation_std_ratio"] for row in rows])
    curve_rmse = torch.tensor([row["centered_curve_rmse_deg"] for row in rows])
    residuals = torch.tensor(
        [row["root_rotation_velocity_residual_max_deg"] for row in rows]
    )
    triggered = [bool(row.get("terminal_triggered", False)) for row in rows]
    return {
        "unit_count": len(rows),
        "absolute_mean_error_median_deg": float(errors.median()),
        "absolute_mean_error_le_2deg_rate": float((errors <= 2.0).float().mean()),
        "centered_curve_correlation_median": float(correlations.nanmedian()),
        "centered_curve_rmse_median_deg": float(curve_rmse.median()),
        "fluctuation_std_ratio_median": float(std_ratios.median()),
        "root_rotation_velocity_residual_max_deg": float(residuals.max()),
        "terminal_trigger_rate": float(sum(triggered) / len(triggered)),
    }


def control_success_gate(summary_by_target: dict[str, dict]) -> dict:
    failures: list[str] = []
    for target, summary in summary_by_target.items():
        if summary["absolute_mean_error_median_deg"] > 2.0:
            failures.append(f"{target}: median absolute mean error > 2 deg")
        if summary["absolute_mean_error_le_2deg_rate"] < 0.90:
            failures.append(f"{target}: fewer than 90% units have <= 2 deg error")
        if summary["root_rotation_velocity_residual_max_deg"] > 1e-4:
            failures.append(f"{target}: root rotation/velocity residual > 1e-4 deg")
        if summary["centered_curve_correlation_median"] < 0.90:
            failures.append(f"{target}: median centered-curve correlation < 0.90")
        ratio = summary["fluctuation_std_ratio_median"]
        if ratio < 0.8 or ratio > 1.2:
            failures.append(f"{target}: median fluctuation std ratio outside [0.8,1.2]")
    return {"passed": not failures, "failures": failures}


def g1_promotion_gate(
    *,
    g0_summary: dict,
    g1_summary: dict,
    naturalness_no_degradation: bool,
) -> dict:
    failures: list[str] = []
    if g1_summary["absolute_mean_error_median_deg"] >= g0_summary["absolute_mean_error_median_deg"]:
        failures.append("G1 did not improve median absolute mean error")
    if g1_summary["terminal_trigger_rate"] > 0.05:
        failures.append("G1 terminal trigger rate exceeds 5%")
    # Per-unit curve RMSE is the auditable curve-error definition used here.
    if g1_summary.get("centered_curve_rmse_median_deg", 0.0) > g0_summary.get(
        "centered_curve_rmse_median_deg", 0.0
    ) + 0.1:
        failures.append("G1 increased median centered-curve RMSE by more than 0.1 deg")
    if not naturalness_no_degradation:
        failures.append("naturalness non-degradation gate failed")
    return {"promote_g1": not failures, "failures": failures}


__all__ = [
    "control_success_gate",
    "evaluate_single",
    "g1_promotion_gate",
    "physical_authority",
    "root_rotation_velocity_residual_degrees",
    "summarize_rows",
]
