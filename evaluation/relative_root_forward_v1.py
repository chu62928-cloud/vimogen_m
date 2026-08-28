"""Evaluation metrics for ``vimogen_relative_root_forward_v1_pose_authoritative``."""

from __future__ import annotations

import math
from typing import Any

import torch

from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe
from motion_rep.pose_authority import (
    PROTOCOL_NAME,
    _geodesic,
    _prefix_source,
    _validate_mask,
    _root_forward,
    consistency_report,
    forward_vector_loss,
    whole_body_audit,
)


def _metric(values: torch.Tensor) -> dict[str, float | None]:
    values = values[torch.isfinite(values)]
    if not values.numel():
        return {"mean": None, "median": None, "p95": None}
    return {"mean": float(values.mean()), "median": float(values.median()), "p95": float(torch.quantile(values, .95))}


def _corr(a: torch.Tensor, b: torch.Tensor, eps: float) -> float | None:
    a = a - a.mean(); b = b - b.mean()
    da = torch.linalg.vector_norm(a); db = torch.linalg.vector_norm(b)
    if da < eps or db < eps:
        return None
    return float((a * b).sum().item() / (da * db).item())


def tail_safety_metrics(baseline: torch.Tensor, candidate: torch.Tensor, valid_mask: torch.Tensor, *, tail_pairs: int = 8) -> dict[str, Any]:
    """Measure excess, rather than absolute, last-frame rotation changes."""

    if baseline.shape != candidate.shape or baseline.ndim != 3:
        raise ValueError("baseline and candidate must be [B,T,276] with equal shapes")
    _validate_mask(valid_mask, tuple(baseline.shape[:2]))
    baseline_source, _ = _prefix_source(baseline, valid_mask)
    candidate_source, _ = _prefix_source(candidate, valid_mask)
    b_root = decode_rot6d_safe(baseline_source[..., MOTION_LAYOUT.root_rotation])
    c_root = decode_rot6d_safe(candidate_source[..., MOTION_LAYOUT.root_rotation])
    b_step = _geodesic(b_root[:, 1:], b_root[:, :-1]) * 180.0 / math.pi
    c_step = _geodesic(c_root[:, 1:], c_root[:, :-1]) * 180.0 / math.pi
    b_phi = _root_forward(b_root)[3]
    c_phi = _root_forward(c_root)[3]
    b_pitch = b_phi[:, 1:] - b_phi[:, :-1]
    c_pitch = c_phi[:, 1:] - c_phi[:, :-1]
    rows = []
    for index in range(baseline.shape[0]):
        pairs = valid_mask[index, 1:] & valid_mask[index, :-1]
        positions = torch.nonzero(pairs, as_tuple=False).flatten()
        positions = positions[-tail_pairs:]
        if positions.numel():
            so3 = (c_step[index, positions] - b_step[index, positions]).abs()
            pitch = (c_pitch[index, positions] - b_pitch[index, positions]).abs()
            all_so3 = (c_step[index] - b_step[index]).abs()[pairs]
            all_pitch = (c_pitch[index] - b_pitch[index]).abs()[pairs]
            rows.append({
                "tail_pair_count": int(positions.numel()),
                "tail_extra_so3_jump_max_deg": float(so3.max()),
                "tail_extra_pitch_step_max_deg": float(pitch.max()),
                "tail_pass": bool(so3.max() <= 2.0 and pitch.max() <= 2.0),
                "full_sequence_extra_so3_p95_deg": float(torch.quantile(all_so3, .95)),
                "full_sequence_extra_pitch_p95_deg": float(torch.quantile(all_pitch, .95)),
            })
        else:
            rows.append({"tail_pair_count": 0, "tail_extra_so3_jump_max_deg": None, "tail_extra_pitch_step_max_deg": None, "tail_pass": False, "full_sequence_extra_so3_p95_deg": None, "full_sequence_extra_pitch_p95_deg": None})
    return {"per_sample": rows}


def compute_relative_root_forward_metrics(
    baseline_physical: torch.Tensor,
    candidate_physical: torch.Tensor,
    valid_mask: torch.Tensor,
    target_delta_deg: float,
    *,
    skeleton=None,
    protocol_name: str = PROTOCOL_NAME,
) -> dict[str, Any]:
    """Return control, curve-shape, tail, and whole-body audit metrics."""

    if baseline_physical.shape != candidate_physical.shape or baseline_physical.ndim != 3:
        raise ValueError("baseline_physical and candidate_physical must both be [B,T,276]")
    if valid_mask.shape != baseline_physical.shape[:2] or valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be bool[B,T]")
    _validate_mask(valid_mask, tuple(baseline_physical.shape[:2]))
    if not math.isfinite(float(target_delta_deg)) or not -10.0 <= float(target_delta_deg) <= 10.0:
        raise ValueError("target_delta_deg must lie in [-10,10]")
    baseline_source, _ = _prefix_source(baseline_physical, valid_mask)
    candidate_source, _ = _prefix_source(candidate_physical, valid_mask)
    b_root = decode_rot6d_safe(baseline_source[..., MOTION_LAYOUT.root_rotation])
    c_root = decode_rot6d_safe(candidate_source[..., MOTION_LAYOUT.root_rotation])
    f0, h0, r0, phi0 = _root_forward(b_root)
    fg, hg, _, phig = _root_forward(c_root)
    target_axis = r0 * (-float(target_delta_deg) * math.pi / 180.0)
    from motion_rep.rotation_transform import axis_angle_to_mat3x3
    target_root = axis_angle_to_mat3x3(target_axis) @ b_root
    target_forward = target_root @ torch.tensor([0., 0., 1.], dtype=b_root.dtype, device=b_root.device)
    mask = valid_mask
    angle_error = (phi0 - phig - float(target_delta_deg)).abs()
    vector_error = torch.atan2(torch.linalg.vector_norm(torch.cross(fg, target_forward, dim=-1), dim=-1), (fg * target_forward).sum(-1).clamp(-1, 1)) * 180.0 / math.pi
    heading_drift = torch.acos((hg * h0).sum(-1).clamp(-1, 1)) * 180.0 / math.pi
    root_target_error = _geodesic(c_root, target_root) * 180.0 / math.pi
    samples = []
    for index in range(baseline_physical.shape[0]):
        m = mask[index]
        std0 = phi0[index][m].std(unbiased=False)
        corr = _corr(phig[index][m], (phi0[index] - float(target_delta_deg))[m], 1e-4)
        q_sigma = None if std0 < 1e-4 else float(phig[index][m].std(unbiased=False) / std0)
        samples.append({
            "mean_absolute_error_deg": float(angle_error[index][m].mean()),
            "median_absolute_error_deg": float(angle_error[index][m].median()),
            "p95_absolute_error_deg": float(torch.quantile(angle_error[index][m], .95)),
            "forward_vector_error_mean_deg": float(vector_error[index][m].mean()),
            "forward_vector_error_p95_deg": float(torch.quantile(vector_error[index][m], .95)),
            "horizontal_heading_drift_mean_deg": float(heading_drift[index][m].mean()),
            "horizontal_heading_drift_p95_deg": float(torch.quantile(heading_drift[index][m], .95)),
            "root_rotation_target_error_mean_deg": float(root_target_error[index][m].mean()),
            "curve_correlation": corr,
            "curve_shape_valid": corr is not None,
            "curve_std_ratio": q_sigma,
            "dose_sign_correct": bool(torch.sign((phi0[index][m] - phig[index][m]).mean()) == torch.sign(torch.as_tensor(target_delta_deg))),
        })
    result = {
        "protocol": protocol_name,
        "target_delta_deg": float(target_delta_deg),
        "per_sample": samples,
        "tail_safety": tail_safety_metrics(baseline_physical, candidate_physical, valid_mask),
        "whole_body": whole_body_audit(baseline_physical, candidate_physical, valid_mask),
        "consistency": {"baseline": consistency_report(baseline_physical, valid_mask, skeleton=skeleton), "candidate": consistency_report(candidate_physical, valid_mask, skeleton=skeleton)},
    }
    return result


__all__ = ["compute_relative_root_forward_metrics", "tail_safety_metrics"]


def dose_monotonicity(
    baseline_physical: torch.Tensor,
    candidate_5deg: torch.Tensor,
    candidate_10deg: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, Any]:
    """Check that the achieved signed root change grows from 5° to 10°."""

    source, _ = _prefix_source(baseline_physical, valid_mask)
    c5_source, _ = _prefix_source(candidate_5deg, valid_mask)
    c10_source, _ = _prefix_source(candidate_10deg, valid_mask)
    phi0 = _root_forward(decode_rot6d_safe(source[..., MOTION_LAYOUT.root_rotation]))[3]
    phi5 = _root_forward(decode_rot6d_safe(c5_source[..., MOTION_LAYOUT.root_rotation]))[3]
    phi10 = _root_forward(decode_rot6d_safe(c10_source[..., MOTION_LAYOUT.root_rotation]))[3]
    achieved5 = (phi0 - phi5)[valid_mask].mean()
    achieved10 = (phi0 - phi10)[valid_mask].mean()
    return {
        "achieved_5deg_mean": float(achieved5),
        "achieved_10deg_mean": float(achieved10),
        "signed_direction_consistent": bool(torch.sign(achieved5) == torch.sign(achieved10)),
        "absolute_monotonic": bool(achieved10.abs() + 1e-6 >= achieved5.abs()),
    }


__all__.append("dose_monotonicity")
