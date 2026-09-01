"""Geometry, contact evidence and gates for pelvis contact compensation v3.

The module is intentionally independent from the optimiser.  It freezes the
M0-defined sagittal frame and contact evidence once, then exposes the same
definitions to the window solver, the full-sequence solver and the evaluator.
The positive dose follows the v1.3 convention: a positive dose lowers the
root forward axis, therefore ``delta = M0_pitch - candidate_pitch``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import torch

from motion_rep.phase1 import MOTION_LAYOUT, SMPLX_22_JOINT_INDEX, decode_rot6d_safe, encode_rot6d, validate_motion_tensor
from motion_rep.pose_authority import _geodesic, _root_forward, authority_project, consistency_report
from motion_rep.rotation_transform import axis_angle_to_mat3x3
from evaluation.relative_root_forward_v1 import tail_safety_metrics
from evaluation.relative_root_trunk_v2_1 import direct_joints_from_motion


PROTOCOL_NAME = "vimogen_pelvis_contact_compensation_v3"
GEOMETRY_PROTOCOL = "vimogen_pelvis_sagittal_delta_v3_0"
PASS = "PASS"
FAIL = "FAIL"
NOT_EVALUABLE = "NOT_EVALUABLE"
TOE_GAP_THRESHOLD_M = 0.035
CONTACT_HEIGHT_M = 0.025
CONTACT_SPEED_M_PER_FRAME = 0.030
FLAT_GAP_M = 0.020
STABLE_CONFIDENCE = 0.80
MIN_EVIDENCE = 3
EPS = 1.0e-8


def wrap_angle_deg(value: torch.Tensor) -> torch.Tensor:
    return torch.remainder(value + 180.0, 360.0) - 180.0


def _unit(value: torch.Tensor, name: str) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    if torch.any(norm <= EPS):
        raise ValueError(f"{name} is degenerate")
    return value / norm


def _project_to_plane(value: torch.Tensor, normal: torch.Tensor, name: str) -> torch.Tensor:
    projected = value - (value * normal).sum(-1, keepdim=True) * normal
    return _unit(projected, name)


def m0_sagittal_frame(m0_root_rotation: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return the frozen M0 forward, heading, right axis and pitch."""

    if m0_root_rotation.shape[-2:] != (3, 3):
        raise ValueError("m0_root_rotation must end in [3,3]")
    forward, heading, right, pitch = _root_forward(m0_root_rotation)
    return {"forward": forward, "heading": heading, "right": right, "pitch_deg": pitch}


def pelvis_pitch_delta_deg(
    m0_root_rotation: torch.Tensor,
    candidate_root_rotation: torch.Tensor,
) -> torch.Tensor:
    """Measure v3 dose in the frozen M0 sagittal plane.

    The sign is deliberately the v1.3 sign.  With M0 facing horizontally,
    ``Rot(+right, -10°) @ R0`` returns ``+10°``.
    """

    if m0_root_rotation.shape != candidate_root_rotation.shape:
        raise ValueError("M0 and candidate root rotations must have equal shape")
    frame = m0_sagittal_frame(m0_root_rotation)
    candidate_forward = candidate_root_rotation @ torch.tensor(
        [0.0, 0.0, 1.0], dtype=candidate_root_rotation.dtype, device=candidate_root_rotation.device
    )
    m0_plane = _project_to_plane(frame["forward"], frame["right"], "M0 forward")
    candidate_plane = _project_to_plane(candidate_forward, frame["right"], "candidate forward")
    # cross(candidate, M0) around +right is positive for the historical +dose.
    sine = (torch.cross(candidate_plane, m0_plane, dim=-1) * frame["right"]).sum(-1)
    cosine = (candidate_plane * m0_plane).sum(-1).clamp(-1.0, 1.0)
    return wrap_angle_deg(torch.atan2(sine, cosine) * (180.0 / math.pi))


def target_root_rotation(
    m0_root_rotation: torch.Tensor,
    target_delta_deg: float | torch.Tensor,
) -> torch.Tensor:
    """Construct the exact target root rotation while freezing M0 heading."""

    frame = m0_sagittal_frame(m0_root_rotation)
    delta = torch.as_tensor(target_delta_deg, dtype=m0_root_rotation.dtype, device=m0_root_rotation.device)
    delta = torch.broadcast_to(delta, m0_root_rotation.shape[:-2])
    correction = axis_angle_to_mat3x3(frame["right"] * (-delta * math.pi / 180.0))
    return correction @ m0_root_rotation


def _summary(values: torch.Tensor, *, absolute: bool = False) -> dict[str, Any]:
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"mean": None, "median": None, "p95": None, "max": None, "count": 0}
    if absolute:
        values = values.abs()
    return {
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
        "count": int(values.numel()),
    }


def contact_evidence(
    heel: torch.Tensor,
    toe: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    contact_height_m: float = CONTACT_HEIGHT_M,
    contact_speed_m_per_frame: float = CONTACT_SPEED_M_PER_FRAME,
    flat_gap_m: float = FLAT_GAP_M,
) -> dict[str, Any]:
    """Create deterministic discrete masks and a continuous M0 confidence."""

    if heel.shape != toe.shape or heel.ndim != 2 or heel.shape[-1] != 3:
        raise ValueError("heel and toe must both have shape [T,3]")
    if not torch.isfinite(heel).all() or not torch.isfinite(toe).all():
        raise ValueError("heel and toe must be finite")
    frames = heel.shape[0]
    valid = torch.ones(frames, dtype=torch.bool, device=heel.device) if valid_mask is None else valid_mask.to(device=heel.device)
    if valid.shape != (frames,) or valid.dtype is not torch.bool:
        raise ValueError("valid_mask must have shape [T] and bool dtype")
    if torch.any(valid[1:] & ~valid[:-1]):
        raise ValueError("valid_mask must be a contiguous prefix")
    if not bool(valid.any()):
        raise ValueError("at least one valid frame is required")
    sole = torch.minimum(heel[:, 2], toe[:, 2])
    centre = 0.5 * (heel + toe)
    speed = torch.full((frames,), float("nan"), dtype=heel.dtype, device=heel.device)
    if frames > 1:
        speed[1:] = torch.linalg.vector_norm(centre[1:, :2] - centre[:-1, :2], dim=-1)
    floor = torch.quantile(sole[valid], 0.05)
    height_score = ((floor + contact_height_m - sole) / contact_height_m).clamp(0.0, 1.0)
    speed_score = ((contact_speed_m_per_frame - speed) / contact_speed_m_per_frame).clamp(0.0, 1.0)
    confidence = (height_score * speed_score).masked_fill(~torch.isfinite(speed), 0.0)
    confidence = confidence * valid.to(confidence.dtype)
    general = valid & torch.isfinite(speed) & (sole <= floor + contact_height_m) & (speed <= contact_speed_m_per_frame)
    flat = general & ((heel[:, 2] - toe[:, 2]).abs() <= flat_gap_m)
    pair = general[1:] & general[:-1] if frames > 1 else general[:0]
    flat_pair = flat[1:] & flat[:-1] if frames > 1 else flat[:0]
    return {
        "floor_height_m": float(floor.item()),
        "contact_frames": int(general.sum().item()),
        "flat_contact_frames": int(flat.sum().item()),
        "continuous_contact_pairs": int(pair.sum().item()),
        "continuous_flat_pairs": int(flat_pair.sum().item()),
        "confidence": confidence.detach().cpu().tolist(),
        "valid_masks": {
            "valid": valid.detach().cpu().tolist(),
            "general_contact": general.detach().cpu().tolist(),
            "flat_contact": flat.detach().cpu().tolist(),
            "continuous_contact_pair": pair.detach().cpu().tolist(),
            "continuous_flat_pair": flat_pair.detach().cpu().tolist(),
        },
        "thresholds": {
            "contact_height_m": float(contact_height_m),
            "contact_speed_m_per_frame": float(contact_speed_m_per_frame),
            "flat_gap_m": float(flat_gap_m),
            "stable_confidence": STABLE_CONFIDENCE,
            "first_frame_speed_is_valid": False,
        },
    }


def longest_true_run(mask: torch.Tensor) -> tuple[int, int] | None:
    if mask.ndim != 1 or mask.dtype is not torch.bool:
        raise ValueError("mask must be one-dimensional bool")
    best: tuple[int, int] | None = None
    start: int | None = None
    for index, value in enumerate(mask.detach().cpu().tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            candidate = (start, index)
            if best is None or (candidate[1] - candidate[0], -candidate[0]) > (best[1] - best[0], -best[0]):
                best = candidate
            start = None
    return best


def select_stable_window(
    confidence: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    pad: int = 4,
) -> dict[str, Any]:
    """Select the longest stable M0 interval; ties choose the earliest one."""

    if confidence.ndim != 1:
        raise ValueError("confidence must be [T]")
    valid = torch.ones_like(confidence, dtype=torch.bool) if valid_mask is None else valid_mask.to(confidence.device)
    stable = (confidence >= STABLE_CONFIDENCE) & valid
    run = longest_true_run(stable)
    if run is None:
        return {"status": NOT_EVALUABLE, "stable_count": int(stable.sum().item()), "frames": []}
    start, end = run
    window_start = max(0, start - int(pad))
    window_end = min(confidence.shape[0], end + int(pad))
    return {
        "status": PASS,
        "stable_count": int(stable.sum().item()),
        "stable_start": start,
        "stable_end_exclusive": end,
        "window_start": window_start,
        "window_end_exclusive": window_end,
        "frames": list(range(window_start, window_end)),
    }


def foot_patches(model: Any) -> dict[str, dict[str, list[int]]]:
    """Freeze the v1.3 neutral-template foot patches as JSON-safe indices."""

    template = model.v_template.detach().cpu()
    joints = model.J_regressor.detach().cpu() @ template
    dominant_joint = model.lbs_weights.detach().cpu().argmax(-1)
    result: dict[str, dict[str, list[int]]] = {}
    for side, ankle_index, foot_index in (("left", 7, 10), ("right", 8, 11)):
        foot_vertices = torch.nonzero((dominant_joint == ankle_index) | (dominant_joint == foot_index), as_tuple=False).flatten()
        candidate = template[foot_vertices]
        sole = foot_vertices[candidate[:, 1] <= torch.quantile(candidate[:, 1], 0.25)]
        forward = joints[foot_index] - joints[ankle_index]
        forward[1] = 0.0
        forward = forward / torch.linalg.vector_norm(forward).clamp_min(EPS)
        longitudinal = ((template[sole] - joints[ankle_index]) * forward).sum(-1)
        heel = sole[longitudinal <= torch.quantile(longitudinal, 0.25)]
        toe = sole[longitudinal >= torch.quantile(longitudinal, 0.75)]
        if heel.numel() < 5 or toe.numel() < 5:
            raise RuntimeError(f"insufficient {side} heel/toe vertices")
        result[side] = {
            "heel": [int(x) for x in heel.tolist()],
            "toe": [int(x) for x in toe.tolist()],
            "sole": [int(x) for x in sole.tolist()],
        }
    return result


def patch_hash(patches: Mapping[str, Mapping[str, list[int]]]) -> str:
    payload = json.dumps(patches, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def patch_centres(vertices: torch.Tensor, patch: Mapping[str, list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    if vertices.ndim != 3:
        raise ValueError("vertices must have shape [T,V,3]")
    heel = vertices[:, torch.as_tensor(patch["heel"], dtype=torch.long, device=vertices.device)].mean(1)
    toe = vertices[:, torch.as_tensor(patch["toe"], dtype=torch.long, device=vertices.device)].mean(1)
    return heel, toe


def _allowed_increase(value: float | None) -> float:
    return 0.001 if value is None else max(abs(float(value)) * 0.05, 0.001)


def _paired_status(candidate: dict[str, Any], baseline: dict[str, Any]) -> str:
    if candidate["count"] < MIN_EVIDENCE or baseline["count"] < MIN_EVIDENCE:
        return NOT_EVALUABLE
    for key in ("mean", "p95"):
        if candidate[key] is None or baseline[key] is None:
            return NOT_EVALUABLE
        if float(candidate[key]) > float(baseline[key]) + _allowed_increase(baseline[key]):
            return FAIL
    return PASS


def evaluate_paired_foot(
    m0_heel: torch.Tensor,
    m0_toe: torch.Tensor,
    candidate_heel: torch.Tensor,
    candidate_toe: torch.Tensor,
    *,
    allow_missing_toe: bool = False,
) -> dict[str, Any]:
    evidence = contact_evidence(m0_heel, m0_toe)
    masks = evidence["valid_masks"]
    general = torch.tensor(masks["general_contact"], dtype=torch.bool, device=m0_heel.device)
    flat = torch.tensor(masks["flat_contact"], dtype=torch.bool, device=m0_heel.device)
    pairs = torch.tensor(masks["continuous_contact_pair"], dtype=torch.bool, device=m0_heel.device)
    floor = torch.as_tensor(evidence["floor_height_m"], dtype=m0_heel.dtype, device=m0_heel.device)
    m0_sole = torch.minimum(m0_heel[:, 2], m0_toe[:, 2])
    candidate_sole = torch.minimum(candidate_heel[:, 2], candidate_toe[:, 2])
    m0_centre = 0.5 * (m0_heel + m0_toe)
    candidate_centre = 0.5 * (candidate_heel + candidate_toe)
    m0_speed = torch.linalg.vector_norm(m0_centre[1:, :2] - m0_centre[:-1, :2], dim=-1)
    candidate_speed = torch.linalg.vector_norm(candidate_centre[1:, :2] - candidate_centre[:-1, :2], dim=-1)
    baseline = {
        "sliding_m_per_frame": _summary(m0_speed[pairs]),
        "lift_m": _summary((m0_sole - floor).clamp_min(0.0)[general]),
        "penetration_m": _summary((floor - m0_sole).clamp_min(0.0)[general]),
    }
    candidate = {
        "sliding_m_per_frame": _summary(candidate_speed[pairs]),
        "lift_m": _summary((candidate_sole - floor).clamp_min(0.0)[general]),
        "penetration_m": _summary((floor - candidate_sole).clamp_min(0.0)[general]),
    }
    statuses = {key: _paired_status(candidate[key], baseline[key]) for key in baseline}
    m0_gap = m0_heel[:, 2] - m0_toe[:, 2]
    candidate_gap = candidate_heel[:, 2] - candidate_toe[:, 2]
    if int(flat.sum().item()) < MIN_EVIDENCE:
        toe_status = NOT_EVALUABLE
        if allow_missing_toe:
            toe_status = NOT_EVALUABLE
        m0_fraction = candidate_fraction = None
    else:
        m0_fraction = float((m0_gap[flat] >= TOE_GAP_THRESHOLD_M).float().mean().item())
        candidate_fraction = float((candidate_gap[flat] >= TOE_GAP_THRESHOLD_M).float().mean().item())
        toe_status = PASS if candidate_fraction <= m0_fraction + 0.05 else FAIL
    all_statuses = list(statuses.values()) + [toe_status]
    status = FAIL if FAIL in all_statuses else NOT_EVALUABLE if NOT_EVALUABLE in all_statuses else PASS
    return {
        "contact_evidence": evidence,
        "baseline": baseline,
        "candidate": candidate,
        "statuses": statuses,
        "toe_contact": {
            "status": toe_status,
            "threshold_gap_m": TOE_GAP_THRESHOLD_M,
            "baseline_fraction": m0_fraction,
            "candidate_fraction": candidate_fraction,
            "new_fraction_tolerance": 0.05,
        },
        "status": status,
        "allow_missing_toe": bool(allow_missing_toe),
    }


def _gate(name: str, observed: float | bool | None, threshold: float | bool | str, count: int, *, reason: str = "") -> dict[str, Any]:
    if observed is None or count <= 0:
        status = NOT_EVALUABLE
    elif isinstance(observed, bool):
        status = PASS if observed else FAIL
    elif not math.isfinite(float(observed)):
        status = FAIL
    else:
        status = PASS if float(observed) <= float(threshold) else FAIL
    return {"name": name, "status": status, "threshold": threshold, "observed": observed, "valid_count": int(count), "reason": reason}


def combine_statuses(statuses: list[str], *, allow_not_evaluable: bool = False) -> str:
    if FAIL in statuses:
        return FAIL
    if NOT_EVALUABLE in statuses and not allow_not_evaluable:
        return NOT_EVALUABLE
    return PASS


def evaluate_v3_pair(
    m0_physical: torch.Tensor,
    candidate_physical: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    target_delta_deg: float,
    m0_vertices: torch.Tensor | None = None,
    candidate_vertices: torch.Tensor | None = None,
    patches: Mapping[str, Mapping[str, list[int]]] | None = None,
    allow_missing_toe: bool = False,
) -> dict[str, Any]:
    """Evaluate one M0/candidate pair using the frozen v3 definitions."""

    if m0_physical.shape != candidate_physical.shape or m0_physical.ndim != 3:
        raise ValueError("motions must both be [B,T,276]")
    validate_motion_tensor(m0_physical)
    validate_motion_tensor(candidate_physical)
    m0_auth = authority_project(m0_physical, valid_mask=valid_mask, output_dtype=torch.float32).physical_motion
    candidate_auth = authority_project(candidate_physical, valid_mask=valid_mask, output_dtype=torch.float32).physical_motion
    b_root = decode_rot6d_safe(m0_auth[..., MOTION_LAYOUT.root_rotation])
    c_root = decode_rot6d_safe(candidate_auth[..., MOTION_LAYOUT.root_rotation])
    actual_delta = pelvis_pitch_delta_deg(b_root, c_root)
    target = torch.full_like(actual_delta, float(target_delta_deg))
    error = wrap_angle_deg(actual_delta - target).abs()
    mask = valid_mask.bool()
    values = error[mask]
    angle = {
        "mae_deg": float(values.mean().item()) if values.numel() else None,
        "p95_deg": float(torch.quantile(values, 0.95).item()) if values.numel() else None,
        "dose_mean_deg": float(actual_delta[mask].mean().item()) if values.numel() else None,
        "dose_sign_correct": bool(torch.sign(actual_delta[mask].mean()) == torch.sign(torch.as_tensor(target_delta_deg, device=actual_delta.device))) if values.numel() else False,
        "valid_count": int(values.numel()),
    }
    b_joints = direct_joints_from_motion(m0_auth)
    c_joints = direct_joints_from_motion(candidate_auth)
    spine1 = SMPLX_22_JOINT_INDEX["spine1"]
    neck = SMPLX_22_JOINT_INDEX["neck"]
    b_trunk = b_joints[..., neck, :] - b_joints[..., spine1, :]
    c_trunk = c_joints[..., neck, :] - c_joints[..., spine1, :]
    trunk_change = torch.acos((torch.nn.functional.normalize(b_trunk, dim=-1) * torch.nn.functional.normalize(c_trunk, dim=-1)).sum(-1).clamp(-1.0, 1.0)) * 180.0 / math.pi
    _, b_heading, _, _ = _root_forward(b_root)
    _, c_heading, _, _ = _root_forward(c_root)
    heading_change = torch.acos((b_heading * c_heading).sum(-1).clamp(-1.0, 1.0)) * 180.0 / math.pi
    root_change = _geodesic(c_root, b_root) * 180.0 / math.pi
    root_median = max(float(torch.median(root_change[mask]).item()) if mask.any() else 0.0, 0.5)
    q_rigid = float(torch.median(trunk_change[mask]).item()) / root_median if mask.any() else None
    tail = tail_safety_metrics(m0_auth, candidate_auth, valid_mask)
    consistency = consistency_report(candidate_auth, valid_mask)
    consistency_pass = all(bool(record.get("passed", False)) for record in consistency)
    result: dict[str, Any] = {
        "protocol": PROTOCOL_NAME,
        "target_delta_deg": float(target_delta_deg),
        "angle": angle,
        "trunk_direction": _summary(trunk_change[mask], absolute=False),
        "heading": _summary(heading_change[mask], absolute=False),
        "root_change": _summary(root_change[mask], absolute=False),
        "q_rigid": q_rigid,
        "tail_safety": tail,
        "consistency": consistency,
        "finite_values": bool(torch.isfinite(m0_auth).all() and torch.isfinite(candidate_auth).all()),
    }
    foot_rows: dict[str, Any] = {}
    if m0_vertices is not None or candidate_vertices is not None or patches is not None:
        if m0_vertices is None or candidate_vertices is None or patches is None:
            raise ValueError("m0_vertices, candidate_vertices and patches must be supplied together")
        for side in ("left", "right"):
            m0_heel, m0_toe = patch_centres(m0_vertices, patches[side])
            c_heel, c_toe = patch_centres(candidate_vertices, patches[side])
            foot_rows[side] = evaluate_paired_foot(m0_heel, m0_toe, c_heel, c_toe, allow_missing_toe=allow_missing_toe)
    result["feet"] = foot_rows
    gates = [
        _gate("pelvis_pitch_mae", angle["mae_deg"], 1.0, angle["valid_count"]),
        _gate("pelvis_pitch_p95", angle["p95_deg"], 2.0, angle["valid_count"]),
        _gate("dose_sign", angle["dose_sign_correct"], True, angle["valid_count"]),
        _gate("trunk_direction_p95", result["trunk_direction"]["p95"], 2.0, result["trunk_direction"]["count"]),
        _gate("horizontal_heading_p95", result["heading"]["p95"], 2.0, result["heading"]["count"]),
        _gate("q_rigid", q_rigid, 0.2, result["trunk_direction"]["count"]),
        _gate("tail_extra_so3_jump", tail["per_sample"][0]["tail_extra_so3_jump_max_deg"], 2.0, tail["per_sample"][0]["tail_pair_count"]),
        _gate("tail_extra_pitch_step", tail["per_sample"][0]["tail_extra_pitch_step_max_deg"], 2.0, tail["per_sample"][0]["tail_pair_count"]),
        _gate("representation_consistency", consistency_pass, True, len(consistency)),
        _gate("finite_values", result["finite_values"], True, int(mask.sum().item())),
    ]
    if foot_rows:
        for side, row in foot_rows.items():
            for name, status in row["statuses"].items():
                gates.append({"name": f"{side}_{name}", "status": status, "threshold": "M0 + max(5%,1mm)", "observed": row["candidate"][name], "valid_count": row["candidate"][name]["count"], "reason": "paired M0 contact evidence"})
            toe = row["toe_contact"]
            gates.append({"name": f"{side}_toe_contact", "status": toe["status"], "threshold": "no new toe-dominance regression", "observed": toe["candidate_fraction"], "valid_count": row["contact_evidence"]["flat_contact_frames"], "reason": "fixed M0 flat-contact frames"})
    result["gates"] = gates
    effective_statuses = [
        gate["status"]
        for gate in gates
        if not (allow_missing_toe and gate["name"].endswith("_toe_contact") and gate["status"] == NOT_EVALUABLE)
    ]
    result["status"] = combine_statuses(effective_statuses)
    return result


__all__ = [
    "PROTOCOL_NAME", "GEOMETRY_PROTOCOL", "PASS", "FAIL", "NOT_EVALUABLE",
    "CONTACT_HEIGHT_M", "CONTACT_SPEED_M_PER_FRAME", "FLAT_GAP_M", "STABLE_CONFIDENCE",
    "pelvis_pitch_delta_deg", "target_root_rotation", "m0_sagittal_frame",
    "contact_evidence", "longest_true_run", "select_stable_window", "foot_patches", "patch_hash", "patch_centres",
    "evaluate_paired_foot", "evaluate_v3_pair", "combine_statuses",
]
