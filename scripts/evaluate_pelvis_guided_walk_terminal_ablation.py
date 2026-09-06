#!/usr/bin/env python3
"""Offline terminal-projection ablation for the frozen v0.4 walk cases.

This module deliberately never calls the ViMoGen sampler.  It reads the three
endpoint tensors saved by ``finalize_outputs`` and compares the endpoint before
terminal projection with the endpoint after terminal projection under the
same frozen M0, contact masks, floor heights, and foot patches.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pelvis_contact_compensation_v3 import (  # noqa: E402
    evaluate_v3_pair,
    pelvis_pitch_delta_deg,
    patch_centres,
)
from evaluation.relative_root_trunk_v2_1 import (  # noqa: E402
    direct_joints_from_motion,
    direct_smpl_parameters,
)
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe  # noqa: E402
from motion_rep.pose_authority import authority_project  # noqa: E402
from sampling.pelvis_contact_flow_projection_v0_1 import (  # noqa: E402
    DOSE_FIRST_CONTACT_ABLATION_PROTOCOL,
    write_strict_json,
)


ANALYSIS_PROTOCOL = "vimogen_pelvis_guided_walk_v0_4_terminal_ablation_v1"
ENDPOINT_FILE = "projection_artifacts/terminal_projection_endpoints.pt"
EXPECTED_MODES = (
    "dose_only",
    "position_only_medium",
    "temporal_weak",
    "temporal_medium",
    "temporal_strong",
)
EXPECTED_DOSES = (2.0, 5.0, 10.0)
MM = 1000.0
MIN_EVIDENCE = 3
EPS = 1.0e-12


def _finite_tensor(value: torch.Tensor, name: str) -> bool:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Infinity")
    return True


def _stats(values: torch.Tensor | Iterable[float], *, scale: float = 1.0) -> dict[str, Any]:
    value = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    finite = torch.isfinite(value)
    value = value[finite]
    if not value.numel():
        return {"mean": None, "p95": None, "max": None, "count": 0}
    value = value * float(scale)
    return {
        "mean": float(value.mean().item()),
        "p95": float(torch.quantile(value, 0.95).item()),
        "max": float(value.max().item()),
        "count": int(value.numel()),
    }


def _stats_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Return right-left for scalar summary fields."""

    result: dict[str, Any] = {}
    for key in ("mean", "p95", "max"):
        a, b = left.get(key), right.get(key)
        result[key] = None if a is None or b is None else float(b) - float(a)
    result["count_left"] = int(left.get("count", 0))
    result["count_right"] = int(right.get("count", 0))
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_batch(value: torch.Tensor, name: str) -> torch.Tensor:
    value = value.float()
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[-1] != MOTION_LAYOUT.total_dim:
        raise ValueError(f"{name} must have shape [1,T,276] or [T,276], got {tuple(value.shape)}")
    if value.shape[0] != 1:
        raise ValueError(f"{name} must contain exactly one sample")
    _finite_tensor(value, name)
    return value


def _resolve_run_root(v04_root: Path, value: str | os.PathLike[str]) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    marker = Path("results/phase8/pelvis_guided_walk_v0_4")
    parts = candidate.parts
    marker_parts = marker.parts
    for index in range(len(parts) - len(marker_parts) + 1):
        if parts[index : index + len(marker_parts)] == marker_parts:
            return v04_root / Path(*parts[index + len(marker_parts) :])
    return v04_root / candidate


def _load_protocol(protocol_root: Path) -> dict[str, Any]:
    protocol_path = protocol_root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol") != DOSE_FIRST_CONTACT_ABLATION_PROTOCOL:
        raise ValueError("protocol root is not the frozen v0.4 current-environment protocol")
    if protocol.get("sample_id") != "94":
        raise ValueError("terminal ablation requires sample94")
    return protocol


def _load_motion_endpoint(
    value: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    valid_mask: torch.Tensor,
    name: str,
) -> torch.Tensor:
    value = _as_batch(value, name)
    physical = value * std.view(1, 1, -1) + mean.view(1, 1, -1)
    _finite_tensor(physical, f"{name}_physical_before_authority")
    result = authority_project(
        physical,
        valid_mask=valid_mask,
        output_dtype=torch.float32,
    ).physical_motion.float()
    _finite_tensor(result, f"{name}_physical")
    return result


def _load_m0(
    protocol_root: Path,
    protocol: Mapping[str, Any],
    *,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = torch.from_numpy(np.load(protocol["inputs"]["mean"]["path"])).float()
    std = torch.from_numpy(np.load(protocol["inputs"]["std"]["path"])).float()
    if mean.shape != (MOTION_LAYOUT.total_dim,) or std.shape != mean.shape:
        raise ValueError("frozen mean/std must both have shape [276]")
    m0 = _as_batch(
        torch.load(protocol_root / "m0_physical.pt", map_location="cpu", weights_only=True),
        "m0_physical",
    )
    m0 = authority_project(m0, valid_mask=valid_mask, output_dtype=torch.float32).physical_motion.float()
    _finite_tensor(m0, "m0_physical_authoritative")
    return mean, std, m0


def _valid_prefix(valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.bool().reshape(-1)
    if valid.numel() == 0 or not bool(valid.all()):
        raise ValueError("v0.4 sample94 requires all endpoint frames to be valid")
    return valid


def _pelvis_metrics(
    m0: torch.Tensor,
    endpoint: torch.Tensor,
    valid: torch.Tensor,
    target: float,
) -> dict[str, Any]:
    m0_root = decode_rot6d_safe(m0[:, MOTION_LAYOUT.root_rotation])
    endpoint_root = decode_rot6d_safe(endpoint[:, MOTION_LAYOUT.root_rotation])
    actual = pelvis_pitch_delta_deg(m0_root.unsqueeze(0), endpoint_root.unsqueeze(0))[0]
    error = (((actual - float(target) + 180.0) % 360.0) - 180.0).abs()[valid]
    values = actual[valid]
    result = {
        "actual_dose_deg": _stats(values),
        "error_deg": _stats(error),
        "direction_correct": bool((values * float(target) >= -1.0e-5).all()) if values.numel() else False,
        "per_frame_actual_dose_deg": actual.detach().cpu().tolist(),
        "per_frame_error_deg": error.detach().cpu().tolist(),
    }
    return result


def _endpoint_contact_metrics(
    vertices: torch.Tensor,
    patches: Mapping[str, Mapping[str, list[int]]],
    side_evidence: Mapping[str, Any],
    valid: torch.Tensor,
) -> dict[str, Any]:
    heel, toe = patch_centres(vertices, patches)
    evidence = side_evidence["evidence"]
    masks = evidence["valid_masks"]
    general = torch.as_tensor(masks["general_contact"], dtype=torch.bool) & valid
    pairs = torch.as_tensor(masks["continuous_contact_pair"], dtype=torch.bool)
    floor = float(evidence["floor_height_m"])
    heel_slip = torch.linalg.vector_norm(heel[1:, :2] - heel[:-1, :2], dim=-1)[pairs]
    toe_slip = torch.linalg.vector_norm(toe[1:, :2] - toe[:-1, :2], dim=-1)[pairs]
    sole = torch.minimum(heel[:, 2], toe[:, 2])
    lift = (sole - floor).clamp_min(0.0)[general]
    penetration = (floor - sole).clamp_min(0.0)[general]
    return {
        "heel_slip_m_per_frame": _stats(heel_slip, scale=MM),
        "toe_slip_m_per_frame": _stats(toe_slip, scale=MM),
        "lift_m": _stats(lift, scale=MM),
        "penetration_m": _stats(penetration, scale=MM),
        "evidence": {
            "general_contact_frames": int(general.sum().item()),
            "continuous_contact_pairs": int(pairs.sum().item()),
            "flat_contact_frames": int(sum(bool(x) for x in masks["flat_contact"])),
            "position_status": "PASS" if int(sum(bool(x) for x in masks["flat_contact"])) >= MIN_EVIDENCE else "NOT_EVALUABLE",
            "velocity_status": "PASS" if int(pairs.sum().item()) >= MIN_EVIDENCE else "NOT_EVALUABLE",
        },
        "per_frame": {
            "heel": heel.detach().cpu().tolist(),
            "toe": toe.detach().cpu().tolist(),
            "general_contact": general.detach().cpu().tolist(),
            "continuous_contact_pair": pairs.detach().cpu().tolist(),
        },
    }


def _temporal_metrics(joints: torch.Tensor, root_translation: torch.Tensor, valid: torch.Tensor) -> dict[str, Any]:
    pair = valid[1:] & valid[:-1]
    triple = valid[2:] & valid[1:-1] & valid[:-2]
    quad = valid[3:] & valid[2:-1] & valid[1:-2] & valid[:-3]
    joint_velocity = joints[1:] - joints[:-1]
    joint_acceleration = joint_velocity[1:] - joint_velocity[:-1]
    joint_jerk = joint_acceleration[1:] - joint_acceleration[:-1]
    root_velocity = root_translation[1:] - root_translation[:-1]
    root_acceleration = root_velocity[1:] - root_velocity[:-1]
    root_jerk = root_acceleration[1:] - root_acceleration[:-1]

    def mean_joint_norm(value: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(value, dim=-1).mean(dim=-1)

    curves = {
        "root_speed_m_per_frame": (torch.linalg.vector_norm(root_velocity, dim=-1), pair),
        "root_acceleration_m_per_frame2": (torch.linalg.vector_norm(root_acceleration, dim=-1), triple),
        "root_jerk_m_per_frame3": (torch.linalg.vector_norm(root_jerk, dim=-1), quad),
        "mean_joint_speed_m_per_frame": (mean_joint_norm(joint_velocity), pair),
        "mean_joint_acceleration_m_per_frame2": (mean_joint_norm(joint_acceleration), triple),
        "mean_joint_jerk_m_per_frame3": (mean_joint_norm(joint_jerk), quad),
    }
    result: dict[str, Any] = {}
    for name, (curve, mask) in curves.items():
        result[name] = _stats(curve[mask], scale=MM)
        result[name]["per_frame"] = curve.detach().cpu().tolist()
    result["root_path_length_m"] = float(torch.linalg.vector_norm(root_velocity[pair], dim=-1).sum().item())
    return result


def _root_metrics(m0: torch.Tensor, endpoint: torch.Tensor, other: torch.Tensor, valid: torch.Tensor) -> dict[str, Any]:
    m0_t = m0[:, MOTION_LAYOUT.root_translation]
    endpoint_t = endpoint[:, MOTION_LAYOUT.root_translation]
    other_t = other[:, MOTION_LAYOUT.root_translation]
    from_m0 = torch.linalg.vector_norm(endpoint_t - m0_t, dim=-1)[valid]
    terminal_delta = torch.linalg.vector_norm(endpoint_t - other_t, dim=-1)[valid]
    m0_root = decode_rot6d_safe(m0[:, MOTION_LAYOUT.root_rotation])
    endpoint_root = decode_rot6d_safe(endpoint[:, MOTION_LAYOUT.root_rotation])
    other_root = decode_rot6d_safe(other[:, MOTION_LAYOUT.root_rotation])

    def geodesic(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        relative = left @ right.transpose(-1, -2)
        cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
        return torch.acos(cosine) * (180.0 / math.pi)

    return {
        "translation_from_m0_m": _stats(from_m0, scale=MM),
        "terminal_translation_delta_m": _stats(terminal_delta, scale=MM),
        "rotation_from_m0_deg": _stats(geodesic(endpoint_root, m0_root)[valid]),
        "terminal_rotation_delta_deg": _stats(geodesic(endpoint_root, other_root)[valid]),
        "per_frame_translation_from_m0_m": from_m0.detach().cpu().tolist(),
        "per_frame_terminal_translation_delta_m": terminal_delta.detach().cpu().tolist(),
        "per_frame_terminal_rotation_delta_deg": geodesic(endpoint_root, other_root).detach().cpu().tolist(),
    }


def _whole_body_metrics(m0_joints: torch.Tensor, endpoint_joints: torch.Tensor, m0_vertices: torch.Tensor, endpoint_vertices: torch.Tensor, valid: torch.Tensor) -> dict[str, Any]:
    joint_error = torch.linalg.vector_norm(endpoint_joints - m0_joints, dim=-1).mean(dim=-1)[valid]
    vertex_rms = torch.sqrt((endpoint_vertices - m0_vertices).square().mean(dim=(1, 2)))[valid]
    return {
        "joint_mpjpe_m": _stats(joint_error, scale=MM),
        "vertex_rms_m": _stats(vertex_rms, scale=MM),
        "per_frame_joint_mpjpe_m": joint_error.detach().cpu().tolist(),
        "per_frame_vertex_rms_m": vertex_rms.detach().cpu().tolist(),
    }


def _load_vertices(model: Any, motion: torch.Tensor, device: torch.device) -> torch.Tensor:
    params = direct_smpl_parameters(motion.unsqueeze(0).to(device))
    params = {key: value[0] for key, value in params.items()}
    with torch.inference_mode():
        return model(**params, return_verts=True).vertices.detach().cpu().float()


def _endpoint_record(
    *,
    name: str,
    m0: torch.Tensor,
    endpoint: torch.Tensor,
    other: torch.Tensor,
    valid: torch.Tensor,
    target: float,
    model: Any,
    device: torch.device,
    patches: Mapping[str, Mapping[str, list[int]]],
    sides: Mapping[str, Any],
    m0_vertices: torch.Tensor,
    m0_joints: torch.Tensor,
) -> dict[str, Any]:
    endpoint_joints = direct_joints_from_motion(endpoint.unsqueeze(0))[0].float()
    endpoint_vertices = _load_vertices(model, endpoint, device)
    _finite_tensor(endpoint_joints, f"{name}_joints")
    _finite_tensor(endpoint_vertices, f"{name}_vertices")
    paired = evaluate_v3_pair(
        m0.unsqueeze(0),
        endpoint.unsqueeze(0),
        valid.unsqueeze(0),
        target_delta_deg=float(target),
        m0_vertices=m0_vertices,
        candidate_vertices=endpoint_vertices,
        patches=patches,
    )
    foot = {
        side: _endpoint_contact_metrics(endpoint_vertices, patches[side], sides[side], valid)
        for side in ("left", "right")
    }
    return {
        "name": name,
        "pelvis": _pelvis_metrics(m0, endpoint, valid, target),
        "feet": foot,
        "root": _root_metrics(m0, endpoint, other, valid),
        "temporal": _temporal_metrics(endpoint_joints, endpoint[:, MOTION_LAYOUT.root_translation], valid),
        "whole_body": _whole_body_metrics(m0_joints, endpoint_joints, m0_vertices, endpoint_vertices, valid),
        "naturalness": {
            "trunk_direction": paired.get("trunk_direction"),
            "pelvis_neck": paired.get("uprightness", {}).get("pelvis_neck"),
            "pelvis_head": paired.get("uprightness", {}).get("pelvis_head"),
            "heading": paired.get("heading"),
            "support_drift": paired.get("uprightness", {}).get("pelvis_support_drift"),
        },
        "finite_values": bool(paired.get("finite_values", False) and torch.isfinite(endpoint).all()),
        "representation_consistency": all(bool(item.get("passed", False)) for item in paired.get("consistency", [])),
        "gate_statuses": {str(item.get("name")): item.get("status") for item in paired.get("gates", [])},
    }


def _allowed_increase(value: float | None, *, unit: str = "mm") -> float | None:
    if value is None:
        return None
    floor = 1.0 if unit == "mm" else 1.0
    return max(abs(float(value)) * 0.05, floor)


def _p95_metric(endpoint: Mapping[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = endpoint
    for key in path:
        value = value.get(key) if isinstance(value, Mapping) else None
    if not isinstance(value, Mapping):
        return None
    return None if value.get("p95") is None else float(value["p95"])


def _terminal_delta(pre: Mapping[str, Any], terminal: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, path in {
        "pelvis_mae_deg": ("pelvis", "error_deg"),
        "pelvis_p95_deg": ("pelvis", "error_deg"),
        "heel_slip_p95_mm_per_frame": ("feet", "left", "heel_slip_m_per_frame"),
        "toe_slip_p95_mm_per_frame": ("feet", "left", "toe_slip_m_per_frame"),
        "lift_p95_mm": ("feet", "left", "lift_m"),
        "penetration_p95_mm": ("feet", "left", "penetration_m"),
        "root_translation_from_m0_p95_mm": ("root", "translation_from_m0_m"),
        "root_terminal_delta_p95_mm": ("root", "terminal_translation_delta_m"),
        "mean_joint_acceleration_p95_mm_per_frame2": ("temporal", "mean_joint_acceleration_m_per_frame2"),
        "mean_joint_jerk_p95_mm_per_frame3": ("temporal", "mean_joint_jerk_m_per_frame3"),
        "joint_mpjpe_p95_mm": ("whole_body", "joint_mpjpe_m"),
        "vertex_rms_p95_mm": ("whole_body", "vertex_rms_m"),
    }.items():
        a = _p95_metric(pre, path)
        b = _p95_metric(terminal, path)
        result[name] = None if a is None or b is None else float(b - a)
    for side in ("left", "right"):
        for metric in ("heel_slip_m_per_frame", "toe_slip_m_per_frame", "lift_m", "penetration_m"):
            path = ("feet", side, metric)
            result[f"{side}_{metric}_p95_delta"] = None if _p95_metric(pre, path) is None or _p95_metric(terminal, path) is None else float(_p95_metric(terminal, path) - _p95_metric(pre, path))
    return result


def _cost_per_degree(pre: Mapping[str, Any], terminal: Mapping[str, Any], delta: Mapping[str, Any]) -> dict[str, Any]:
    pre_mae = _p95_metric(pre, ("pelvis", "error_deg"))
    terminal_mae = _p95_metric(terminal, ("pelvis", "error_deg"))
    # The requested cost uses MAE, not P95.  Keep both the denominator and
    # the cost explicit so the result cannot be mistaken for a scalar score.
    pre_mae_exact = pre.get("pelvis", {}).get("error_deg", {}).get("mean")
    terminal_mae_exact = terminal.get("pelvis", {}).get("error_deg", {}).get("mean")
    denominator = None if pre_mae_exact is None or terminal_mae_exact is None else float(pre_mae_exact) - float(terminal_mae_exact)
    result: dict[str, Any] = {
        "pelvis_mae_reduction_deg": denominator,
        "pre_p95_reference_deg": pre_mae,
        "terminal_p95_reference_deg": terminal_mae,
    }
    if denominator is None or denominator <= EPS:
        for key, value in delta.items():
            if key == "pelvis_mae_deg" or value is None:
                continue
            result[f"{key}_per_corrected_degree"] = None
        return result
    for key, value in delta.items():
        if key == "pelvis_mae_deg" or value is None:
            continue
        result[f"{key}_per_corrected_degree"] = float(value) / denominator
    return result


def _classification(pre: Mapping[str, Any], terminal: Mapping[str, Any], delta: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    reasons: list[str] = []
    if not terminal.get("finite_values") or not terminal.get("representation_consistency"):
        return "TERMINAL_HARMFUL", {"reasons": ["non_finite_or_representation_inconsistent"]}
    for name, pre_status in pre.get("gate_statuses", {}).items():
        if pre_status == "PASS" and terminal.get("gate_statuses", {}).get(name) == "FAIL":
            reasons.append(f"gate_regression:{name}")
    checks: dict[str, Any] = {}
    foot_paths = {
        "heel_slip_p95_mm_per_frame": ("heel_slip_m_per_frame",),
        "toe_slip_p95_mm_per_frame": ("toe_slip_m_per_frame",),
        "lift_p95_mm": ("lift_m",),
        "penetration_p95_mm": ("penetration_m",),
    }
    for side in ("left", "right"):
        for key, suffix in foot_paths.items():
            before = _p95_metric(pre, ("feet", side, suffix[0]))
            after = _p95_metric(terminal, ("feet", side, suffix[0]))
            allowed = _allowed_increase(before)
            check_name = f"{side}_{key}"
            checks[check_name] = {"pre": before, "terminal": after, "allowed_increase": allowed, "within": None if before is None or after is None else after <= before + allowed + EPS}
            if checks[check_name]["within"] is False:
                reasons.append(f"tradeoff:{check_name}")
    for key, path in {
        "joint_mpjpe_p95_mm": ("whole_body", "joint_mpjpe_m"),
        "vertex_rms_p95_mm": ("whole_body", "vertex_rms_m"),
    }.items():
        before = _p95_metric(pre, path)
        after = _p95_metric(terminal, path)
        allowed = _allowed_increase(before)
        checks[key] = {"pre": before, "terminal": after, "allowed_increase": allowed, "within": None if before is None or after is None else after <= before + allowed + EPS}
        if checks[key]["within"] is False:
            reasons.append(f"tradeoff:{key}")
    root_p95 = _p95_metric(terminal, ("root", "terminal_translation_delta_m"))
    root_max = terminal.get("root", {}).get("terminal_translation_delta_m", {}).get("max")
    checks["terminal_root_translation"] = {"p95_mm": root_p95, "max_mm": root_max, "within": root_p95 is not None and root_max is not None and root_p95 <= 1.0 + EPS and root_max <= 10.0 + EPS}
    if checks["terminal_root_translation"]["within"] is False:
        reasons.append("tradeoff:terminal_root_translation")
    for key in ("mean_joint_acceleration_p95_mm_per_frame2", "mean_joint_jerk_p95_mm_per_frame3"):
        before = _p95_metric(pre, ("temporal", "mean_joint_acceleration_m_per_frame2" if "acceleration" in key else "mean_joint_jerk_m_per_frame3"))
        after = None if before is None else before + float(delta.get(key, 0.0) or 0.0)
        allowed = _allowed_increase(before)
        checks[key] = {"pre": before, "terminal": after, "allowed_increase": allowed, "within": None if before is None or after is None else after <= before + allowed + EPS}
        if checks[key]["within"] is False:
            reasons.append(f"tradeoff:{key}")
    penetration_after_values = [checks[f"{side}_penetration_p95_mm"]["terminal"] for side in ("left", "right")]
    if any(value is not None and value > 10.0 for value in penetration_after_values) and any(reason.startswith("tradeoff:") and "penetration_p95_mm" in reason for reason in reasons):
        return "TERMINAL_HARMFUL", {"reasons": reasons, "checks": checks}
    if reasons:
        return "TERMINAL_TRADEOFF", {"reasons": reasons, "checks": checks}
    return "TERMINAL_SAFE", {"reasons": [], "checks": checks}


def _flatten_row(record: Mapping[str, Any]) -> dict[str, Any]:
    pre, terminal = record["pre_cast"], record["terminal"]
    delta = record["delta"]
    row: dict[str, Any] = {
        "case_id": record["case_id"],
        "mode": record["mode"],
        "dose_deg": record["dose_deg"],
        "classification": record["classification"],
        "pelvis_pre_mae_deg": pre["pelvis"]["error_deg"]["mean"],
        "pelvis_terminal_mae_deg": terminal["pelvis"]["error_deg"]["mean"],
        "pelvis_delta_mae_deg": delta.get("pelvis_mae_deg"),
        "pelvis_pre_p95_deg": pre["pelvis"]["error_deg"]["p95"],
        "pelvis_terminal_p95_deg": terminal["pelvis"]["error_deg"]["p95"],
        "heel_slip_pre_p95_mm_per_frame": _p95_metric(pre, ("feet", "left", "heel_slip_m_per_frame")),
        "heel_slip_terminal_p95_mm_per_frame": _p95_metric(terminal, ("feet", "left", "heel_slip_m_per_frame")),
        "heel_slip_delta_p95_mm_per_frame": delta.get("heel_slip_p95_mm_per_frame"),
        "toe_slip_pre_p95_mm_per_frame": _p95_metric(pre, ("feet", "left", "toe_slip_m_per_frame")),
        "toe_slip_terminal_p95_mm_per_frame": _p95_metric(terminal, ("feet", "left", "toe_slip_m_per_frame")),
        "toe_slip_delta_p95_mm_per_frame": delta.get("toe_slip_p95_mm_per_frame"),
        "lift_pre_p95_mm": _p95_metric(pre, ("feet", "left", "lift_m")),
        "lift_terminal_p95_mm": _p95_metric(terminal, ("feet", "left", "lift_m")),
        "lift_delta_p95_mm": delta.get("lift_p95_mm"),
        "penetration_pre_p95_mm": _p95_metric(pre, ("feet", "left", "penetration_m")),
        "penetration_terminal_p95_mm": _p95_metric(terminal, ("feet", "left", "penetration_m")),
        "penetration_delta_p95_mm": delta.get("penetration_p95_mm"),
        "terminal_root_translation_p95_mm": _p95_metric(terminal, ("root", "terminal_translation_delta_m")),
        "terminal_root_translation_max_mm": terminal["root"]["terminal_translation_delta_m"]["max"],
        "terminal_root_rotation_p95_deg": _p95_metric(terminal, ("root", "terminal_rotation_delta_deg")),
        "mean_joint_acceleration_pre_p95_mm_per_frame2": _p95_metric(pre, ("temporal", "mean_joint_acceleration_m_per_frame2")),
        "mean_joint_acceleration_terminal_p95_mm_per_frame2": _p95_metric(terminal, ("temporal", "mean_joint_acceleration_m_per_frame2")),
        "mean_joint_acceleration_delta_p95_mm_per_frame2": delta.get("mean_joint_acceleration_p95_mm_per_frame2"),
        "mean_joint_jerk_delta_p95_mm_per_frame3": delta.get("mean_joint_jerk_p95_mm_per_frame3"),
        "joint_mpjpe_delta_p95_mm": delta.get("joint_mpjpe_p95_mm"),
        "vertex_rms_delta_p95_mm": delta.get("vertex_rms_p95_mm"),
        "right_position_status_pre": pre["feet"]["right"]["evidence"]["position_status"],
        "right_position_status_terminal": terminal["feet"]["right"]["evidence"]["position_status"],
        "right_velocity_status_pre": pre["feet"]["right"]["evidence"]["velocity_status"],
        "right_velocity_status_terminal": terminal["feet"]["right"]["evidence"]["velocity_status"],
    }
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pad_per_frame(values: Iterable[Any], length: int = 100) -> list[Any]:
    """Align pair/triple/quad difference curves to the original frame axis."""

    result = list(values)
    if len(result) > length:
        raise ValueError(f"per-frame curve has {len(result)} values; expected at most {length}")
    return result + [""] * (length - len(result))


def _write_summary_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    rows = payload["rows"]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    lines = [
        "# v0.4 Terminal Projection Offline Ablation",
        "",
        f"- Protocol: `{ANALYSIS_PROTOCOL}`",
        "- Scope: 15 existing sample94/seed0 cases; no motion regeneration.",
        "- Main comparison: `official_pre_cast_norm` → `terminal_projection_norm`.",
        "- `last_sampling_projection_norm` is retained only as a scheduler-rebound audit.",
        "",
        "## Classification",
        "",
        " | ".join(["status", "count"]),
        " | ".join(["---", "---"]),
    ]
    lines.extend(f"| {key} | {counts[key]} |" for key in sorted(counts))
    lines.extend([
        "",
        "## Per-case summary",
        "",
        "| case | dose | pelvis MAE pre | pelvis MAE terminal | heel slip Δ P95 (mm/frame) | toe slip Δ P95 (mm/frame) | penetration Δ P95 (mm) | terminal root Δ P95 (mm) | joint MPJPE Δ P95 (mm) | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in payload["flat_rows"]:
        def fmt(value: Any) -> str:
            return "NA" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| {row['case_id']} | {row['dose_deg']:g} | {fmt(row['pelvis_pre_mae_deg'])} | {fmt(row['pelvis_terminal_mae_deg'])} | {fmt(row['heel_slip_delta_p95_mm_per_frame'])} | {fmt(row['toe_slip_delta_p95_mm_per_frame'])} | {fmt(row['penetration_delta_p95_mm'])} | {fmt(row['terminal_root_translation_p95_mm'])} | {fmt(row['joint_mpjpe_delta_p95_mm'])} | {row['classification']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Positive non-pelvis Δ means terminal projection increased that metric. The denominator for cost-per-degree is the reduction in pelvis MAE; no cross-unit scalar score is used.",
        "",
        "Right-foot position evidence with fewer than three frozen flat-contact frames remains `NOT_EVALUABLE` and is never promoted to PASS.",
        "",
        "The historical `pre_residuals.pelvis_geodesic_rms_deg` field is not used because the low-memory terminal path constructed it after replacing the root rotation with the target. This analysis recomputes the value from the saved official pre-cast endpoint.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plots(output: Path, flat_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - server dependency check
        raise RuntimeError("matplotlib is required for terminal ablation plots") from exc
    output.mkdir(parents=True, exist_ok=True)
    doses = np.asarray([float(row["dose_deg"]) for row in flat_rows], dtype=float)
    gain = np.asarray([-float(row["pelvis_delta_mae_deg"] or 0.0) for row in flat_rows], dtype=float)
    mode = [str(row["mode"]) for row in flat_rows]
    palette = {name: color for name, color in zip(EXPECTED_MODES, ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"))}

    def scatter(y_key: str, ylabel: str, filename: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, row in enumerate(flat_rows):
            y = row.get(y_key)
            if y is None:
                continue
            ax.scatter(gain[idx], float(y), color=palette.get(mode[idx], "#333333"), marker={2.0: "o", 5.0: "s", 10.0: "^"}.get(doses[idx], "o"), s=55)
        ax.axhline(0.0, color="#888888", linewidth=0.8)
        ax.set_xlabel("Pelvis MAE reduction (deg; positive is better)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)

    scatter("heel_slip_delta_p95_mm_per_frame", "Heel slip Δ P95 (mm/frame)", "pelvis_gain_vs_heel_slip_delta.png", "Pelvis gain vs heel-slip cost")
    scatter("penetration_delta_p95_mm", "Penetration Δ P95 (mm)", "pelvis_gain_vs_penetration_delta.png", "Pelvis gain vs penetration cost")
    scatter("terminal_root_translation_p95_mm", "Terminal root translation Δ P95 (mm)", "terminal_root_translation_delta.png", "Terminal root-translation edit")
    scatter("joint_mpjpe_delta_p95_mm", "Joint MPJPE Δ P95 (mm)", "temporal_whole_body_delta.png", "Temporal / whole-body deviation")


def evaluate_all(
    *,
    v04_root: Path,
    protocol_root: Path,
    summary_path: Path,
    output: Path,
    device_name: str,
    plots: bool = True,
) -> dict[str, Any]:
    protocol = _load_protocol(protocol_root)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_rows = summary.get("rows")
    if not isinstance(source_rows, list):
        raise ValueError("v0.4 summary must contain a rows list")
    expected = {(float(dose), mode) for dose in EXPECTED_DOSES for mode in EXPECTED_MODES}
    actual = {(float(row.get("dose")), str(row.get("mode"))) for row in source_rows}
    if actual != expected or len(source_rows) != 15:
        raise ValueError(f"summary must contain exactly 15 unique dose/mode rows; got {len(source_rows)}")

    valid_mask = torch.load(protocol_root / "valid_mask.pt", map_location="cpu", weights_only=True).bool()
    valid_mask = valid_mask[0:1] if valid_mask.ndim == 2 else valid_mask.unsqueeze(0)
    valid = _valid_prefix(valid_mask[0])
    if valid.numel() != 100:
        raise ValueError(f"sample94 v0.4 terminal ablation expects 100 frames, got {valid.numel()}")
    mean, std, m0_batch = _load_m0(protocol_root, protocol, valid_mask=valid_mask)
    m0 = m0_batch[0]
    patches = json.loads((protocol_root / "foot_patches.json").read_text(encoding="utf-8"))
    case = next(item for item in protocol["cases"] if str(item["sample_id"]) == "94")
    sides = case["sides"]
    device = torch.device(device_name)
    from smplx import SMPLX  # imported only on the server/runtime with SMPL-X installed

    model = SMPLX(
        model_path=protocol["inputs"]["smplx_model"]["path"],
        gender="neutral",
        num_betas=10,
        batch_size=int(valid.numel()),
        use_pca=False,
    ).to(device)
    m0_joints = direct_joints_from_motion(m0.unsqueeze(0))[0].float()
    m0_vertices = _load_vertices(model, m0, device)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for source in (
        protocol_root / "protocol.json",
        protocol_root / "valid_mask.pt",
        protocol_root / "m0_physical.pt",
        protocol_root / "foot_patches.json",
        summary_path,
    ):
        hashes[str(source)] = _sha256(source)
    for source_row in sorted(source_rows, key=lambda row: (float(row["dose"]), str(row["mode"]))):
        run_root = _resolve_run_root(v04_root, source_row["run_root"])
        endpoint_path = run_root / ENDPOINT_FILE
        if not endpoint_path.is_file():
            raise FileNotFoundError(endpoint_path)
        hashes[str(endpoint_path)] = _sha256(endpoint_path)
        run_record_path = run_root / "run_record.json"
        if not run_record_path.is_file():
            raise FileNotFoundError(run_record_path)
        hashes[str(run_record_path)] = _sha256(run_record_path)
        payload = torch.load(endpoint_path, map_location="cpu", weights_only=True)
        for key in ("official_pre_cast_norm", "terminal_projection_norm", "last_sampling_projection_norm"):
            if key not in payload:
                raise ValueError(f"{endpoint_path} lacks {key}")
        pre = _load_motion_endpoint(payload["official_pre_cast_norm"], mean=mean, std=std, valid_mask=valid_mask, name="official_pre_cast")
        terminal = _load_motion_endpoint(payload["terminal_projection_norm"], mean=mean, std=std, valid_mask=valid_mask, name="terminal_projection")
        pre_motion, terminal_motion = pre[0], terminal[0]
        pre_record = _endpoint_record(name="pre_cast", m0=m0, endpoint=pre_motion, other=terminal_motion, valid=valid, target=float(source_row["dose"]), model=model, device=device, patches=patches, sides=sides, m0_vertices=m0_vertices, m0_joints=m0_joints)
        terminal_record = _endpoint_record(name="terminal", m0=m0, endpoint=terminal_motion, other=pre_motion, valid=valid, target=float(source_row["dose"]), model=model, device=device, patches=patches, sides=sides, m0_vertices=m0_vertices, m0_joints=m0_joints)
        delta = _terminal_delta(pre_record, terminal_record)
        cost = _cost_per_degree(pre_record, terminal_record, delta)
        classification, decision = _classification(pre_record, terminal_record, delta)
        case_id = f"{source_row['mode']}_plus{float(source_row['dose']):g}deg"
        record = {
            "case_id": case_id,
            "run_root": str(run_root),
            "mode": str(source_row["mode"]),
            "dose_deg": float(source_row["dose"]),
            "source_endpoint_sha256": hashes[str(endpoint_path)],
            "pre_cast": pre_record,
            "terminal": terminal_record,
            "delta": delta,
            "cost_per_corrected_degree": cost,
            "classification": classification,
            "decision": decision,
            "last_sampling_projection_available": payload["last_sampling_projection_norm"] is not None,
        }
        records.append(record)
        per_case = output / "per_case" / case_id
        per_case.mkdir(parents=True, exist_ok=True)
        per_frame = {
            "frame": list(range(100)),
            "pelvis_error_pre_deg": pre_record["pelvis"]["per_frame_error_deg"],
            "pelvis_error_terminal_deg": terminal_record["pelvis"]["per_frame_error_deg"],
            "root_translation_delta_terminal_m": terminal_record["root"]["per_frame_terminal_translation_delta_m"],
            "root_rotation_delta_terminal_deg": terminal_record["root"]["per_frame_terminal_rotation_delta_deg"],
            "joint_mpjpe_pre_mm": pre_record["whole_body"]["per_frame_joint_mpjpe_m"],
            "joint_mpjpe_terminal_mm": terminal_record["whole_body"]["per_frame_joint_mpjpe_m"],
            "vertex_rms_pre_mm": pre_record["whole_body"]["per_frame_vertex_rms_m"],
            "vertex_rms_terminal_mm": terminal_record["whole_body"]["per_frame_vertex_rms_m"],
        }
        for key in (
            "root_speed_m_per_frame",
            "root_acceleration_m_per_frame2",
            "root_jerk_m_per_frame3",
            "mean_joint_speed_m_per_frame",
            "mean_joint_acceleration_m_per_frame2",
            "mean_joint_jerk_m_per_frame3",
        ):
            per_frame[f"{key}_pre"] = _pad_per_frame(pre_record["temporal"][key]["per_frame"])
            per_frame[f"{key}_terminal"] = _pad_per_frame(terminal_record["temporal"][key]["per_frame"])
        for side in ("left", "right"):
            for key in ("heel", "toe"):
                pre_values = pre_record["feet"][side]["per_frame"][key]
                terminal_values = terminal_record["feet"][side]["per_frame"][key]
                for axis_index, axis_name in enumerate(("x", "y", "z")):
                    per_frame[f"{side}_{key}_pre_{axis_name}_m"] = [float(value[axis_index]) for value in pre_values]
                    per_frame[f"{side}_{key}_terminal_{axis_name}_m"] = [float(value[axis_index]) for value in terminal_values]
                pre_speed = [float("nan")] + [float(torch.linalg.vector_norm(torch.as_tensor(pre_values[i][:2]) - torch.as_tensor(pre_values[i - 1][:2])).item()) * MM for i in range(1, 100)]
                terminal_speed = [float("nan")] + [float(torch.linalg.vector_norm(torch.as_tensor(terminal_values[i][:2]) - torch.as_tensor(terminal_values[i - 1][:2])).item()) * MM for i in range(1, 100)]
                per_frame[f"{side}_{key}_slip_pre_mm_per_frame"] = pre_speed
                per_frame[f"{side}_{key}_slip_terminal_mm_per_frame"] = terminal_speed
        with (per_case / "per_frame.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_frame.keys()))
            writer.writeheader()
            for index in range(100):
                writer.writerow({key: value[index] for key, value in per_frame.items()})

    flat_rows = [_flatten_row(record) for record in records]
    payload = {
        "protocol": ANALYSIS_PROTOCOL,
        "source_generation_protocol": DOSE_FIRST_CONTACT_ABLATION_PROTOCOL,
        "scope": {"sample_id": "94", "seed": 0, "frame_count": 100, "case_count": len(records), "regeneration": False},
        "endpoint_definition": {"pre_cast": "official_pre_cast_norm", "terminal": "terminal_projection_norm", "audit_only": "last_sampling_projection_norm"},
        "source_hashes": hashes,
        "metric_delta_definition": "terminal - pre_cast; positive non-pelvis delta means worse",
        "rows": records,
        "flat_rows": flat_rows,
    }
    write_strict_json(output / "terminal_ablation.json", payload)
    _write_csv(output / "terminal_ablation.csv", flat_rows)
    _write_summary_markdown(output / "terminal_ablation_summary.md", payload)
    if plots:
        _write_plots(output / "figures", flat_rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v04-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    summary = args.summary or (args.v04_root / "v0_4_ablation_summary.json")
    payload = evaluate_all(
        v04_root=args.v04_root,
        protocol_root=args.protocol_root,
        summary_path=summary,
        output=args.output,
        device_name=args.device,
        plots=not args.no_plots,
    )
    counts: dict[str, int] = {}
    for row in payload["rows"]:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    print(json.dumps({"protocol": ANALYSIS_PROTOCOL, "cases": len(payload["rows"]), "classifications": counts, "output": str(args.output)}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
