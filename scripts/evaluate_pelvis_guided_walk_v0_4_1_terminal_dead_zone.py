#!/usr/bin/env python3
"""Offline v0.4.1 terminal dead-zone and pelvis/contact attribution audit.

The command reads existing v0.4 endpoint tensors and performs only the small
terminal projection on saved endpoints.  It never invokes the ViMoGen
sampler.  The same frozen M0, contact evidence, floor and patches are used for
every dead-zone and causal switch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pelvis_contact_compensation_v3 import patch_centres  # noqa: E402
from evaluation.relative_root_trunk_v2_1 import direct_joints_from_motion  # noqa: E402
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe  # noqa: E402
from motion_rep.pose_authority import authority_project  # noqa: E402
from sampling.pelvis_contact_flow_projection_v0_1 import (  # noqa: E402
    DOSE_FIRST_CONTACT_ABLATION_PROTOCOL,
    ProjectorConfig,
    PelvisContactFlowProjector,
    write_strict_json,
)
from sampling.terminal_projection_policy import TerminalProjectionPolicy  # noqa: E402


ANALYSIS_PROTOCOL = "vimogen_pelvis_guided_walk_v0_4_1_terminal_dead_zone_v1"
SOURCE_PROTOCOL = DOSE_FIRST_CONTACT_ABLATION_PROTOCOL
ENDPOINT_FILE = "projection_artifacts/terminal_projection_endpoints.pt"
MODES = {
    "dose_only": (0.0, 0.0),
    "position_only_medium": (1.0e5, 0.0),
    "temporal_weak": (1.0e4, 1.0e4),
}
DEAD_ZONES = (0.0, 0.1, 0.25, 0.5)
MM = 1000.0
EPS = 1.0e-8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Infinity")


def _stats(value: torch.Tensor, scale: float = 1.0) -> dict[str, Any]:
    value = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    _finite(value, "metric")
    if not value.numel():
        return {"mean": None, "p95": None, "max": None, "count": 0}
    value = value * float(scale)
    return {
        "mean": float(value.mean()),
        "p95": float(torch.quantile(value, 0.95)),
        "max": float(value.max()),
        "count": int(value.numel()),
    }


def _stats_signed(value: torch.Tensor, scale: float = 1.0) -> dict[str, Any]:
    value = torch.as_tensor(value, dtype=torch.float64)
    _finite(value, "signed metric")
    return {"vector_mean": (value.mean(dim=0) * scale).tolist(), "norm": _stats(torch.linalg.vector_norm(value, dim=-1), scale)}


def _as_batch(value: torch.Tensor, name: str) -> torch.Tensor:
    value = value.float()
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != 276:
        raise ValueError(f"{name} must have shape [1,T,276], got {tuple(value.shape)}")
    _finite(value, name)
    return value


def _load_motion(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, valid: torch.Tensor, name: str) -> torch.Tensor:
    norm = _as_batch(value, name)
    physical = norm * std.view(1, 1, -1) + mean.view(1, 1, -1)
    return authority_project(physical, valid_mask=valid, output_dtype=torch.float32).physical_motion.float()


def _load_frozen(protocol_root: Path) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
    protocol = json.loads((protocol_root / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError("protocol root is not the frozen v0.4 protocol")
    if str(protocol.get("sample_id")) != "94":
        raise ValueError("v0.4.1 requires sample94")
    mean = torch.from_numpy(np.load(protocol["inputs"]["mean"]["path"])).float()
    std = torch.from_numpy(np.load(protocol["inputs"]["std"]["path"])).float()
    m0 = torch.load(protocol_root / "m0_physical.pt", map_location="cpu", weights_only=True).float()
    valid = torch.load(protocol_root / "valid_mask.pt", map_location="cpu", weights_only=True).bool()
    if m0.ndim == 2:
        m0 = m0.unsqueeze(0)
    if valid.ndim == 1:
        valid = valid.unsqueeze(0)
    if m0.shape[-1] != 276 or m0.shape[:2] != valid.shape:
        raise ValueError("frozen M0/valid mask dimensions differ")
    m0 = authority_project(m0, valid_mask=valid, output_dtype=torch.float32).physical_motion.float()
    patches = json.loads((protocol_root / "foot_patches.json").read_text(encoding="utf-8"))
    case = next(item for item in protocol["cases"] if str(item.get("sample_id")) == "94")
    return protocol, mean, std, m0, valid, patches, case["sides"]


def _resolve_run_root(v04_root: Path, value: str | Path) -> Path:
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


def _endpoint_foot_metrics(vertices: torch.Tensor, patches: Mapping[str, list[int]], side: Mapping[str, Any], valid: torch.Tensor) -> dict[str, Any]:
    heel, toe = patch_centres(vertices, patches)
    evidence = side["evidence"]
    masks = evidence["valid_masks"]
    general = torch.as_tensor(masks["general_contact"], dtype=torch.bool) & valid
    pairs = torch.as_tensor(masks["continuous_contact_pair"], dtype=torch.bool)
    floor = float(evidence["floor_height_m"])
    heel_slip = torch.linalg.vector_norm(heel[1:, :2] - heel[:-1, :2], dim=-1)[pairs]
    toe_slip = torch.linalg.vector_norm(toe[1:, :2] - toe[:-1, :2], dim=-1)[pairs]
    sole = torch.minimum(heel[:, 2], toe[:, 2])
    lift = (sole - floor).clamp_min(0.0)[general]
    penetration = (floor - sole).clamp_min(0.0)[general]
    flat_count = int(sum(bool(item) for item in masks["flat_contact"]))
    pair_count = int(pairs.sum())
    return {
        "heel_slip_p95_mm_per_frame": _stats(heel_slip, MM),
        "toe_slip_p95_mm_per_frame": _stats(toe_slip, MM),
        "lift_p95_mm": _stats(lift, MM),
        "penetration_p95_mm": _stats(penetration, MM),
        "evidence": {
            "general_contact_frames": int(general.sum()),
            "continuous_contact_pairs": pair_count,
            "flat_contact_frames": flat_count,
            "position_status": "PASS" if flat_count >= 3 else "NOT_EVALUABLE",
            "velocity_status": "PASS" if pair_count >= 3 else "NOT_EVALUABLE",
        },
        "per_frame": {"heel": heel.tolist(), "toe": toe.tolist(), "general_contact": general.tolist(), "continuous_contact_pair": pairs.tolist()},
    }


def _geodesic(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    relative = a @ b.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cosine) * (180.0 / math.pi)


def _root_decomposition(m0: torch.Tensor, endpoint: torch.Tensor, pre: torch.Tensor, valid: torch.Tensor, vertical_axis: int) -> dict[str, Any]:
    m0_t = m0[:, MOTION_LAYOUT.root_translation]
    endpoint_t = endpoint[:, MOTION_LAYOUT.root_translation]
    pre_t = pre[:, MOTION_LAYOUT.root_translation]
    terminal = endpoint_t - pre_t
    overall = endpoint_t - m0_t
    offset = overall[valid].mean(dim=0)
    aligned = overall - offset
    horizontal = [axis for axis in range(3) if axis != vertical_axis]
    return {
        "terminal_increment": {
            "xyz_m": _stats_signed(terminal[valid], MM),
            "horizontal_mm": _stats(torch.linalg.vector_norm(terminal[valid][:, horizontal], dim=-1), MM),
            "vertical_abs_mm": _stats(terminal[valid, vertical_axis].abs(), MM),
        },
        "overall_from_m0": {
            "xyz_m": _stats_signed(overall[valid], MM),
            "horizontal_mm": _stats(torch.linalg.vector_norm(overall[valid][:, horizontal], dim=-1), MM),
            "vertical_abs_mm": _stats(overall[valid, vertical_axis].abs(), MM),
        },
        "global_offset": {"xyz_mm": (offset * MM).tolist(), "norm": _stats(offset[None], MM)},
        "aligned_shape_deviation": {
            "xyz_m": _stats_signed(aligned[valid], MM),
            "horizontal_mm": _stats(torch.linalg.vector_norm(aligned[valid][:, horizontal], dim=-1), MM),
            "vertical_abs_mm": _stats(aligned[valid, vertical_axis].abs(), MM),
        },
        "path_length_m": float(torch.linalg.vector_norm((endpoint_t[1:] - endpoint_t[:-1])[valid[1:] & valid[:-1]], dim=-1).sum()),
        "pre_path_length_m": float(torch.linalg.vector_norm((pre_t[1:] - pre_t[:-1])[valid[1:] & valid[:-1]], dim=-1).sum()),
    }


def _temporal(endpoint_joints: torch.Tensor, root_t: torch.Tensor, valid: torch.Tensor) -> dict[str, Any]:
    pair, triple, quad = valid[1:] & valid[:-1], valid[2:] & valid[1:-1] & valid[:-2], valid[3:] & valid[2:-1] & valid[1:-2] & valid[:-3]
    jv = endpoint_joints[1:] - endpoint_joints[:-1]
    ja = jv[1:] - jv[:-1]
    jj = ja[1:] - ja[:-1]
    rv = root_t[1:] - root_t[:-1]
    ra = rv[1:] - rv[:-1]
    rj = ra[1:] - ra[:-1]
    mean_norm = lambda x: torch.linalg.vector_norm(x, dim=-1).mean(dim=-1)
    curves = {
        "root_speed": (torch.linalg.vector_norm(rv, dim=-1), pair),
        "root_acceleration": (torch.linalg.vector_norm(ra, dim=-1), triple),
        "mean_joint_speed": (mean_norm(jv), pair),
        "mean_joint_acceleration": (mean_norm(ja), triple),
        "mean_joint_jerk": (mean_norm(jj), quad),
    }
    result = {}
    for name, (curve, mask) in curves.items():
        result[name + "_p95_mm"] = _stats(curve[mask], MM)
        result[name + "_per_frame"] = curve.tolist()
    return result


def _load_vertices(model: Any, motion: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    joints = direct_joints_from_motion(motion.unsqueeze(0))[0].float()
    params_module = __import__("evaluation.relative_root_trunk_v2_1", fromlist=["direct_smpl_parameters"])
    params = params_module.direct_smpl_parameters(motion.unsqueeze(0).to(device))
    params = {key: value[0] for key, value in params.items()}
    with torch.inference_mode():
        vertices = model(**params, return_verts=True).vertices.detach().cpu().float()
    _finite(vertices, "vertices")
    return joints, vertices


def _endpoint_record(m0: torch.Tensor, endpoint: torch.Tensor, pre: torch.Tensor, valid: torch.Tensor, target: float, model: Any, device: torch.device, patches: Mapping[str, Any], sides: Mapping[str, Any], m0_joints: torch.Tensor, m0_vertices: torch.Tensor, vertical_axis: int) -> dict[str, Any]:
    joints, vertices = _load_vertices(model, endpoint, device)
    root0 = decode_rot6d_safe(m0[:, MOTION_LAYOUT.root_rotation])
    root = decode_rot6d_safe(endpoint[:, MOTION_LAYOUT.root_rotation])
    actual = __import__("evaluation.pelvis_contact_compensation_v3", fromlist=["pelvis_pitch_delta_deg"]).pelvis_pitch_delta_deg(root0, root)
    error = (actual - float(target)).abs()[valid]
    feet = {side: _endpoint_foot_metrics(vertices, patches[side], sides[side], valid) for side in ("left", "right")}
    return {
        "pelvis": {"actual_dose_deg": _stats(actual[valid]), "error_deg": _stats(error), "direction_correct": bool((actual[valid] * target >= -1e-5).all()), "per_frame_actual_dose_deg": actual.tolist(), "per_frame_error_deg": (actual - float(target)).abs().tolist()},
        "feet": feet,
        "root": _root_decomposition(m0, endpoint, pre, valid, vertical_axis),
        "temporal": _temporal(joints, endpoint[:, MOTION_LAYOUT.root_translation], valid),
        "whole_body": {"joint_mpjpe_mm": _stats(torch.linalg.vector_norm(joints - m0_joints, dim=-1).mean(dim=-1)[valid], MM), "vertex_rms_mm": _stats(torch.sqrt((vertices - m0_vertices).square().mean(dim=(1, 2)))[valid], MM), "per_frame_joint_mpjpe_mm": torch.linalg.vector_norm(joints - m0_joints, dim=-1).mean(dim=-1).tolist(), "per_frame_vertex_rms_mm": torch.sqrt((vertices - m0_vertices).square().mean(dim=(1, 2))).tolist()},
        "finite_values": True,
    }


def _get(path: Mapping[str, Any], *keys: str) -> Any:
    value: Any = path
    for key in keys:
        value = value[key]
    return value


def _metric_delta(pre: Mapping[str, Any], terminal: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side in ("left", "right"):
        for metric in ("heel_slip_p95_mm_per_frame", "toe_slip_p95_mm_per_frame", "lift_p95_mm", "penetration_p95_mm"):
            a = pre["feet"][side][metric]["p95"]
            b = terminal["feet"][side][metric]["p95"]
            result[f"{side}_{metric}_delta"] = None if a is None or b is None else float(b - a)
    for name, a_path, b_path in (
        ("pelvis_mae_deg", ("pelvis", "error_deg", "mean"), ("pelvis", "error_deg", "mean")),
        ("pelvis_p95_deg", ("pelvis", "error_deg", "p95"), ("pelvis", "error_deg", "p95")),
        ("terminal_root_translation_p95_mm", ("root", "terminal_increment", "xyz_m", "norm", "p95"), ("root", "terminal_increment", "xyz_m", "norm", "p95")),
        ("overall_root_from_m0_p95_mm", ("root", "overall_from_m0", "xyz_m", "norm", "p95"), ("root", "overall_from_m0", "xyz_m", "norm", "p95")),
        ("aligned_root_shape_p95_mm", ("root", "aligned_shape_deviation", "xyz_m", "norm", "p95"), ("root", "aligned_shape_deviation", "xyz_m", "norm", "p95")),
        ("mean_joint_acceleration_p95_mm", ("temporal", "mean_joint_acceleration_p95_mm", "p95"), ("temporal", "mean_joint_acceleration_p95_mm", "p95")),
        ("mean_joint_jerk_p95_mm", ("temporal", "mean_joint_jerk_p95_mm", "p95"), ("temporal", "mean_joint_jerk_p95_mm", "p95")),
        ("joint_mpjpe_p95_mm", ("whole_body", "joint_mpjpe_mm", "p95"), ("whole_body", "joint_mpjpe_mm", "p95")),
    ):
        av = _get(pre, *a_path)
        bv = _get(terminal, *b_path)
        result[name] = None if av is None or bv is None else float(bv - av)
    return result


def _classification(pre: Mapping[str, Any], terminal: Mapping[str, Any], delta: Mapping[str, Any], dead_zone: float) -> tuple[str, dict[str, Any]]:
    checks: dict[str, Any] = {}
    reasons: list[str] = []
    pelvis = terminal["pelvis"]["error_deg"]
    checks["pelvis_accuracy"] = {"mae_deg": pelvis["mean"], "p95_deg": pelvis["p95"], "within": pelvis["mean"] <= 1.0 + EPS and pelvis["p95"] <= 2.0 + EPS and pelvis["max"] <= dead_zone + 1.0e-4}
    if not checks["pelvis_accuracy"]["within"]:
        reasons.append("pelvis_accuracy")
    for key, limit in (("terminal_root_translation_p95_mm", 1.0),):
        value = terminal["root"]["terminal_increment"]["xyz_m"]["norm"]["p95"]
        maximum = terminal["root"]["terminal_increment"]["xyz_m"]["norm"]["max"]
        checks[key] = {"p95": value, "max": maximum, "within": value <= limit + EPS and maximum <= 10.0 + EPS}
        if not checks[key]["within"]:
            reasons.append(key)
    for side in ("left", "right"):
        for metric in ("heel_slip_p95_mm_per_frame", "toe_slip_p95_mm_per_frame", "lift_p95_mm", "penetration_p95_mm"):
            before = pre["feet"][side][metric]["p95"]
            after = terminal["feet"][side][metric]["p95"]
            if before is None or after is None:
                continue
            allowed = max(abs(before) * 0.05, 1.0)
            within = after <= before + allowed + EPS
            checks[f"{side}_{metric}"] = {"pre": before, "terminal": after, "allowed_increase": allowed, "within": within}
            if not within:
                reasons.append(f"{side}_{metric}")
    for key in ("mean_joint_acceleration_p95_mm", "mean_joint_jerk_p95_mm", "joint_mpjpe_p95_mm"):
        value = delta.get(key)
        if value is None:
            continue
        pre_value = _get(pre, "temporal", "mean_joint_acceleration_p95_mm", "p95") if "acceleration" in key else (_get(pre, "temporal", "mean_joint_jerk_p95_mm", "p95") if "jerk" in key else _get(pre, "whole_body", "joint_mpjpe_mm", "p95"))
        allowed = max(abs(pre_value) * 0.05, 1.0)
        checks[key] = {"delta": value, "allowed_increase": allowed, "within": value <= allowed + EPS}
        if not checks[key]["within"]:
            reasons.append(key)
    if any("penetration" in reason for reason in reasons) and terminal["feet"]["left"]["penetration_p95_mm"]["p95"] > 10.0:
        return "TERMINAL_HARMFUL", {"reasons": reasons, "checks": checks}
    if reasons:
        return "TERMINAL_TRADEOFF", {"reasons": reasons, "checks": checks}
    return "TERMINAL_STAGE_SAFE", {"reasons": [], "checks": checks}


def _flat_row(record: Mapping[str, Any]) -> dict[str, Any]:
    pre, terminal, delta = record["pre_cast"], record["terminal"], record["delta"]
    return {
        "case_id": record["case_id"], "mode": record["mode"], "dead_zone_deg": record["dead_zone_deg"], "classification": record["classification"],
        "pelvis_pre_mae_deg": pre["pelvis"]["error_deg"]["mean"], "pelvis_terminal_mae_deg": terminal["pelvis"]["error_deg"]["mean"], "pelvis_delta_mae_deg": delta["pelvis_mae_deg"],
        "left_heel_slip_pre_p95_mm_per_frame": pre["feet"]["left"]["heel_slip_p95_mm_per_frame"]["p95"], "left_heel_slip_terminal_p95_mm_per_frame": terminal["feet"]["left"]["heel_slip_p95_mm_per_frame"]["p95"], "left_heel_slip_delta_p95_mm_per_frame": delta["left_heel_slip_p95_mm_per_frame_delta"],
        "left_toe_slip_pre_p95_mm_per_frame": pre["feet"]["left"]["toe_slip_p95_mm_per_frame"]["p95"], "left_toe_slip_terminal_p95_mm_per_frame": terminal["feet"]["left"]["toe_slip_p95_mm_per_frame"]["p95"], "left_toe_slip_delta_p95_mm_per_frame": delta["left_toe_slip_p95_mm_per_frame_delta"],
        "left_penetration_pre_p95_mm": pre["feet"]["left"]["penetration_p95_mm"]["p95"], "left_penetration_terminal_p95_mm": terminal["feet"]["left"]["penetration_p95_mm"]["p95"], "left_penetration_delta_p95_mm": delta["left_penetration_p95_mm_delta"],
        "terminal_root_delta_p95_mm": terminal["root"]["terminal_increment"]["xyz_m"]["norm"]["p95"], "terminal_root_delta_max_mm": terminal["root"]["terminal_increment"]["xyz_m"]["norm"]["max"], "overall_root_m0_p95_mm": terminal["root"]["overall_from_m0"]["xyz_m"]["norm"]["p95"], "aligned_root_shape_p95_mm": terminal["root"]["aligned_shape_deviation"]["xyz_m"]["norm"]["p95"],
        "joint_acceleration_pre_p95_mm": pre["temporal"]["mean_joint_acceleration_p95_mm"]["p95"], "joint_acceleration_terminal_p95_mm": terminal["temporal"]["mean_joint_acceleration_p95_mm"]["p95"], "joint_acceleration_delta_p95_mm": delta["mean_joint_acceleration_p95_mm"], "joint_jerk_delta_p95_mm": delta["mean_joint_jerk_p95_mm"], "joint_mpjpe_delta_p95_mm": delta["joint_mpjpe_p95_mm"],
        "right_position_status": terminal["feet"]["right"]["evidence"]["position_status"], "right_velocity_status": terminal["feet"]["right"]["evidence"]["velocity_status"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_per_frame(path: Path, record: Mapping[str, Any]) -> None:
    pre, terminal = record["pre_cast"], record["terminal"]
    columns: dict[str, list[Any]] = {
        "frame": list(range(100)),
        "pelvis_error_pre_deg": pre["pelvis"]["per_frame_error_deg"],
        "pelvis_error_terminal_deg": terminal["pelvis"]["per_frame_error_deg"],
        "root_terminal_increment_norm_mm": [None] * 100,
        "joint_mpjpe_pre_mm": pre["whole_body"]["per_frame_joint_mpjpe_mm"],
        "joint_mpjpe_terminal_mm": terminal["whole_body"]["per_frame_joint_mpjpe_mm"],
    }
    for side in ("left", "right"):
        for marker in ("heel", "toe"):
            for idx, axis in enumerate(("x", "y", "z")):
                columns[f"{side}_{marker}_pre_{axis}_m"] = [float(value[idx]) for value in pre["feet"][side]["per_frame"][marker]]
                columns[f"{side}_{marker}_terminal_{axis}_m"] = [float(value[idx]) for value in terminal["feet"][side]["per_frame"][marker]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for index in range(100):
            writer.writerow({key: values[index] for key, values in columns.items()})


def _write_plots(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    path.mkdir(parents=True, exist_ok=True)
    mode_colors = {"dose_only": "#1b9e77", "position_only_medium": "#d95f02", "temporal_weak": "#7570b3"}
    for y_key, ylabel, filename in (("pelvis_terminal_mae_deg", "Terminal pelvis MAE (deg)", "dead_zone_vs_pelvis_residual.png"), ("left_heel_slip_delta_p95_mm_per_frame", "Heel slip change (mm/frame)", "dead_zone_vs_heel_slip.png"), ("left_penetration_delta_p95_mm", "Penetration change (mm)", "dead_zone_vs_penetration.png"), ("terminal_root_delta_p95_mm", "Terminal root edit P95 (mm)", "dead_zone_vs_terminal_root.png"), ("joint_acceleration_delta_p95_mm", "Joint acceleration change P95 (mm/frame²)", "dead_zone_vs_acceleration.png")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for mode, color in mode_colors.items():
            subset = [row for row in rows if row["mode"] == mode]
            ax.plot([row["dead_zone_deg"] for row in subset], [row[y_key] for row in subset], marker="o", label=mode, color=color)
        ax.set_xlabel("Terminal dead-zone (deg)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path / filename, dpi=160)
        plt.close(fig)


def evaluate_all(*, v04_root: Path, protocol_root: Path, summary_path: Path, output: Path, device_name: str, plots: bool = True, run_2x2: bool = True) -> dict[str, Any]:
    protocol, mean, std, m0_batch, valid_batch, patches, sides = _load_frozen(protocol_root)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_rows = summary.get("rows", [])
    selected = [row for row in source_rows if float(row.get("dose")) == 2.0 and str(row.get("mode")) in MODES]
    if len(selected) != 3:
        raise ValueError("v0.4 summary must contain exactly the three sample94 +2 degree modes")
    valid = valid_batch[0]
    if valid.numel() != 100 or not bool(valid.all()):
        raise ValueError("sample94 requires 100 valid frames")
    m0 = m0_batch[0]
    m0_norm = (m0 - mean.view(1, -1)) / std.view(1, -1)
    device = torch.device(device_name)
    from smplx import SMPLX
    model_path = protocol["inputs"]["smplx_model"]["path"]
    model = SMPLX(model_path=model_path, gender="neutral", num_betas=10, batch_size=100, use_pca=False).to(device)
    m0_joints, m0_vertices = _load_vertices(model, m0, device)
    vertical_axis = int(protocol.get("ground_axis_index", 2))
    output.mkdir(parents=True, exist_ok=True)
    hashes = {str(path): _sha256(path) for path in (protocol_root / "protocol.json", protocol_root / "valid_mask.pt", protocol_root / "m0_physical.pt", protocol_root / "foot_patches.json", summary_path)}
    records: list[dict[str, Any]] = []
    for source in sorted(selected, key=lambda item: str(item["mode"])):
        mode = str(source["mode"])
        run_root = _resolve_run_root(v04_root, source["run_root"])
        endpoint_path = run_root / ENDPOINT_FILE
        payload = torch.load(endpoint_path, map_location="cpu", weights_only=True)
        pre_norm = _as_batch(payload["official_pre_cast_norm"], "official_pre_cast_norm")
        pre = _load_motion(pre_norm, mean, std, valid_batch, "official_pre_cast")
        pre_motion = pre[0]
        for dead_zone in DEAD_ZONES:
            config = ProjectorConfig(protocol=SOURCE_PROTOCOL, projection_scope="full_sequence", acceptance_mode="dose_first", contact_mode=mode, contact_position_weight=MODES[mode][0], contact_velocity_weight=MODES[mode][1], artifact_dir=None)
            projector = PelvisContactFlowProjector.from_frozen_protocol(protocol_root=protocol_root, sample_id="94", side=None, baseline_motion_norm=m0_norm, valid_mask=valid_batch, motion_mean=mean, motion_std=std, target_dose=2.0, config=config, device=device)
            result = projector.project_clean_endpoint(pre_norm.to(device), projector.baseline_motion, projector.contact_data, 2.0, {"valid_mask": valid_batch.to(device), "terminal_dead_zone_deg": dead_zone, "terminal_pelvis_enabled": True, "terminal_contact_enabled": True})
            endpoint_norm = result.projected_clean_motion.detach().cpu()
            endpoint = _load_motion(endpoint_norm, mean, std, valid_batch, "dead_zone_endpoint")[0]
            pre_record = _endpoint_record(m0, pre_motion, pre_motion, valid, 2.0, model, device, patches, sides, m0_joints, m0_vertices, vertical_axis)
            terminal_record = _endpoint_record(m0, endpoint, pre_motion, valid, 2.0, model, device, patches, sides, m0_joints, m0_vertices, vertical_axis)
            delta = _metric_delta(pre_record, terminal_record)
            classification, decision = _classification(pre_record, terminal_record, delta, dead_zone)
            case_id = f"{mode}_plus2deg_deadzone_{dead_zone:g}deg"
            record = {"case_id": case_id, "mode": mode, "dose_deg": 2.0, "dead_zone_deg": dead_zone, "source_endpoint_sha256": _sha256(endpoint_path), "pre_cast": pre_record, "terminal": terminal_record, "delta": delta, "classification": classification, "decision": decision, "pelvis_active_frame_count": int(sum(bool(x) for x in result.records[0].get("pelvis_active_frame_indices", []))), "terminal_result": result.diagnostics(), "endpoint_sha256": None}
            endpoint_out = output / "endpoints" / mode / f"dead_zone_{dead_zone:g}.pt"
            endpoint_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(endpoint_norm, endpoint_out)
            record["endpoint_sha256"] = _sha256(endpoint_out)
            records.append(record)
            _write_per_frame(output / "per_case" / case_id / "per_frame.csv", record)
    flat_rows = [_flat_row(record) for record in records]
    payload_out: dict[str, Any] = {"protocol": ANALYSIS_PROTOCOL, "source_generation_protocol": SOURCE_PROTOCOL, "scope": {"sample_id": "94", "seed": 0, "dose_deg": 2.0, "frame_count": 100, "case_count": len(records), "regeneration": False, "sampler_invoked": False}, "dead_zones_deg": list(DEAD_ZONES), "modes": {key: {"contact_position_weight": value[0], "contact_velocity_weight": value[1]} for key, value in MODES.items()}, "ground_axis_index": vertical_axis, "ground_axis_source": "frozen_protocol_or_v0_4_z_default", "source_hashes": hashes, "rows": records, "flat_rows": flat_rows}
    write_strict_json(output / "terminal_dead_zone_ablation.json", payload_out)
    _write_csv(output / "terminal_dead_zone_ablation.csv", flat_rows)
    counts: dict[str, int] = {}
    for row in flat_rows:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    lines = ["# v0.4.1 终端死区离线消融", "", f"协议：`{ANALYSIS_PROTOCOL}`", "输入为现有 sample94、seed0、+2° 投影前端点；未调用采样器。", "", "## 分类", "", "| 状态 | 数量 |", "|---|---:|"]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(counts.items()))
    lines.extend(["", "## 判定说明", "", "死区按模式独立选择；骨盆精度门为 MAE≤1°、P95≤2°，终端根平移 P95≤1 mm 且最大值≤10 mm。`TERMINAL_STAGE_SAFE` 不等于整体动作通过；整体根轨迹仍单独报告相对 M0 的原始偏离和去全局平移后的形变。", "", "右脚平足证据不足时保持 `NOT_EVALUABLE`，不升级为 PASS。"])
    (output / "terminal_dead_zone_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if plots:
        _write_plots(output / "figures", flat_rows)
    if run_2x2:
        _run_2x2_if_needed(output, payload_out, v04_root, protocol_root, summary_path, model, device, m0, mean, std, valid_batch, patches, sides, m0_joints, m0_vertices, vertical_axis)
    return payload_out


def _run_2x2_if_needed(output: Path, payload: Mapping[str, Any], v04_root: Path, protocol_root: Path, summary_path: Path, model: Any, device: torch.device, m0: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, valid_batch: torch.Tensor, patches: Mapping[str, Any], sides: Mapping[str, Any], m0_joints: torch.Tensor, m0_vertices: torch.Tensor, vertical_axis: int) -> None:
    candidates = [row for row in payload["flat_rows"] if row["mode"] == "temporal_weak" and float(row["dead_zone_deg"]) == 0.5]
    if not candidates or (candidates[0]["terminal_root_delta_p95_mm"] is not None and candidates[0]["terminal_root_delta_p95_mm"] <= 10.0):
        return
    source = next(row for row in json.loads(summary_path.read_text(encoding="utf-8")).get("rows", []) if float(row.get("dose")) == 2.0 and str(row.get("mode")) == "temporal_weak")
    endpoint_path = _resolve_run_root(v04_root, source["run_root"]) / ENDPOINT_FILE
    pre_norm = _as_batch(torch.load(endpoint_path, map_location="cpu", weights_only=True)["official_pre_cast_norm"], "official_pre_cast_norm")
    m0_norm = (m0 - mean.view(1, -1)) / std.view(1, -1)
    rows: list[dict[str, Any]] = []
    for pelvis_enabled in (False, True):
        for contact_enabled in (False, True):
            config = ProjectorConfig(protocol=SOURCE_PROTOCOL, projection_scope="full_sequence", acceptance_mode="dose_first", contact_mode="temporal_weak", contact_position_weight=1.0e4, contact_velocity_weight=1.0e4)
            projector = PelvisContactFlowProjector.from_frozen_protocol(protocol_root=protocol_root, sample_id="94", side=None, baseline_motion_norm=m0_norm, valid_mask=valid_batch, motion_mean=mean, motion_std=std, target_dose=2.0, config=config, device=device)
            result = projector.project_clean_endpoint(pre_norm.to(device), projector.baseline_motion, projector.contact_data, 2.0, {"valid_mask": valid_batch.to(device), "terminal_dead_zone_deg": 0.0, "terminal_pelvis_enabled": pelvis_enabled, "terminal_contact_enabled": contact_enabled})
            endpoint = _load_motion(result.projected_clean_motion.detach().cpu(), mean, std, valid_batch, "2x2_endpoint")[0]
            pre = _load_motion(pre_norm, mean, std, valid_batch, "2x2_pre")[0]
            record = _endpoint_record(m0[0], endpoint, pre, valid_batch[0], 2.0, model, device, patches, sides, m0_joints, m0_vertices, vertical_axis)
            rows.append({"pelvis_terminal_enabled": pelvis_enabled, "contact_terminal_enabled": contact_enabled, "root_terminal_p95_mm": record["root"]["terminal_increment"]["xyz_m"]["norm"]["p95"], "root_terminal_max_mm": record["root"]["terminal_increment"]["xyz_m"]["norm"]["max"], "pelvis_mae_deg": record["pelvis"]["error_deg"]["mean"], "left_heel_slip_p95_mm_per_frame": record["feet"]["left"]["heel_slip_p95_mm_per_frame"]["p95"], "left_penetration_p95_mm": record["feet"]["left"]["penetration_p95_mm"]["p95"]})
    write_strict_json(output / "pelvis_contact_2x2.json", {"protocol": ANALYSIS_PROTOCOL, "condition": "temporal_weak_plus2deg", "rows": rows, "effects": "pelvis/contact main effects and interaction are computed from the four rows; metrics retain their native units."})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v04-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-2x2", action="store_true")
    args = parser.parse_args()
    summary = args.summary or (args.v04_root / "v0_4_ablation_summary.json")
    result = evaluate_all(v04_root=args.v04_root, protocol_root=args.protocol_root, summary_path=summary, output=args.output, device_name=args.device, plots=not args.no_plots, run_2x2=not args.no_2x2)
    counts: dict[str, int] = {}
    for row in result["flat_rows"]:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    print(json.dumps({"protocol": ANALYSIS_PROTOCOL, "cases": len(result["rows"]), "classifications": counts, "output": str(args.output)}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
