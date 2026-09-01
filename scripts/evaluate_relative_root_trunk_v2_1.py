#!/usr/bin/env python3
"""Canonical paired evaluation for the independent root--trunk v2.1 protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from smplx import SMPLX


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.relative_root_forward_v1 import tail_safety_metrics
from evaluation.relative_root_trunk_v2_1 import (
    FAIL,
    NOT_EVALUABLE,
    PASS,
    direct_joints_from_motion,
    direct_smpl_parameters,
    evaluate_paired_foot_metrics,
    relative_angle_metrics,
    root_trunk_relative_angle_deg,
)
from motion_rep.motion_checker import _default_smpl_model_path
from motion_rep.phase1 import MOTION_LAYOUT, SMPLX_22_JOINT_INDEX, decode_rot6d_safe
from motion_rep.pose_authority import _geodesic, _root_forward, authority_project, consistency_report


PROTOCOL_NAME = "vimogen_relative_root_trunk_v2_1_minimal_source_noise"
TOE_GAP_THRESHOLD_M = 0.035


def _summary(values: torch.Tensor) -> dict[str, Any]:
    values = values[torch.isfinite(values)]
    if not values.numel():
        return {"mean": None, "p95": None, "max": None, "count": 0}
    return {
        "mean": float(values.mean().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
        "count": int(values.numel()),
    }


def _gate(name: str, observed: float | None, threshold: float, count: int, *, reason: str = "") -> dict[str, Any]:
    if observed is None or count <= 0:
        status = NOT_EVALUABLE
    elif not np.isfinite(observed):
        status = FAIL
    else:
        status = PASS if float(observed) <= threshold else FAIL
    return {"name": name, "status": status, "threshold": threshold, "observed": observed, "valid_count": count, "reason": reason}


def _combine(gates: list[dict[str, Any]]) -> str:
    if any(g["status"] == FAIL for g in gates):
        return FAIL
    if any(g["status"] == NOT_EVALUABLE for g in gates):
        return NOT_EVALUABLE
    return PASS


def _load_pair(run_root: Path, mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Path, Path]:
    m0_path = next(run_root.glob("m0_artifacts/batch_*/m0_official_norm_batch.pt"))
    candidate_path = next(run_root.glob("trainer/test_visualization/*/batch_*/mbench_raw_norm_batch.pt"))
    m0_norm = torch.load(m0_path, map_location="cpu", weights_only=True).float()
    archive = torch.load(candidate_path, map_location="cpu", weights_only=True)
    candidate_norm = archive["motion_norm"].float()
    candidate_mean = archive["motion_mean"].float()
    candidate_std = archive["motion_std"].float()
    m0_physical = m0_norm * std.view(1, 1, -1) + mean.view(1, 1, -1)
    candidate_physical = candidate_norm * candidate_std[:, None, :] + candidate_mean[:, None, :]
    valid_mask = archive["motion_mask"].bool()
    if m0_physical.shape != candidate_physical.shape or valid_mask.shape != m0_physical.shape[:2]:
        raise ValueError("M0, candidate, and valid mask are not paired")
    m0_authority = authority_project(m0_physical, valid_mask=valid_mask, output_dtype=torch.float32)
    candidate_authority = authority_project(candidate_physical, valid_mask=valid_mask, output_dtype=torch.float32)
    return m0_authority.physical_motion, candidate_authority.physical_motion, valid_mask, m0_path, candidate_path


@torch.inference_mode()
def _direct_mesh_vertices(motion: torch.Tensor, model: SMPLX, device: torch.device) -> torch.Tensor:
    """Render only direct body/root/translation streams; never integrate velocities."""

    parameters = direct_smpl_parameters(motion.unsqueeze(0).to(device))
    parameters = {key: value[0] for key, value in parameters.items()}
    return model(**parameters).vertices.detach().float().cpu()


def _centres(vertices: torch.Tensor, patch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return vertices[:, patch["heel"]].mean(1), vertices[:, patch["toe"]].mean(1)


def _foot_patches(model: SMPLX) -> dict[str, dict[str, torch.Tensor]]:
    template = model.v_template.detach().cpu()
    joints = model.J_regressor.detach().cpu() @ template
    dominant_joint = model.lbs_weights.detach().cpu().argmax(-1)
    result = {}
    for side, ankle_index, foot_index in (("left", 7, 10), ("right", 8, 11)):
        foot_vertices = torch.nonzero((dominant_joint == ankle_index) | (dominant_joint == foot_index), as_tuple=False).flatten()
        candidate = template[foot_vertices]
        sole = foot_vertices[candidate[:, 1] <= torch.quantile(candidate[:, 1], 0.25)]
        forward = joints[foot_index] - joints[ankle_index]
        forward[1] = 0.0
        forward = forward / torch.linalg.vector_norm(forward).clamp_min(1e-8)
        longitudinal = ((template[sole] - joints[ankle_index]) * forward).sum(-1)
        heel = sole[longitudinal <= torch.quantile(longitudinal, 0.25)]
        toe = sole[longitudinal >= torch.quantile(longitudinal, 0.75)]
        if heel.numel() < 5 or toe.numel() < 5:
            raise RuntimeError(f"insufficient {side} heel/toe vertices")
        result[side] = {"heel": heel, "toe": toe, "sole": sole}
    return result


def _foot_rows(m0_vertices: torch.Tensor, candidate_vertices: torch.Tensor, patches: dict[str, dict[str, torch.Tensor]]) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    rows: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    toe_gates = []
    for side in ("left", "right"):
        m0_heel, m0_toe = _centres(m0_vertices, patches[side])
        candidate_heel, candidate_toe = _centres(candidate_vertices, patches[side])
        paired = evaluate_paired_foot_metrics(m0_heel, m0_toe, candidate_heel, candidate_toe)
        m0_gap = m0_heel[:, 2] - m0_toe[:, 2]
        candidate_gap = candidate_heel[:, 2] - candidate_toe[:, 2]
        mask = torch.tensor(paired["contact_evidence"]["valid_masks"]["flat_contact"], dtype=torch.bool)
        if int(mask.sum()) < 3:
            toe_status = NOT_EVALUABLE
            m0_fraction = candidate_fraction = None
        else:
            m0_fraction = float((m0_gap[mask] >= TOE_GAP_THRESHOLD_M).float().mean())
            candidate_fraction = float((candidate_gap[mask] >= TOE_GAP_THRESHOLD_M).float().mean())
            toe_status = PASS if candidate_fraction <= m0_fraction + 0.05 else FAIL
        toe_gates.append(toe_status)
        rows[side] = {
            **paired,
            "toe_contact": {
                "status": toe_status,
                "threshold_gap_m": TOE_GAP_THRESHOLD_M,
                "baseline_fraction": m0_fraction,
                "candidate_fraction": candidate_fraction,
                "new_fraction_tolerance": 0.05,
            },
        }
        m0_centre = 0.5 * (m0_heel + m0_toe)
        candidate_centre = 0.5 * (candidate_heel + candidate_toe)
        m0_speed = torch.full((m0_centre.shape[0],), float("nan"))
        candidate_speed = torch.full_like(m0_speed, float("nan"))
        if m0_centre.shape[0] > 1:
            m0_speed[1:] = torch.linalg.vector_norm(m0_centre[1:, :2] - m0_centre[:-1, :2], dim=-1)
            candidate_speed[1:] = torch.linalg.vector_norm(candidate_centre[1:, :2] - candidate_centre[:-1, :2], dim=-1)
        general = torch.tensor(paired["contact_evidence"]["valid_masks"]["general_contact"], dtype=torch.bool)
        flat = torch.tensor(paired["contact_evidence"]["valid_masks"]["flat_contact"], dtype=torch.bool)
        for frame in range(m0_centre.shape[0]):
            csv_rows.append({
                "side": side,
                "frame": frame,
                "general_contact": bool(general[frame]),
                "flat_contact": bool(flat[frame]),
                "m0_heel_x": float(m0_heel[frame, 0]), "m0_heel_y": float(m0_heel[frame, 1]), "m0_heel_z": float(m0_heel[frame, 2]),
                "candidate_heel_x": float(candidate_heel[frame, 0]), "candidate_heel_y": float(candidate_heel[frame, 1]), "candidate_heel_z": float(candidate_heel[frame, 2]),
                "m0_toe_x": float(m0_toe[frame, 0]), "m0_toe_y": float(m0_toe[frame, 1]), "m0_toe_z": float(m0_toe[frame, 2]),
                "candidate_toe_x": float(candidate_toe[frame, 0]), "candidate_toe_y": float(candidate_toe[frame, 1]), "candidate_toe_z": float(candidate_toe[frame, 2]),
                "m0_speed_m_per_frame": None if not torch.isfinite(m0_speed[frame]) else float(m0_speed[frame]),
                "candidate_speed_m_per_frame": None if not torch.isfinite(candidate_speed[frame]) else float(candidate_speed[frame]),
                "m0_heel_minus_toe_gap_m": float(m0_gap[frame]), "candidate_heel_minus_toe_gap_m": float(candidate_gap[frame]),
            })
    toe_status = FAIL if FAIL in toe_gates else NOT_EVALUABLE if NOT_EVALUABLE in toe_gates else PASS
    return rows, csv_rows, toe_status


def evaluate(run_root: Path, output_dir: Path, device_name: str = "cuda:0") -> dict[str, Any]:
    mean = torch.from_numpy(np.load(ROOT / "data/meta_info/mean.npy")).float()
    std = torch.from_numpy(np.load(ROOT / "data/meta_info/std.npy")).float()
    device = torch.device(device_name)
    m0, candidate, valid_mask, m0_path, candidate_path = _load_pair(run_root, mean, std)
    b_root = decode_rot6d_safe(m0[..., MOTION_LAYOUT.root_rotation])
    c_root = decode_rot6d_safe(candidate[..., MOTION_LAYOUT.root_rotation])
    b_joints = direct_joints_from_motion(m0)
    c_joints = direct_joints_from_motion(candidate)
    angle = relative_angle_metrics(m0, candidate, valid_mask, 10.0)
    _, heading, _, _ = _root_forward(b_root)
    b_rel = root_trunk_relative_angle_deg(b_root, b_joints, m0_heading=heading)
    c_rel = root_trunk_relative_angle_deg(c_root, c_joints, m0_heading=heading)
    trunk_index = SMPLX_22_JOINT_INDEX["neck"]
    spine_index = SMPLX_22_JOINT_INDEX["spine1"]
    b_trunk = torch.nn.functional.normalize(b_joints[..., trunk_index, :] - b_joints[..., spine_index, :], dim=-1)
    c_trunk = torch.nn.functional.normalize(c_joints[..., trunk_index, :] - c_joints[..., spine_index, :], dim=-1)
    trunk_change = torch.acos((b_trunk * c_trunk).sum(-1).clamp(-1.0, 1.0)) * 180.0 / np.pi
    _, b_heading, _, _ = _root_forward(b_root)
    _, c_heading, _, _ = _root_forward(c_root)
    heading_change = torch.acos((b_heading * c_heading).sum(-1).clamp(-1.0, 1.0)) * 180.0 / np.pi
    root_change = _geodesic(c_root, b_root) * 180.0 / np.pi
    m = valid_mask.bool()
    trunk_summary = _summary(trunk_change[m])
    heading_summary = _summary(heading_change[m])
    root_summary = _summary(root_change[m])
    q_rigid = float(torch.median(trunk_change[m]) / torch.maximum(torch.median(root_change[m]), torch.tensor(0.5)))
    tail = tail_safety_metrics(m0, candidate, valid_mask)

    model = SMPLX(model_path=_default_smpl_model_path("smplx"), gender="neutral", num_betas=10, batch_size=100, use_pca=False).to(device)
    patches = _foot_patches(model)
    m0_vertices = _direct_mesh_vertices(m0[0], model, device)
    candidate_vertices = _direct_mesh_vertices(candidate[0], model, device)
    foot_rows, foot_csv, toe_status = _foot_rows(m0_vertices, candidate_vertices, patches)

    consistency = {
        "baseline": consistency_report(m0, valid_mask),
        "candidate": consistency_report(candidate, valid_mask),
    }
    consistency_pass = all(row["passed"] for records in consistency.values() for row in records)
    finite_pass = bool(torch.isfinite(m0).all() and torch.isfinite(candidate).all())
    source_delta_path = next(run_root.glob("source_noise_artifacts/batch_*/text/source_delta.pt"), None)
    summary_path = next(run_root.glob("source_noise_artifacts/batch_*/text/guidance_summary.json"), None)
    replay_pass = source_delta_path is not None and torch.isfinite(torch.load(source_delta_path, map_location="cpu", weights_only=True)).all().item()
    gates = [
        _gate("relative_angle_mae", angle["relative_angle_mae_deg"], 1.0, angle["valid_count"]),
        _gate("relative_angle_p95", angle["relative_angle_p95_deg"], 2.0, angle["valid_count"]),
        {"name": "dose_sign", "status": PASS if angle["dose_sign_correct"] else FAIL, "threshold": True, "observed": angle["dose_sign_correct"], "valid_count": angle["valid_count"], "reason": "signed relative-angle dose"},
        _gate("trunk_direction_p95", trunk_summary["p95"], 2.0, trunk_summary["count"]),
        _gate("q_rigid", q_rigid, 0.2, int(m.sum())),
        _gate("horizontal_heading_p95", heading_summary["p95"], 2.0, heading_summary["count"]),
        _gate("tail_extra_so3_jump", tail["per_sample"][0]["tail_extra_so3_jump_max_deg"], 2.0, tail["per_sample"][0]["tail_pair_count"]),
        _gate("tail_extra_pitch_step", tail["per_sample"][0]["tail_extra_pitch_step_max_deg"], 2.0, tail["per_sample"][0]["tail_pair_count"]),
        {"name": "toe_contact_regression", "status": toe_status, "threshold": "no new regression", "observed": toe_status == PASS, "valid_count": sum(1 for row in foot_rows.values() if row["toe_contact"]["status"] == PASS), "reason": "fixed M0 flat-contact frames"},
        {"name": "foot_sliding_lift_penetration", "status": _combine([{"status": value} for row in foot_rows.values() for value in row["statuses"].values()]), "threshold": "M0 + max(5%,1mm)", "observed": {side: row["statuses"] for side, row in foot_rows.items()}, "valid_count": sum(row["contact_evidence"]["contact_frames"] for row in foot_rows.values()), "reason": "paired M0 general-contact evidence"},
        {"name": "representation_consistency", "status": PASS if consistency_pass else FAIL, "threshold": "all residuals within authority thresholds", "observed": consistency_pass, "valid_count": 2, "reason": "direct pose authority and FK"},
        {"name": "finite_values", "status": PASS if finite_pass else FAIL, "threshold": True, "observed": finite_pass, "valid_count": int(m.sum()), "reason": "M0/candidate canonical tensors"},
        {"name": "final_noise_replay_record", "status": PASS if replay_pass else NOT_EVALUABLE, "threshold": "finite source delta and guidance summary", "observed": replay_pass, "valid_count": 1 if replay_pass else 0, "reason": str(summary_path) if summary_path else "missing source-noise record"},
    ]
    status = _combine(gates)

    output_dir.mkdir(parents=True, exist_ok=True)
    relative_csv = output_dir / "relative_angle_frames.csv"
    with relative_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "valid", "m0_relative_angle_deg", "candidate_relative_angle_deg", "target_relative_angle_deg", "absolute_error_deg", "trunk_change_deg", "heading_change_deg", "root_change_deg"])
        writer.writeheader()
        target = b_rel + 10.0
        for frame in range(m0.shape[1]):
            error_value = torch.remainder(c_rel[0, frame] - target[0, frame] + 180.0, 360.0) - 180.0
            writer.writerow({"frame": frame, "valid": bool(valid_mask[0, frame]), "m0_relative_angle_deg": float(b_rel[0, frame]), "candidate_relative_angle_deg": float(c_rel[0, frame]), "target_relative_angle_deg": float(target[0, frame]), "absolute_error_deg": abs(float(error_value)), "trunk_change_deg": float(trunk_change[0, frame]), "heading_change_deg": float(heading_change[0, frame]), "root_change_deg": float(root_change[0, frame])})
    foot_csv_path = output_dir / "foot_frames.csv"
    with foot_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(foot_csv[0].keys()))
        writer.writeheader(); writer.writerows(foot_csv)
    np.savez_compressed(output_dir / "relative_angle_curve.npz", m0=b_rel[0].cpu().numpy(), candidate=c_rel[0].cpu().numpy(), target=(b_rel[0] + 10).cpu().numpy(), valid=valid_mask[0].cpu().numpy())

    result = {
        "protocol": PROTOCOL_NAME,
        "run_root": str(run_root),
        "candidate_path": str(candidate_path),
        "m0_path": str(m0_path),
        "authority": {"source": "direct body_pose/root_rotation/root_translation -> FK -> finalizer", "velocity_integration": False},
        "relative_angle": angle,
        "whole_body": {"trunk_direction_change_deg": trunk_summary, "root_change_deg": root_summary, "horizontal_heading_change_deg": heading_summary, "q_rigid": q_rigid},
        "tail_safety": tail,
        "feet": foot_rows,
        "consistency": consistency,
        "m0_baseline": {
            "status": PASS if consistency_pass and finite_pass else FAIL,
            "authority": "direct body_pose/root_rotation/root_translation -> FK -> finalizer",
            "finite_values": finite_pass,
            "consistency": consistency["baseline"],
            "foot_baseline_metrics": {side: row["baseline"] for side, row in foot_rows.items()},
            "foot_contact_evidence": {side: row["contact_evidence"] for side, row in foot_rows.items()},
        },
        "gates": gates,
        "status": status,
        "outputs": {"relative_angle_csv": str(relative_csv), "foot_csv": str(foot_csv_path), "relative_angle_curve": str(output_dir / "relative_angle_curve.npz")},
    }
    (output_dir / "gates.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "naturalness.json").write_text(json.dumps({"protocol": PROTOCOL_NAME, "feet": foot_rows, "status": _combine([{"status": value} for row in foot_rows.values() for value in row["statuses"].values()])}, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "m0_baseline.json").write_text(json.dumps(result["m0_baseline"], indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = evaluate(args.run_root, args.output_dir, args.device)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir), "gates": result["gates"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
