#!/usr/bin/env python3
"""Evaluate a v3.1/v3.2 run against its frozen M0 protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from smplx import SMPLX

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pelvis_contact_compensation_v3 import evaluate_v3_pair, temporal_naturalness_metrics
from evaluation.relative_root_trunk_v2_1 import direct_joints_from_motion, direct_smpl_parameters
from motion_rep.smplx_utils import default_smpl_model_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")


def _vertices(motion: torch.Tensor, model: SMPLX, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        params = direct_smpl_parameters(motion.unsqueeze(0).to(device))
        params = {key: value[0] for key, value in params.items()}
        return model(**params, return_verts=True).vertices.detach().cpu()


def _naturalness_rows(result: dict[str, Any], temporal: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, record in temporal.items():
        if name == "root_path_length":
            rows.append(
                {
                    "group": "temporal",
                    "metric": name,
                    "unit": record["unit"],
                    "m0_mean": record["m0"],
                    "m0_p95": None,
                    "candidate_mean": record["candidate"],
                    "candidate_p95": None,
                    "absolute_delta_mean": abs(record["candidate"] - record["m0"]),
                    "absolute_delta_p95": None,
                    "status": "REPORT_ONLY",
                    "evidence_count": None,
                }
            )
            continue
        rows.append(
            {
                "group": "temporal",
                "metric": name,
                "unit": record["unit"],
                "m0_mean": record["m0"]["mean"],
                "m0_p95": record["m0"]["p95"],
                "candidate_mean": record["candidate"]["mean"],
                "candidate_p95": record["candidate"]["p95"],
                "absolute_delta_mean": record["absolute_delta"]["mean"],
                "absolute_delta_p95": record["absolute_delta"]["p95"],
                "status": "REPORT_ONLY",
                "evidence_count": record["candidate"]["count"],
            }
        )
    for side, foot in result.get("feet", {}).items():
        for metric, status in foot["statuses"].items():
            baseline = foot["baseline"][metric]
            candidate = foot["candidate"][metric]
            rows.append(
                {
                    "group": f"{side}_foot",
                    "metric": metric,
                    "unit": "m/frame" if metric == "sliding_m_per_frame" else "m",
                    "m0_mean": baseline["mean"],
                    "m0_p95": baseline["p95"],
                    "candidate_mean": candidate["mean"],
                    "candidate_p95": candidate["p95"],
                    "absolute_delta_mean": None if baseline["mean"] is None or candidate["mean"] is None else abs(candidate["mean"] - baseline["mean"]),
                    "absolute_delta_p95": None if baseline["p95"] is None or candidate["p95"] is None else abs(candidate["p95"] - baseline["p95"]),
                    "status": status,
                    "evidence_count": candidate["count"],
                }
            )
        toe = foot["toe_contact"]
        rows.append(
            {
                "group": f"{side}_foot",
                "metric": "toe_contact_fraction",
                "unit": "fraction",
                "m0_mean": toe["baseline_fraction"],
                "m0_p95": None,
                "candidate_mean": toe["candidate_fraction"],
                "candidate_p95": None,
                "absolute_delta_mean": None if toe["baseline_fraction"] is None or toe["candidate_fraction"] is None else abs(toe["candidate_fraction"] - toe["baseline_fraction"]),
                "absolute_delta_p95": None,
                "status": toe["status"],
                "evidence_count": foot["contact_evidence"]["flat_contact_frames"],
            }
        )
    gate_status = {gate["name"]: gate["status"] for gate in result["gates"]}
    posture = (
        ("trunk_direction_change", "deg", result["trunk_direction"], gate_status.get("trunk_direction_p95")),
        ("horizontal_heading_change", "deg", result["heading"], gate_status.get("horizontal_heading_p95")),
        ("pelvis_neck_change", "deg", result["uprightness"]["pelvis_neck"], gate_status.get("pelvis_neck_upright_p95")),
        ("pelvis_head_change", "deg", result["uprightness"]["pelvis_head"], gate_status.get("pelvis_head_upright_p95")),
        ("pelvis_support_drift", "m", result["uprightness"]["pelvis_support_drift"], gate_status.get("pelvis_support_drift_p95")),
    )
    for name, unit, record, status in posture:
        rows.append(
            {
                "group": "posture",
                "metric": name,
                "unit": unit,
                "m0_mean": 0.0,
                "m0_p95": 0.0,
                "candidate_mean": record["mean"],
                "candidate_p95": record["p95"],
                "absolute_delta_mean": record["mean"],
                "absolute_delta_p95": record["p95"],
                "status": status,
                "evidence_count": record["count"],
            }
        )
    return rows


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--target-delta-deg", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--candidate-path", type=Path, default=None)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol = json.loads((args.protocol_root / "protocol.json").read_text(encoding="utf-8"))
    patches = json.loads((args.protocol_root / "foot_patches.json").read_text(encoding="utf-8"))
    cases = [case for case in protocol["cases"] if str(case["sample_id"]) == str(args.sample_id)]
    if len(cases) != 1:
        raise ValueError(f"expected one frozen case, found {len(cases)}")
    m0_path = args.run_root / "m0_physical.pt"
    candidate_path = args.candidate_path or (args.run_root / "selected_motion.pt")
    if not m0_path.is_file() or not candidate_path.is_file():
        raise FileNotFoundError("run must contain m0_physical.pt and selected_motion.pt")
    m0 = torch.load(m0_path, map_location="cpu", weights_only=True).float()
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True).float()
    if m0.ndim == 2:
        m0 = m0.unsqueeze(0)
    if candidate.ndim == 2:
        candidate = candidate.unsqueeze(0)
    if m0.shape != candidate.shape or m0.shape[0] != 1:
        raise ValueError("M0 and selected candidate must be paired [1,T,276]")
    valid = torch.ones(m0.shape[:2], dtype=torch.bool)
    model_path = args.model_path or default_smpl_model_path("smplx", ROOT)
    device = torch.device(args.device)
    model = SMPLX(model_path=str(model_path), gender="neutral", num_betas=10, batch_size=int(m0.shape[1]), use_pca=False).to(device)
    m0_vertices = _vertices(m0[0], model, device)
    candidate_vertices = _vertices(candidate[0], model, device)
    result = evaluate_v3_pair(
        m0,
        candidate,
        valid,
        target_delta_deg=float(args.target_delta_deg),
        m0_vertices=m0_vertices,
        candidate_vertices=candidate_vertices,
        patches=patches,
        allow_missing_toe=str(args.sample_id) == "94",
    )
    run_record_path = args.run_root / "run_record.json"
    run_record = json.loads(run_record_path.read_text(encoding="utf-8")) if run_record_path.is_file() else {}
    if run_record.get("feasible") is False or run_record.get("fallback_is_m0") is True:
        reason = (
            "diagnostic candidate is explicitly ineligible; the official selected motion remains M0"
            if args.diagnostic_only
            else "selected_motion is the required M0 fallback; best infeasible candidate is retained separately"
        )
        result["gates"].append({"name": "solver_feasible", "status": "FAIL", "threshold": True, "observed": run_record.get("feasible", False), "valid_count": 1, "reason": reason})
        result["status"] = "FAIL"
    result["diagnostic_only"] = bool(args.diagnostic_only)
    result["eligible"] = not bool(args.diagnostic_only)
    temporal = temporal_naturalness_metrics(
        direct_joints_from_motion(m0),
        direct_joints_from_motion(candidate),
        valid,
    )
    naturalness_rows = _naturalness_rows(result, temporal)
    output = args.output_dir or (args.run_root / "evaluation")
    output.mkdir(parents=True, exist_ok=True)
    # Keep the machine-readable gate file limited to statuses and observations;
    # human explanations remain in the run README/paired summary.
    gate_only = {
        "protocol": result["protocol"],
        "status": result["status"],
        "target_delta_deg": result["target_delta_deg"],
        "gates": [{key: value for key, value in gate.items() if key != "reason"} for gate in result["gates"]],
        "diagnostic_only": bool(args.diagnostic_only),
        "eligible": not bool(args.diagnostic_only),
    }
    _write_json(output / "gates.json", gate_only)
    explanations = [
        f"# v3 evaluation: sample {args.sample_id}",
        "",
        f"Overall status: `{result['status']}`.",
        "",
        "Gate explanations:",
    ]
    explanations.extend(
        f"- `{gate['name']}`: {gate.get('reason') or 'no additional explanation'}"
        for gate in result["gates"]
    )
    explanations.extend(
        [
            "",
            "Naturalness comparison:",
            "",
            "| Group | Metric | Unit | M0 mean | M0 P95 | Candidate mean | Candidate P95 | abs(delta) P95 | Status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    explanations.extend(
        "| {group} | {metric} | {unit} | {m0_mean} | {m0_p95} | {candidate_mean} | {candidate_p95} | {absolute_delta_p95} | {status} |".format(
            **{key: _format_value(value) for key, value in row.items()}
        )
        for row in naturalness_rows
    )
    (output / "README.md").write_text("\n".join(explanations) + "\n", encoding="utf-8")
    _write_json(output / "metrics.json", result)
    _write_json(output / "naturalness.json", {"temporal": temporal, "rows": naturalness_rows})
    with (output / "naturalness_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(naturalness_rows[0].keys()))
        writer.writeheader()
        writer.writerows(naturalness_rows)
    rows = []
    from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe
    from evaluation.pelvis_contact_compensation_v3 import pelvis_pitch_delta_deg
    b_root = decode_rot6d_safe(m0[..., MOTION_LAYOUT.root_rotation])[0]
    c_root = decode_rot6d_safe(candidate[..., MOTION_LAYOUT.root_rotation])[0]
    actual = pelvis_pitch_delta_deg(b_root, c_root)
    for frame in range(m0.shape[1]):
        rows.append({"frame": frame, "valid": True, "m0_pitch_delta_deg": 0.0, "candidate_pitch_delta_deg": float(actual[frame]), "target_delta_deg": float(args.target_delta_deg), "absolute_error_deg": abs(float(actual[frame]) - float(args.target_delta_deg))})
    with (output / "per_frame_angles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"sample_id": str(args.sample_id), "status": result["status"], "solver_feasible": run_record.get("feasible"), "diagnostic_only": bool(args.diagnostic_only), "eligible": not bool(args.diagnostic_only), "candidate_path": str(candidate_path), "pelvis_pitch_mae_deg": result["angle"]["mae_deg"], "pelvis_pitch_p95_deg": result["angle"]["p95_deg"], "q_rigid": result["q_rigid"], "output_dir": str(output), "run_sha256": _sha256(args.run_root / "run_record.json") if (args.run_root / "run_record.json").is_file() else None}
    _write_json(output / "paired_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    if result["status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
