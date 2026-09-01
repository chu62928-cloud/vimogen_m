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

from evaluation.pelvis_contact_compensation_v3 import evaluate_v3_pair, foot_patches, patch_centres
from evaluation.relative_root_trunk_v2_1 import direct_smpl_parameters
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--target-delta-deg", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol = json.loads((args.protocol_root / "protocol.json").read_text(encoding="utf-8"))
    patches = json.loads((args.protocol_root / "foot_patches.json").read_text(encoding="utf-8"))
    cases = [case for case in protocol["cases"] if str(case["sample_id"]) == str(args.sample_id)]
    if len(cases) != 1:
        raise ValueError(f"expected one frozen case, found {len(cases)}")
    m0_path = args.run_root / "m0_physical.pt"
    candidate_path = args.run_root / "selected_motion.pt"
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
        result["gates"].append({"name": "solver_feasible", "status": "FAIL", "threshold": True, "observed": run_record.get("feasible", False), "valid_count": 1, "reason": "selected_motion is the required M0 fallback; best infeasible candidate is retained separately"})
        result["status"] = "FAIL"
    output = args.output_dir or (args.run_root / "evaluation")
    output.mkdir(parents=True, exist_ok=True)
    # Keep the machine-readable gate file limited to statuses and observations;
    # human explanations remain in the run README/paired summary.
    gate_only = {
        "protocol": result["protocol"],
        "status": result["status"],
        "target_delta_deg": result["target_delta_deg"],
        "gates": [{key: value for key, value in gate.items() if key != "reason"} for gate in result["gates"]],
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
    (output / "README.md").write_text("\n".join(explanations) + "\n", encoding="utf-8")
    _write_json(output / "metrics.json", result)
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
    summary = {"sample_id": str(args.sample_id), "status": result["status"], "solver_feasible": run_record.get("feasible"), "pelvis_pitch_mae_deg": result["angle"]["mae_deg"], "pelvis_pitch_p95_deg": result["angle"]["p95_deg"], "q_rigid": result["q_rigid"], "output_dir": str(output), "run_sha256": _sha256(args.run_root / "run_record.json") if (args.run_root / "run_record.json").is_file() else None}
    _write_json(output / "paired_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    if result["status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
