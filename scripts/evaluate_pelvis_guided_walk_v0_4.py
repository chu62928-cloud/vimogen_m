#!/usr/bin/env python3
"""Full-sequence paired evaluation for the v0.4 sample94 ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml
from smplx import SMPLX

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pelvis_contact_compensation_v3 import (  # noqa: E402
    evaluate_v3_pair,
    patch_centres,
    pelvis_pitch_delta_deg,
)
from evaluation.relative_root_trunk_v2_1 import direct_smpl_parameters  # noqa: E402
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe  # noqa: E402
from motion_rep.pose_authority import authority_project  # noqa: E402
from sampling.pelvis_contact_flow_projection_v0_1 import (  # noqa: E402
    DOSE_FIRST_CONTACT_ABLATION_PROTOCOL,
    write_strict_json,
)
from scripts.evaluate_pelvis_contact_flow_projection_v0_1 import (  # noqa: E402
    _frozen_window_foot,
    _summary,
)


def _vertices(model: SMPLX, motion: torch.Tensor, device: torch.device) -> torch.Tensor:
    params = direct_smpl_parameters(motion.to(device))
    params = {key: value[0] for key, value in params.items()}
    with torch.inference_mode():
        return model(**params, return_verts=True).vertices.detach().cpu()


def _angle(m0: torch.Tensor, candidate: torch.Tensor, valid: torch.Tensor, target: float) -> dict[str, Any]:
    m0_root = decode_rot6d_safe(m0[..., MOTION_LAYOUT.root_rotation])
    candidate_root = decode_rot6d_safe(candidate[..., MOTION_LAYOUT.root_rotation])
    actual = pelvis_pitch_delta_deg(m0_root, candidate_root)
    actual_sequence = actual[0] if actual.ndim == 2 else actual
    error = ((actual_sequence - target + 180.0) % 360.0 - 180.0).abs()[valid]
    return {
        "dose_mean_deg": float(actual_sequence[valid].mean()) if error.numel() else None,
        "mae_deg": float(error.mean()) if error.numel() else None,
        "p95_deg": float(torch.quantile(error, 0.95)) if error.numel() else None,
        "max_deg": float(error.max()) if error.numel() else None,
        "direction_correct": bool((actual_sequence[valid] * target >= -1.0e-5).all()) if error.numel() else False,
        "valid_count": int(error.numel()),
        "pass": bool(error.numel() and float(error.mean()) <= 1.0 and float(torch.quantile(error, 0.95)) <= 2.0),
    }


def _position_residuals(
    m0_vertices: torch.Tensor,
    candidate_vertices: torch.Tensor,
    patches: dict[str, Any],
    evidence: dict[str, Any],
    valid: torch.Tensor,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side in ("left", "right"):
        heel_patch = torch.as_tensor(patches[side]["heel"], dtype=torch.long)
        toe_patch = torch.as_tensor(patches[side]["toe"], dtype=torch.long)
        m0_heel = m0_vertices[:, heel_patch].mean(1)
        m0_toe = m0_vertices[:, toe_patch].mean(1)
        c_heel = candidate_vertices[:, heel_patch].mean(1)
        c_toe = candidate_vertices[:, toe_patch].mean(1)
        masks = evidence[side]["evidence"]["valid_masks"]
        flat = torch.as_tensor(masks["flat_contact"], dtype=torch.bool) & valid
        pairs = torch.as_tensor(masks["continuous_contact_pair"], dtype=torch.bool)
        pos_heel = torch.linalg.vector_norm(c_heel - m0_heel, dim=-1)[flat]
        pos_toe = torch.linalg.vector_norm(c_toe - m0_toe, dim=-1)[flat]
        vel_heel = torch.linalg.vector_norm((c_heel[1:] - c_heel[:-1]) - (m0_heel[1:] - m0_heel[:-1]), dim=-1)[pairs]
        vel_toe = torch.linalg.vector_norm((c_toe[1:] - c_toe[:-1]) - (m0_toe[1:] - m0_toe[:-1]), dim=-1)[pairs]
        result[side] = {
            "position": {"heel_m": _summary(pos_heel), "toe_m": _summary(pos_toe)},
            "velocity": {"heel_m_per_frame": _summary(vel_heel), "toe_m_per_frame": _summary(vel_toe)},
            "position_status": "PASS" if pos_heel.numel() >= 3 and pos_toe.numel() >= 3 and float(pos_heel.max()) <= 0.001 and float(pos_toe.max()) <= 0.001 else "NOT_EVALUABLE" if pos_heel.numel() < 3 or pos_toe.numel() < 3 else "FAIL",
            "velocity_status": "PASS" if vel_heel.numel() >= 3 and vel_toe.numel() >= 3 and float(vel_heel.max()) <= 0.001 and float(vel_toe.max()) <= 0.001 else "NOT_EVALUABLE" if vel_heel.numel() < 3 or vel_toe.numel() < 3 else "FAIL",
        }
    return result


def evaluate(run_root: Path, protocol_root: Path, *, device: str = "cuda:0") -> dict[str, Any]:
    run_record = json.loads((run_root / "run_record.json").read_text(encoding="utf-8"))
    protocol = json.loads((protocol_root / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("protocol") != DOSE_FIRST_CONTACT_ABLATION_PROTOCOL:
        raise ValueError("protocol-root is not v0.4")
    case = next(item for item in protocol["cases"] if str(item["sample_id"]) == "94")
    mean = torch.from_numpy(np.load(protocol["inputs"]["mean"]["path"])).float()
    std = torch.from_numpy(np.load(protocol["inputs"]["std"]["path"])).float()
    valid = torch.load(protocol_root / "valid_mask.pt", map_location="cpu", weights_only=True).bool()[0:1]
    m0_physical = torch.load(protocol_root / "m0_physical.pt", map_location="cpu", weights_only=True).float()[0:1]
    m0 = authority_project(m0_physical, valid_mask=valid).physical_motion
    candidate_norm = torch.load(run_root / "projection_artifacts" / "batch_000" / "projected_g0_norm_batch.pt", map_location="cpu", weights_only=True).float()
    candidate = authority_project(candidate_norm * std.view(1, 1, -1) + mean.view(1, 1, -1), valid_mask=valid).physical_motion
    model = SMPLX(model_path=protocol["inputs"]["smplx_model"]["path"], gender="neutral", num_betas=10, batch_size=int(valid.shape[-1]), use_pca=False).to(device)
    m0_vertices = _vertices(model, m0, torch.device(device))
    candidate_vertices = _vertices(model, candidate, torch.device(device))
    patches = json.loads((protocol_root / "foot_patches.json").read_text(encoding="utf-8"))
    evidence = case["sides"]
    paired = evaluate_v3_pair(m0, candidate, valid, target_delta_deg=float(run_record["target_delta_deg"]), m0_vertices=m0_vertices, candidate_vertices=candidate_vertices, patches=patches)
    target = float(run_record["target_delta_deg"])
    valid_frames = valid[0]
    angle = _angle(m0, candidate, valid_frames, target)
    foot = _frozen_window_foot(
        m0_vertices,
        candidate_vertices,
        patches,
        evidence,
        window_start=0,
        window_end=int(valid_frames.shape[0]),
        boundary_halo=0,
    )
    positions = _position_residuals(m0_vertices, candidate_vertices, patches, evidence, valid_frames)
    projection_log_path = run_root / "projection_artifacts" / "batch_000" / "sampling_projection_log.json"
    projection_log = json.loads(projection_log_path.read_text(encoding="utf-8")) if projection_log_path.is_file() else {}
    pairing_status = "M0_PAIRING_PASS" if run_record.get("m0_pairing_status") == "M0_PAIRING_PASS" else run_record.get("m0_pairing_status", "UNKNOWN")
    contact_statuses = [foot[side]["status"] for side in ("left", "right")]
    contact_gate = "FAIL" if "FAIL" in contact_statuses or any(positions[s]["position_status"] == "FAIL" or positions[s]["velocity_status"] == "FAIL" for s in positions) else "NOT_EVALUABLE" if "NOT_EVALUABLE" in contact_statuses or any(positions[s]["position_status"] == "NOT_EVALUABLE" or positions[s]["velocity_status"] == "NOT_EVALUABLE" for s in positions) else "PASS"
    config = yaml.safe_load((run_root / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
    finite = bool(torch.isfinite(candidate).all())
    trust_records = [record for group in projection_log.get("records", []) if isinstance(group, dict) for record in group.get("step_records", []) if isinstance(record, dict)]
    trust_violation = any(float(record.get("max_joint_increment_deg", 0.0)) > 5.0001 or float(record.get("delta_root_translation", 0.0)) > 0.0100001 for record in trust_records)
    status = "FORMAL_PASS" if pairing_status == "M0_PAIRING_PASS" and angle["pass"] and contact_gate == "PASS" and finite and not trust_violation else "NOT_EVALUABLE" if contact_gate == "NOT_EVALUABLE" else "DIAGNOSTIC_CONTACT_FAIL" if pairing_status == "M0_PAIRING_PASS" and angle["pass"] else "INELIGIBLE_M0_MISMATCH"
    terminal_path = run_root / "projection_artifacts" / "terminal_projection_endpoints.pt"
    terminal = {}
    if terminal_path.is_file():
        payload = torch.load(terminal_path, map_location="cpu", weights_only=True)
        diff = payload["terminal_difference_norm"].float()
        terminal = {"max_abs": float(diff.abs().max()), "rms": float(torch.sqrt(diff.square().mean())), "per_frame_rms": torch.sqrt(diff.square().mean(dim=-1)).reshape(-1).tolist()}
    result = {
        "protocol": DOSE_FIRST_CONTACT_ABLATION_PROTOCOL,
        "protocol_root": str(protocol_root),
        "m0_physical_sha256": hashlib.sha256(
            (protocol_root / "m0_physical.pt").read_bytes()
        ).hexdigest(),
        "sample_id": "94",
        "contact_mode": run_record.get("contact_mode", config.get("pelvis_contact_projection", {}).get("contact_mode")),
        "target_delta_deg": target,
        "status": status,
        "m0_pairing": {"status": pairing_status, "baseline_origin": "current_environment_refreeze", "legacy_v3_relation": "reference_only"},
        "dose_gate": angle,
        "contact_gate": {"status": contact_gate, "per_side": foot, "position_velocity": positions},
        "naturalness": {key: paired.get(key) for key in ("trunk", "pelvis_neck", "pelvis_head", "heading", "support_drift", "finite_values") if key in paired},
        "walk_metrics": {"valid_frames": int(valid_frames.sum()), "forward_distance_m": float((candidate[0, -1, MOTION_LAYOUT.root_translation.start] - candidate[0, 0, MOTION_LAYOUT.root_translation.start]).item()) if valid_frames.numel() else None},
        "trust_region_violation": trust_violation,
        "terminal_rebound": terminal,
        "projection": projection_log,
        "interpretation": "FORMAL_PASS" if status == "FORMAL_PASS" else "CONTACT_STRENGTH_ABLATION_RESULT",
    }
    write_strict_json(run_root / "evaluation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run_root, args.protocol_root, device=args.device), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
