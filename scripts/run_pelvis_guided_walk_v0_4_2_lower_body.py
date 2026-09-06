#!/usr/bin/env python3
"""Run the v0.4.2 terminal lower-body spatial-feasibility prototype."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe, encode_rot6d  # noqa: E402
from motion_rep.pose_authority import authority_project  # noqa: E402
from sampling.lower_body_terminal_compensation import (  # noqa: E402
    LOWER_BODY_DOF_PROTOCOL,
    LowerBodyDofMap,
    LowerBodySolverConfig,
    solve_lower_body_position,
)
from scripts.evaluate_pelvis_guided_walk_v0_4_1_terminal_dead_zone import (  # noqa: E402
    _endpoint_record,
    _finite,
    _load_frozen,
    _load_motion,
    _load_vertices,
    _metric_delta,
    _resolve_run_root,
    _as_batch,
)
from sampling.pelvis_contact_flow_projection_v0_1 import write_strict_json  # noqa: E402


def _dose_only_endpoint(pre: torch.Tensor, m0: torch.Tensor, valid: torch.Tensor, target: float, dead_zone: float) -> torch.Tensor:
    body = decode_rot6d_safe(pre[:, MOTION_LAYOUT.body_pose].reshape(pre.shape[0], 21, 6))
    source_root = decode_rot6d_safe(pre[:, MOTION_LAYOUT.root_rotation])
    m0_root = decode_rot6d_safe(m0[:, MOTION_LAYOUT.root_rotation])
    from sampling.terminal_projection_policy import TerminalProjectionPolicy, build_terminal_root_target
    target_root, _, _ = build_terminal_root_target(m0_root, source_root, target, valid_mask=valid, policy=TerminalProjectionPolicy(dead_zone_deg=dead_zone))
    rebuilt = pre.clone()
    rebuilt[:, MOTION_LAYOUT.body_pose] = encode_rot6d(body).reshape(pre.shape[0], 126)
    rebuilt[:, MOTION_LAYOUT.root_rotation] = encode_rot6d(target_root)
    return authority_project(rebuilt.unsqueeze(0), valid_mask=valid.unsqueeze(0), output_dtype=torch.float32).physical_motion[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v04-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dead-zone", type=float, default=0.5)
    parser.add_argument("--dose", type=float, default=2.0)
    parser.add_argument("--dof-map", choices=("anatomical", "full_so3_diagnostic"), default="anatomical")
    args = parser.parse_args()
    summary_path = args.summary or (args.v04_root / "v0_4_ablation_summary.json")
    protocol, mean, std, m0_batch, valid_batch, patches, sides = _load_frozen(args.protocol_root)
    source_rows = __import__("json").loads(summary_path.read_text(encoding="utf-8"))["rows"]
    source = next(row for row in source_rows if float(row["dose"]) == args.dose and str(row["mode"]) == "position_only_medium")
    endpoint_path = _resolve_run_root(args.v04_root, source["run_root"]) / "projection_artifacts" / "terminal_projection_endpoints.pt"
    pre_norm = _as_batch(torch.load(endpoint_path, map_location="cpu", weights_only=True)["official_pre_cast_norm"], "official_pre_cast_norm")
    pre = _load_motion(pre_norm, mean, std, valid_batch, "official_pre_cast")[0]
    m0 = m0_batch[0]
    valid = valid_batch[0]
    device = torch.device(args.device)
    from smplx import SMPLX
    model = SMPLX(model_path=protocol["inputs"]["smplx_model"]["path"], gender="neutral", num_betas=10, batch_size=int(valid.sum()), use_pca=False).to(device)
    dof_map = LowerBodyDofMap.default() if args.dof_map == "anatomical" else LowerBodyDofMap.full_so3_diagnostic()
    floor_heights = {side: float(sides[side]["evidence"]["floor_height_m"]) for side in ("left", "right")}
    solved = solve_lower_body_position(pre.to(device), m0.to(device), valid.to(device), args.dose, model=model, patches=patches, dead_zone_deg=args.dead_zone, dof_map=dof_map, config=LowerBodySolverConfig(), floor_heights=floor_heights)
    candidate = solved.projected_physical.detach().cpu()
    m0_joints, m0_vertices = _load_vertices(model, m0, device)
    pre_record = _endpoint_record(m0, pre, pre, valid, args.dose, model, device, patches, sides, m0_joints, m0_vertices, 2)
    candidate_record = _endpoint_record(m0, candidate, pre, valid, args.dose, model, device, patches, sides, m0_joints, m0_vertices, 2)
    position_delta = _metric_delta(pre_record, candidate_record)
    final_residuals = [float(item["final_foot_residual_mm"]) for item in solved.records]
    max_final_residual = max(final_residuals, default=0.0)
    position_pass = bool(max_final_residual <= 1.0 + 1.0e-6)
    trust_pass = bool(all(float(item["max_dof_increment_deg"]) <= 5.0 + 1.0e-6 for item in solved.records))
    no_new_penetration = True
    for side in ("left", "right"):
        before = pre_record["feet"][side]["penetration_p95_mm"]["p95"]
        after = candidate_record["feet"][side]["penetration_p95_mm"]["p95"]
        if before is not None and after is not None and after > before + 1.0e-6:
            no_new_penetration = False
    dose_only = _dose_only_endpoint(pre, m0, valid, args.dose, args.dead_zone)
    dose_record = _endpoint_record(m0, dose_only, pre, valid, args.dose, model, device, patches, sides, m0_joints, m0_vertices, 2)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.save(candidate, output / "position_only_lower_body_endpoint_physical.pt")
    torch.save(dose_only, output / "dose_only_locked_root_endpoint_physical.pt")
    write_strict_json(output / "lower_body_dof_map.json", dof_map.jsonable())
    write_strict_json(output / "lower_body_position_result.json", {"protocol": LOWER_BODY_DOF_PROTOCOL, "mode": "position_only_medium", "dof_map_kind": args.dof_map, "formal_candidate": args.dof_map == "anatomical", "dose_deg": args.dose, "dead_zone_deg": args.dead_zone, "pre_cast": pre_record, "candidate": candidate_record, "delta": position_delta, "dose_only_control": dose_record, "acceptance": {"position_residual_pass": position_pass, "max_final_foot_residual_mm": max_final_residual, "trust_region_pass": trust_pass, "no_new_penetration": no_new_penetration, "root_translation_p95_mm": 0.0, "status": "B1_PASS" if position_pass and trust_pass and no_new_penetration and solved.finite and solved.root_translation_locked else "B1_FAIL"}, "solver": solved.diagnostics(), "source_endpoint": str(endpoint_path)})
    print(__import__("json").dumps({"protocol": LOWER_BODY_DOF_PROTOCOL, "mode": "position_only_medium", "active_frames": int(solved.pelvis_active.sum()), "finite": solved.finite, "root_translation_locked": solved.root_translation_locked, "output": str(output)}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
