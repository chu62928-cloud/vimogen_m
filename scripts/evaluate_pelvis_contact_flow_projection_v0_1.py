#!/usr/bin/env python3
"""Evaluate one pelvis/contact projection pilot against its paired M0."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
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
from sampling.pelvis_contact_flow_projection_v0_1 import write_strict_json  # noqa: E402


def _vertices(model: SMPLX, motion: torch.Tensor, device: torch.device) -> torch.Tensor:
    params = direct_smpl_parameters(motion.to(device))
    params = {key: value[0] for key, value in params.items()}
    with torch.inference_mode():
        return model(**params, return_verts=True).vertices.detach().cpu()


def _window_angle(
    m0: torch.Tensor,
    candidate: torch.Tensor,
    mask: torch.Tensor,
    target: float,
) -> dict[str, Any]:
    m0_root = decode_rot6d_safe(m0[..., MOTION_LAYOUT.root_rotation])
    candidate_root = decode_rot6d_safe(candidate[..., MOTION_LAYOUT.root_rotation])
    actual = pelvis_pitch_delta_deg(m0_root, candidate_root)
    error = ((actual - float(target) + 180.0) % 360.0 - 180.0).abs()[mask]
    return {
        "mae_deg": float(error.mean().item()) if error.numel() else None,
        "p95_deg": float(torch.quantile(error, 0.95).item()) if error.numel() else None,
        "max_deg": float(error.max().item()) if error.numel() else None,
        "dose_mean_deg": float(actual[mask].mean().item()) if error.numel() else None,
        "valid_count": int(error.numel()),
        "pass": bool(error.numel() and float(error.mean()) <= 1.0),
    }


def evaluate(run_root: Path, protocol_root: Path, *, device: str = "cuda:0") -> dict[str, Any]:
    run_record = json.loads((run_root / "run_record.json").read_text(encoding="utf-8"))
    protocol = json.loads((protocol_root / "protocol.json").read_text(encoding="utf-8"))
    case = next(item for item in protocol["cases"] if str(item["sample_id"]) == "34122")
    side = str(run_record["side"])
    target = float(run_record["target_delta_deg"])
    mean = torch.from_numpy(np.load(protocol["inputs"]["mean"]["path"])).float()
    std = torch.from_numpy(np.load(protocol["inputs"]["std"]["path"])).float()
    replay_m0_norm = torch.load(
        run_root / "m0_artifacts" / "batch_000" / "m0_official_norm_batch.pt",
        map_location="cpu",
        weights_only=True,
    ).float()
    candidate_norm = torch.load(
        run_root / "projection_artifacts" / "batch_000" / "projected_g0_norm_batch.pt",
        map_location="cpu",
        weights_only=True,
    ).float()
    valid = torch.load(
        protocol_root / "valid_mask.pt", map_location="cpu", weights_only=True
    ).bool()[1:2]
    # The paired baseline is the frozen v3.0.1 physical M0.  The current
    # sampler replay is retained separately as an audit signal because the
    # server-side sampler has a known numerical drift from that endpoint.
    frozen_m0_physical = torch.load(
        protocol_root / "m0_physical.pt", map_location="cpu", weights_only=True
    ).float()[1:2]
    m0 = authority_project(frozen_m0_physical, valid_mask=valid).physical_motion
    replay_m0 = authority_project(
        replay_m0_norm * std.view(1, 1, -1) + mean.view(1, 1, -1),
        valid_mask=valid,
    ).physical_motion
    candidate = authority_project(candidate_norm * std.view(1, 1, -1) + mean.view(1, 1, -1), valid_mask=valid).physical_motion
    model = SMPLX(
        model_path=protocol["inputs"]["smplx_model"]["path"],
        gender="neutral",
        num_betas=10,
        batch_size=int(valid.shape[-1]),
        use_pca=False,
    ).to(device)
    m0_vertices = _vertices(model, m0, torch.device(device))
    candidate_vertices = _vertices(model, candidate, torch.device(device))
    patches = json.loads((protocol_root / "foot_patches.json").read_text(encoding="utf-8"))
    paired = evaluate_v3_pair(
        m0,
        candidate,
        valid,
        target_delta_deg=target,
        m0_vertices=m0_vertices.unsqueeze(0),
        candidate_vertices=candidate_vertices.unsqueeze(0),
        patches=patches,
    )
    window = case["sides"][side]["stable_window"]
    window_mask = torch.zeros_like(valid)
    window_mask[:, int(window["window_start"]) : int(window["window_end_exclusive"])] = True
    window_angle = _window_angle(m0, candidate, window_mask, target)
    foot = paired["feet"].get(side, {})
    contact_status = foot.get("status", "NOT_EVALUABLE")
    primary_pass = bool(
        window_angle["pass"]
        and contact_status != "FAIL"
        and paired["finite_values"]
        and foot.get("candidate", {}).get("penetration_m", {}).get("p95") is not None
    )
    projection_log = json.loads(
        (run_root / "projection_artifacts" / "batch_000" / "sampling_projection_log.json").read_text(encoding="utf-8")
    )
    result = {
        "protocol": run_record["protocol"],
        "sample_id": "34122",
        "seed": 0,
        "side": side,
        "metric": run_record["metric"],
        "target_delta_deg": target,
        "m0_pairing": {
            "primary_baseline": "frozen_v3_0_1_m0_physical",
            "replay_artifact": str(
                run_root / "m0_artifacts" / "batch_000" / "m0_official_norm_batch.pt"
            ),
            "replay_direct_max_abs": float(
                (replay_m0[..., MOTION_LAYOUT.body_pose] - m0[..., MOTION_LAYOUT.body_pose]).abs().max().item()
            ),
            "replay_full_max_abs": float((replay_m0 - m0).abs().max().item()),
            "status": str(projection_log.get("m0_match_status", "UNKNOWN")),
        },
        "primary_control": {
            "window": window,
            "window_angle": window_angle,
            "full_sequence_angle": paired["angle"],
            "pass": primary_pass,
        },
        "contact": foot,
        "full_sequence_evaluation": paired,
        "projection": projection_log,
        "interpretation": (
            "PRIMARY_PASS_WINDOW_CONTROL_AND_NO_CONTACT_REGRESSION"
            if primary_pass
            else "PRIMARY_FAIL_OR_NOT_EVALUABLE"
        ),
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
