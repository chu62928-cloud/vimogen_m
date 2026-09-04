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
from sampling.pelvis_contact_flow_projection_v0_1 import write_strict_json  # noqa: E402
from sampling.pelvis_contact_flow_projection_v0_2 import (  # noqa: E402
    PROTOCOL_NAME as TEMPORAL_CONTACT_PROTOCOL,
)


def _summary(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().float().reshape(-1)
    if not values.numel():
        return {"count": 0, "mean": None, "p95": None, "max": None}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


def _paired_status(candidate: dict[str, Any], baseline: dict[str, Any]) -> str:
    if candidate["count"] < 3 or baseline["count"] < 3:
        return "NOT_EVALUABLE"
    for key in ("mean", "p95"):
        tolerance = max(abs(float(baseline[key])) * 0.05, 0.001)
        if float(candidate[key]) > float(baseline[key]) + tolerance:
            return "FAIL"
    return "PASS"


def m0_pairing_eligible(status: str) -> bool:
    """Only the strict replay status can unlock a formal result."""

    return str(status) in {"PASS", "M0_PAIRING_PASS"}


def _frozen_window_foot(
    m0_vertices: torch.Tensor,
    candidate_vertices: torch.Tensor,
    patches: dict[str, dict[str, list[int]]],
    evidence: dict[str, Any],
    *,
    window_start: int,
    window_end: int,
    boundary_halo: int,
) -> dict[str, Any]:
    """Evaluate contact using frozen full-sequence masks and floor heights."""

    if m0_vertices.ndim != 3 or candidate_vertices.shape != m0_vertices.shape:
        raise ValueError("vertices must both have shape [T,V,3]")
    context_start = max(0, window_start - boundary_halo)
    context_end = min(m0_vertices.shape[0], window_end + boundary_halo)
    rows: dict[str, Any] = {}
    for side in ("left", "right"):
        patch = patches[side]
        indices_heel = torch.as_tensor(patch["heel"], dtype=torch.long)
        indices_toe = torch.as_tensor(patch["toe"], dtype=torch.long)
        m0_heel = m0_vertices[:, indices_heel].mean(1)
        m0_toe = m0_vertices[:, indices_toe].mean(1)
        c_heel = candidate_vertices[:, indices_heel].mean(1)
        c_toe = candidate_vertices[:, indices_toe].mean(1)
        side_evidence = evidence[side]["evidence"]
        masks = side_evidence["valid_masks"]
        general = torch.as_tensor(masks["general_contact"], dtype=torch.bool)
        pairs = torch.as_tensor(masks["continuous_contact_pair"], dtype=torch.bool)
        flat = torch.as_tensor(masks["flat_contact"], dtype=torch.bool)
        frame_mask = torch.zeros(m0_vertices.shape[0], dtype=torch.bool)
        frame_mask[window_start:window_end] = True
        pair_mask = torch.zeros(max(m0_vertices.shape[0] - 1, 0), dtype=torch.bool)
        pair_start = max(0, context_start)
        pair_end = min(m0_vertices.shape[0] - 1, context_end)
        if pair_end > pair_start:
            pair_mask[pair_start:pair_end] = True
        pair_mask &= pairs
        floor = float(side_evidence["floor_height_m"])
        m0_sole = torch.minimum(m0_heel[:, 2], m0_toe[:, 2])
        c_sole = torch.minimum(c_heel[:, 2], c_toe[:, 2])
        m0_center = 0.5 * (m0_heel + m0_toe)
        c_center = 0.5 * (c_heel + c_toe)
        m0_speed = torch.linalg.vector_norm(m0_center[1:, :2] - m0_center[:-1, :2], dim=-1)
        c_speed = torch.linalg.vector_norm(c_center[1:, :2] - c_center[:-1, :2], dim=-1)
        baseline = {
            "sliding_m_per_frame": _summary(m0_speed[pair_mask]),
            "lift_m": _summary((m0_sole - floor).clamp_min(0.0)[general & frame_mask]),
            "penetration_m": _summary((floor - m0_sole).clamp_min(0.0)[general & frame_mask]),
        }
        candidate = {
            "sliding_m_per_frame": _summary(c_speed[pair_mask]),
            "lift_m": _summary((c_sole - floor).clamp_min(0.0)[general & frame_mask]),
            "penetration_m": _summary((floor - c_sole).clamp_min(0.0)[general & frame_mask]),
        }
        statuses = {
            key: _paired_status(candidate[key], baseline[key]) for key in baseline
        }
        flat_window = flat & frame_mask
        m0_gap = m0_heel[:, 2] - m0_toe[:, 2]
        c_gap = c_heel[:, 2] - c_toe[:, 2]
        if int(flat_window.sum().item()) < 3:
            toe_status = "NOT_EVALUABLE"
            m0_fraction = c_fraction = None
        else:
            m0_fraction = float((m0_gap[flat_window] >= 0.035).float().mean().item())
            c_fraction = float((c_gap[flat_window] >= 0.035).float().mean().item())
            toe_status = "PASS" if c_fraction <= m0_fraction + 0.05 else "FAIL"
        all_statuses = list(statuses.values()) + [toe_status]
        status = "FAIL" if "FAIL" in all_statuses else (
            "NOT_EVALUABLE" if "NOT_EVALUABLE" in all_statuses else "PASS"
        )
        rows[side] = {
            "status": status,
            "baseline": baseline,
            "candidate": candidate,
            "statuses": statuses,
            "toe_contact": {
                "status": toe_status,
                "baseline_fraction": m0_fraction,
                "candidate_fraction": c_fraction,
            },
            "contact_evidence": side_evidence,
            "window": {
                "start": window_start,
                "end_exclusive": window_end,
                "context_start": context_start,
                "context_end_exclusive": context_end,
            },
        }
    return rows


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
    is_temporal = str(run_record.get("protocol")) == TEMPORAL_CONTACT_PROTOCOL
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
    replay_m0_vertices = _vertices(model, replay_m0, torch.device(device))
    candidate_vertices = _vertices(model, candidate, torch.device(device))
    patches = json.loads((protocol_root / "foot_patches.json").read_text(encoding="utf-8"))
    # Keep replay-paired diagnostics to expose numerical drift, while the
    # temporal protocol's primary window evaluation below uses the frozen M0
    # and its frozen evidence exactly as required by v0.2.
    paired = evaluate_v3_pair(
        replay_m0,
        candidate,
        valid,
        target_delta_deg=target,
        m0_vertices=replay_m0_vertices,
        candidate_vertices=candidate_vertices,
        patches=patches,
    )
    paired_frozen = evaluate_v3_pair(
        m0,
        candidate,
        valid,
        target_delta_deg=target,
        # The v3 evaluator consumes the per-sequence [T,V,3] mesh while
        # motion tensors remain batched [B,T,276].
        m0_vertices=m0_vertices,
        candidate_vertices=candidate_vertices,
        patches=patches,
    )
    window = case["sides"][side]["stable_window"]
    window_mask = torch.zeros_like(valid)
    window_mask[:, int(window["window_start"]) : int(window["window_end_exclusive"])] = True
    window_angle = _window_angle(replay_m0, candidate, window_mask, target)
    frozen_window_angle = _window_angle(m0, candidate, window_mask, target)
    # The v0.1 actuator is explicitly window-local.  Evaluate the primary
    # contact gate on that same window; retain the full-sequence paired result
    # as a separate regression diagnostic.
    window_start = int(window["window_start"])
    window_end = int(window["window_end_exclusive"])
    window_m0 = replay_m0[:, window_start:window_end]
    window_candidate = candidate[:, window_start:window_end]
    window_valid = torch.ones(
        (1, window_end - window_start), dtype=torch.bool
    )
    paired_window = evaluate_v3_pair(
        window_m0,
        window_candidate,
        window_valid,
        target_delta_deg=target,
        m0_vertices=replay_m0_vertices[window_start:window_end],
        candidate_vertices=candidate_vertices[window_start:window_end],
        patches=patches,
    )
    projection_log = json.loads(
        (run_root / "projection_artifacts" / "batch_000" / "sampling_projection_log.json").read_text(encoding="utf-8")
    )
    projection_case = projection_log.get("case", {})
    if not projection_case:
        records = projection_log.get("records", [])
        if records and isinstance(records[0], dict):
            projection_case = records[0].get("case", {})
    pairing_status = projection_case.get("m0_match_status", "UNKNOWN")
    if pairing_status == "UNKNOWN" and float((replay_m0 - m0).abs().max()) > 2.0e-3:
        pairing_status = "MISMATCH_ALLOWED"
    resolved_config = {}
    config_path = run_root / "resolved_config.yaml"
    if config_path.is_file():
        resolved_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    projection_config = resolved_config.get("pelvis_contact_projection", {})
    if is_temporal:
        frozen_feet = _frozen_window_foot(
            m0_vertices,
            candidate_vertices,
            patches,
            case["sides"],
            window_start=window_start,
            window_end=window_end,
            boundary_halo=int(
                projection_config.get("boundary_halo_frames", 1)
            )
            if projection_config else 1,
        )
        foot = frozen_feet.get(side, {})
    else:
        frozen_feet = None
        foot = paired_window["feet"].get(side, {})
    contact_status = foot.get("status", "NOT_EVALUABLE")
    primary_angle = frozen_window_angle if is_temporal else window_angle
    primary_pass = bool(
        primary_angle["pass"]
        and contact_status == "PASS"
        and m0_pairing_eligible(pairing_status)
        and paired["finite_values"]
        and foot.get("candidate", {}).get("penetration_m", {}).get("p95") is not None
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
            "status": str(pairing_status),
        },
        "primary_control": {
            "window": window,
            "window_angle": window_angle,
            "frozen_window_angle": frozen_window_angle,
            "full_sequence_angle": paired["angle"],
            "pass": primary_pass,
        },
        "contact": foot,
        "frozen_window_contact": frozen_feet,
        "full_sequence_evaluation": paired,
        "window_evaluation": paired_window,
        "frozen_baseline_evaluation": paired_frozen,
        "projection": projection_log,
        "interpretation": (
            "PRIMARY_PASS_WINDOW_CONTROL_AND_NO_CONTACT_REGRESSION"
            if primary_pass
            else (
                "INELIGIBLE_M0_MISMATCH"
                if not m0_pairing_eligible(pairing_status)
                else "PRIMARY_FAIL_OR_NOT_EVALUABLE"
            )
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
