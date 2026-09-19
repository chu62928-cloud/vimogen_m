"""External, non-optimizing contact audit for one v2 source-noise run.

The candidate is compared with its own M0 using the fixed M0 contact frames.
This module never contributes a gradient or selects an optimization candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from smplx import SMPLX

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_rep.motion_checker import _default_smpl_model_path
from scripts.diagnose_relative_root_forward_v1_3_foot_contact import (
    CONTACT_HEIGHT_M,
    CONTACT_SPEED_M,
    FLAT_GAP_THRESHOLD_M,
    REGRESSION_FRACTION,
    TOE_GAP_THRESHOLD_M,
    _contact_mask,
    _centres,
    _dose_metrics,
    _foot_patches,
    _mesh_vertices,
)


def _load_motion(path: Path, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True).float()
    return value * std.view(1, 1, -1) + mean.view(1, 1, -1)


def _summary(values: torch.Tensor) -> dict[str, float | None]:
    if values.numel() == 0:
        return {"mean": None, "p95": None, "max": None, "count": 0}
    return {
        "mean": float(values.mean()),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
        "count": int(values.numel()),
    }


def _foot_external_metrics(
    m0_vertices: torch.Tensor,
    candidate_vertices: torch.Tensor,
    patch: dict[str, torch.Tensor],
) -> dict:
    m0_heel, m0_toe = _centres(m0_vertices, patch)
    candidate_heel, candidate_toe = _centres(candidate_vertices, patch)
    flat, contact_info = _contact_mask(m0_heel, m0_toe)
    m0_sole = torch.minimum(m0_heel[:, 2], m0_toe[:, 2])
    candidate_sole = torch.minimum(candidate_heel[:, 2], candidate_toe[:, 2])
    m0_floor = torch.quantile(m0_sole, 0.05)
    m0_centre = 0.5 * (m0_heel + m0_toe)
    candidate_centre = 0.5 * (candidate_heel + candidate_toe)
    m0_speed = torch.zeros_like(m0_sole)
    m0_speed[:] = float("nan")
    candidate_speed = torch.zeros_like(candidate_sole)
    candidate_speed[:] = float("nan")
    m0_speed[1:] = torch.linalg.vector_norm(m0_centre[1:, :2] - m0_centre[:-1, :2], dim=-1)
    candidate_speed[1:] = torch.linalg.vector_norm(
        candidate_centre[1:, :2] - candidate_centre[:-1, :2], dim=-1
    )
    transition_mask = flat.clone()
    transition_mask[0] = False
    candidate_speed = candidate_speed[transition_mask]
    m0_speed = m0_speed[transition_mask]
    candidate_lift = (candidate_sole - m0_floor).clamp_min(0.0)[flat]
    candidate_penetration = (m0_floor - candidate_sole).clamp_min(0.0)[flat]
    baseline_lift = (m0_sole - m0_floor).clamp_min(0.0)[flat]
    baseline_penetration = (m0_floor - m0_sole).clamp_min(0.0)[flat]
    return {
        "contact": contact_info,
        "m0_floor_height_m": float(m0_floor),
        "candidate_sliding_m_per_frame": _summary(candidate_speed),
        "m0_sliding_m_per_frame": _summary(m0_speed),
        "candidate_lift_m": _summary(candidate_lift),
        "m0_lift_m": _summary(baseline_lift),
        "candidate_penetration_m": _summary(candidate_penetration),
        "m0_penetration_m": _summary(baseline_penetration),
        "evidence": {
            "height_minimum_frames": 3,
            "sliding_minimum_continuous_pairs": 3,
            "height_status": "PASS" if int(flat.sum()) >= 3 else "NOT_EVALUABLE",
            "sliding_status": "PASS" if int(transition_mask.sum()) >= 3 else "NOT_EVALUABLE",
        },
    }


def _allowed_increase(baseline: float | None) -> float:
    if baseline is None:
        return 0.001
    return max(abs(float(baseline)) * 0.05, 0.001)


def _metric_pass(candidate: dict, baseline: dict) -> bool:
    for key in ("mean", "p95"):
        value = candidate.get(key)
        reference = baseline.get(key)
        if value is None or reference is None:
            return False
        if float(value) > float(reference) + _allowed_increase(reference):
            return False
    return True


def _metric_status(candidate: dict, baseline: dict, minimum_count: int = 3) -> str:
    if int(candidate.get("count", 0)) < minimum_count or int(baseline.get("count", 0)) < minimum_count:
        return "NOT_EVALUABLE"
    return "PASS" if _metric_pass(candidate, baseline) else "FAIL"


def evaluate(run_root: Path, device: str = "cuda:0") -> dict:
    mean = torch.from_numpy(np.load(ROOT / "data/meta_info/mean.npy")).float()
    std = torch.from_numpy(np.load(ROOT / "data/meta_info/std.npy")).float()
    m0_path = next(run_root.glob("m0_artifacts/batch_*/m0_official_norm_batch.pt"))
    archive_path = next(run_root.glob("trainer/test_visualization/*/batch_*/mbench_raw_norm_batch.pt"))
    m0 = _load_motion(m0_path, mean, std)
    archive = torch.load(archive_path, map_location="cpu", weights_only=True)
    candidate = archive["motion_norm"].float() * archive["motion_std"].float()[:, None, :] + archive["motion_mean"].float()[:, None, :]
    mask = archive["motion_mask"].bool()
    model = SMPLX(
        model_path=_default_smpl_model_path("smplx"), gender="neutral", num_betas=10,
        batch_size=100, use_pca=False,
    ).to(device)
    patches = _foot_patches(model)
    m0_vertices = _mesh_vertices(m0[0], model, torch.device(device))
    candidate_vertices = _mesh_vertices(candidate[0], model, torch.device(device))
    dose_rows, toe_regression = _dose_metrics(
        m0[0], candidate[0], m0_vertices, candidate_vertices, patches, 10
    )
    rows = []
    for dose_row in dose_rows:
        side = dose_row["side"]
        rows.append({
            **dose_row,
            "external": _foot_external_metrics(
                m0_vertices, candidate_vertices, patches[side]
            ),
        })
    sliding_pass = all(
        _metric_pass(row["external"]["candidate_sliding_m_per_frame"], row["external"]["m0_sliding_m_per_frame"])
        for row in rows
    )
    lift_pass = all(
        _metric_pass(row["external"]["candidate_lift_m"], row["external"]["m0_lift_m"])
        for row in rows
    )
    penetration_pass = all(
        _metric_pass(row["external"]["candidate_penetration_m"], row["external"]["m0_penetration_m"])
        for row in rows
    )
    sliding_statuses = [
        _metric_status(row["external"]["candidate_sliding_m_per_frame"], row["external"]["m0_sliding_m_per_frame"])
        for row in rows
    ]
    lift_statuses = [
        _metric_status(row["external"]["candidate_lift_m"], row["external"]["m0_lift_m"])
        for row in rows
    ]
    penetration_statuses = [
        _metric_status(row["external"]["candidate_penetration_m"], row["external"]["m0_penetration_m"])
        for row in rows
    ]
    naturalness_status = "PASS"
    all_statuses = sliding_statuses + lift_statuses + penetration_statuses
    if any(status == "FAIL" for status in all_statuses):
        naturalness_status = "FAIL"
    elif any(status == "NOT_EVALUABLE" for status in all_statuses):
        naturalness_status = "NOT_EVALUABLE"
    result = {
        "protocol": "vimogen_relative_root_forward_v2_minimal_source_noise",
        "run_root": str(run_root),
        "candidate_path": str(archive_path),
        "m0_path": str(m0_path),
        "fixed_thresholds": {"allowed_relative_increase": 0.05, "absolute_tolerance_m": 0.001, "toe_regression_fraction": REGRESSION_FRACTION, "toe_gap_m": TOE_GAP_THRESHOLD_M, "flat_gap_m": FLAT_GAP_THRESHOLD_M, "contact_height_m": CONTACT_HEIGHT_M, "contact_speed_m_per_frame": CONTACT_SPEED_M},
        "rows": rows,
        "naturalness_gate": {
            "toe_contact_regression": "PASS" if not toe_regression else "FAIL",
            "sliding": sliding_statuses,
            "lift": lift_statuses,
            "penetration": penetration_statuses,
            "status": "FAIL" if toe_regression else naturalness_status,
            "passed": not toe_regression and naturalness_status == "PASS",
        },
    }
    output = run_root / "source_noise_naturalness_evaluation.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["output"] = str(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = evaluate(args.run_root, args.device)
    print(json.dumps({"output": result["output"], "naturalness_gate": result["naturalness_gate"]}, ensure_ascii=False))
    if not result["naturalness_gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
