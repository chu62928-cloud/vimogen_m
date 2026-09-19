"""Diagnose heel-to-toe contact changes in v1.3 generated motions.

The detector fixes flat-foot contact frames from each seed's M0 mesh and then
checks whether G0 pivots onto the toe patch: the toe remains lower while the
heel-to-toe vertical gap grows beyond 25 mm.  Foot patches are derived once
from the neutral SMPL-X template using skinning weights, so no video pixels or
camera assumptions enter the metric.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
import sys

import numpy as np
import torch
from smplx import SMPLX

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_rep.motion_checker import _default_smpl_model_path
from motion_rep.phase1 import MOTION_LAYOUT, SMPLX_22_JOINT_INDEX, decode_rot6d_safe
from motion_rep.pose_authority import _geodesic
from motion_rep.retarget_motion import motion_rep_to_SMPL


PARAMETER_KEY = "pitch_1_pstep_6_heading_0.75_hstep_2_trunk_0.75_tstep_6_sigma_0.0662879_to_0.65"
DELTAS = (-10, -5, 5, 10)
TOE_GAP_THRESHOLD_M = 0.025
FLAT_GAP_THRESHOLD_M = 0.020
CONTACT_HEIGHT_M = 0.025
CONTACT_SPEED_M = 0.030
REGRESSION_FRACTION = 0.20


def _latest_attempt(config_dir: Path, delta: int) -> Path:
    sign = "+" if delta >= 0 else ""
    attempts = []
    for path in (config_dir / f"delta_{sign}{delta}deg").glob("attempt_*"):
        summary = path / "guided_artifacts" / "batch_000" / "guidance_summary.json"
        if summary.is_file():
            attempts.append((int(path.name.rsplit("_", 1)[-1]), path))
    if not attempts:
        raise FileNotFoundError(f"no completed delta {delta:+d} attempt under {config_dir}")
    return max(attempts)[1]


def _load_physical(path: Path, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True).float()
    if value.ndim != 3 or value.shape[-1] != 276:
        raise ValueError(f"{path} must be [B,T,276]")
    return value * std.view(1, 1, -1) + mean.view(1, 1, -1)


def _foot_patches(model: SMPLX) -> dict[str, dict[str, torch.Tensor]]:
    template = model.v_template.detach().cpu()
    joints = (model.J_regressor.detach().cpu() @ template)
    dominant_joint = model.lbs_weights.detach().cpu().argmax(-1)
    result = {}
    for side, ankle_index, foot_index in (("left", 7, 10), ("right", 8, 11)):
        foot_vertices = torch.nonzero(
            (dominant_joint == ankle_index) | (dominant_joint == foot_index), as_tuple=False
        ).flatten()
        candidate = template[foot_vertices]
        # SMPL-X neutral-template vertical is +Y; retain the bottom quartile.
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


@torch.inference_mode()
def _mesh_vertices(motion: torch.Tensor, model: SMPLX, device: torch.device) -> torch.Tensor:
    parameters, _ = motion_rep_to_SMPL(motion.to(device), recover_from_velocity=True)
    return model(**parameters).vertices.detach().float().cpu()


def _centres(vertices: torch.Tensor, patch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return vertices[:, patch["heel"]].mean(1), vertices[:, patch["toe"]].mean(1)


def _contact_mask(heel: torch.Tensor, toe: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    sole_height = torch.minimum(heel[:, 2], toe[:, 2])
    floor = torch.quantile(sole_height, 0.05)
    centre = 0.5 * (heel + toe)
    speed = torch.zeros_like(sole_height)
    speed[1:] = torch.linalg.vector_norm(centre[1:, :2] - centre[:-1, :2], dim=-1)
    gap = heel[:, 2] - toe[:, 2]
    contact = (sole_height <= floor + CONTACT_HEIGHT_M) & (speed <= CONTACT_SPEED_M)
    flat = contact & (gap.abs() <= FLAT_GAP_THRESHOLD_M)
    return flat, {
        "floor_height_m": float(floor),
        "contact_frames": int(contact.sum()),
        "flat_contact_frames": int(flat.sum()),
    }


def _summary(values: torch.Tensor) -> dict[str, float | None]:
    if values.numel() == 0:
        return {"mean": None, "median": None, "p95_abs": None, "max_abs": None}
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95_abs": float(torch.quantile(values.abs(), 0.95)),
        "max_abs": float(values.abs().max()),
    }


def _local_rotation_changes(m0: torch.Tensor, candidate: torch.Tensor, mask: torch.Tensor) -> dict[str, dict[str, float | None]]:
    m0_body = decode_rot6d_safe(m0[..., MOTION_LAYOUT.body_pose].reshape(m0.shape[0], 21, 6))
    candidate_body = decode_rot6d_safe(candidate[..., MOTION_LAYOUT.body_pose].reshape(candidate.shape[0], 21, 6))
    angle = _geodesic(candidate_body, m0_body) * 180.0 / math.pi
    result = {}
    for name in ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_foot", "right_foot"):
        body_index = SMPLX_22_JOINT_INDEX[name] - 1
        result[name] = _summary(angle[:, body_index][mask])
    return result


def _dose_metrics(
    m0_motion: torch.Tensor,
    candidate_motion: torch.Tensor,
    m0_vertices: torch.Tensor,
    candidate_vertices: torch.Tensor,
    patches: dict[str, dict[str, torch.Tensor]],
    dose: int,
) -> tuple[list[dict], bool]:
    rows = []
    any_regression = False
    for side in ("left", "right"):
        m0_heel, m0_toe = _centres(m0_vertices, patches[side])
        heel, toe = _centres(candidate_vertices, patches[side])
        flat, contact_info = _contact_mask(m0_heel, m0_toe)
        m0_gap = m0_heel[:, 2] - m0_toe[:, 2]
        gap = heel[:, 2] - toe[:, 2]
        horizontal = torch.linalg.vector_norm((toe - heel)[:, :2], dim=-1).clamp_min(1e-8)
        m0_horizontal = torch.linalg.vector_norm((m0_toe - m0_heel)[:, :2], dim=-1).clamp_min(1e-8)
        toe_down_pitch = torch.atan2(gap, horizontal) * 180.0 / math.pi
        m0_toe_down_pitch = torch.atan2(m0_gap, m0_horizontal) * 180.0 / math.pi
        heel_delta = heel[:, 2] - m0_heel[:, 2]
        toe_delta = toe[:, 2] - m0_toe[:, 2]
        selected_count = int(flat.sum())
        toe_only = flat & (gap >= TOE_GAP_THRESHOLD_M)
        toe_only_fraction = float(toe_only.sum() / max(selected_count, 1))
        regression = dose == 10 and selected_count >= 3 and toe_only_fraction >= REGRESSION_FRACTION
        any_regression = any_regression or regression
        flagged = torch.nonzero(toe_only, as_tuple=False).flatten()
        rows.append({
            "side": side,
            "dose_deg": dose,
            **contact_info,
            "toe_only_threshold_m": TOE_GAP_THRESHOLD_M,
            "toe_only_fraction": toe_only_fraction,
            "toe_contact_regression": regression,
            "flagged_frames": flagged.tolist(),
            "heel_minus_toe_gap_delta_m": _summary((gap - m0_gap)[flat]),
            "heel_height_delta_m": _summary(heel_delta[flat]),
            "toe_height_delta_m": _summary(toe_delta[flat]),
            "toe_down_pitch_delta_deg": _summary((toe_down_pitch - m0_toe_down_pitch)[flat]),
            "local_rotation_change_deg": _local_rotation_changes(m0_motion, candidate_motion, flat),
        })
    return rows, any_regression


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mean", type=Path, required=True)
    parser.add_argument("--std", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()

    mean = torch.from_numpy(np.load(args.mean)).float()
    std = torch.from_numpy(np.load(args.std)).float()
    device = torch.device(args.device)
    model = SMPLX(
        model_path=_default_smpl_model_path("smplx"), gender="neutral", num_betas=10,
        batch_size=100, use_pca=False,
    ).to(device)
    patches = _foot_patches(model)
    rows = []
    any_regression = False
    for seed in args.seeds:
        config_dir = args.root / "runs" / "smoke" / f"seed_{seed:03d}" / PARAMETER_KEY
        attempts = {delta: _latest_attempt(config_dir, delta) for delta in DELTAS}
        m0_batch = _load_physical(
            attempts[5] / "guided_artifacts" / "batch_000" / "m0_consistent_norm_batch.pt", mean, std
        )
        candidates = {
            delta: _load_physical(
                attempts[delta] / "guided_artifacts" / "batch_000" / "g0_norm_batch.pt", mean, std
            ) for delta in DELTAS
        }
        for sample_index, sample_id in enumerate(("94", "34122")):
            m0_motion = m0_batch[sample_index]
            m0_vertices = _mesh_vertices(m0_motion, model, device)
            for delta in DELTAS:
                candidate_motion = candidates[delta][sample_index]
                candidate_vertices = _mesh_vertices(candidate_motion, model, device)
                dose_rows, regression = _dose_metrics(
                    m0_motion, candidate_motion, m0_vertices, candidate_vertices, patches, delta
                )
                for row in dose_rows:
                    row.update({"seed": seed, "sample_id": sample_id})
                rows.extend(dose_rows)
                any_regression = any_regression or regression

    result = {
        "protocol": "vimogen_relative_root_forward_v1_3_shadow_pose_hierarchical",
        "detector": {
            "m0_contact_height_m": CONTACT_HEIGHT_M,
            "m0_contact_speed_m_per_frame": CONTACT_SPEED_M,
            "m0_flat_gap_m": FLAT_GAP_THRESHOLD_M,
            "candidate_toe_gap_m": TOE_GAP_THRESHOLD_M,
            "regression_fraction": REGRESSION_FRACTION,
            "patch_source": "neutral_SMPL-X_template_bottom_quartile_and_skinning_weights",
        },
        "toe_contact_regression_detected": any_regression,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    flagged = [row for row in rows if row["toe_contact_regression"]]
    print(json.dumps({
        "toe_contact_regression_detected": any_regression,
        "flagged_rows": len(flagged),
        "output": str(args.output),
    }, ensure_ascii=False))
    if args.fail_on_regression and any_regression:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
