#!/usr/bin/env python3
"""Probe one frozen M7 tensor for sign and rigid-body lean semantics.

The probe is read-only and uses the already generated physical 276-D motion,
its paired M0 reference, and the reviewed SMPL-X pelvis marker calibration.
The historical v4 marker file has ASIS/PSIS labels swapped; this probe applies
the documented correction in memory by treating the stored PSIS groups as the
anterior groups and the stored ASIS groups as the posterior groups.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path.cwd()))

import torch

from geometry.pelvis_angle import pelvis_angle_curve_deg, wrap_angle_deg
from guidance.base import tensor_sha256
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe


PROTOCOL = "vimogen_existing_m7_tensor_semantics_probe_v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_motion(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        value = value["motion"]
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or value.shape[-1] != MOTION_LAYOUT.total_dim:
        raise ValueError(f"{path} must contain physical [T,276] motion")
    return value.float()


def load_row(path: Path, sample_id: str, seed: int, dose: float) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("method") == "M7"
            and row.get("config_id") == "reference"
            and str(row.get("sample_id")) == str(sample_id)
            and int(row.get("seed", -1)) == int(seed)
            and math.isclose(float(row.get("target_dose_deg", "nan")), dose, abs_tol=1e-12)
        ]
    if len(rows) != 1:
        raise ValueError(f"expected one M7 row, found {len(rows)}")
    return rows[0]


def paired_m0(
    reference_path: Path,
    sample_id: str,
    seed: int,
    expected_sha256: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    matches = []
    for index, (sample, source_seed) in enumerate(
        zip(reference["sample_ids"], reference["seeds"])
    ):
        if str(sample) != str(sample_id) or int(source_seed) != int(seed):
            continue
        motion = reference["m0_motion_physical"][index].detach().cpu().float()
        if tensor_sha256(motion[None]) == expected_sha256:
            matches.append(index)
    if len(matches) != 1:
        raise ValueError(f"expected one paired M0 entry, found {len(matches)}")
    index = matches[0]
    motion = reference["m0_motion_physical"][index].detach().cpu().float()
    valid = reference["valid_frame_mask"][index].detach().cpu().bool()
    return motion, valid


def joints_and_root(motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    joints = motion[:, MOTION_LAYOUT.joints].reshape(motion.shape[0], 22, 3)
    root = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation])
    return joints, root


def unit(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-8)


def quantiles(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().cpu().float()
    return {
        "mean": float(value.mean()),
        "median": float(torch.quantile(value, 0.5)),
        "p95_abs": float(torch.quantile(value.abs(), 0.95)),
        "min": float(value.min()),
        "max": float(value.max()),
    }


def corrected_anatomical_angle(root: torch.Tensor, calibration_path: Path) -> torch.Tensor:
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    points = calibration["marker_local_points"]
    # The frozen v4 labels are reversed.  Corrected anterior = stored PSIS;
    # corrected posterior = stored ASIS, as independently verified by +z/nose.
    anterior = 0.5 * (
        torch.tensor(points["LPSI"], dtype=root.dtype)
        + torch.tensor(points["RPSI"], dtype=root.dtype)
    )
    posterior = 0.5 * (
        torch.tensor(points["LASI"], dtype=root.dtype)
        + torch.tensor(points["RASI"], dtype=root.dtype)
    )
    local_axis = anterior - posterior
    world_axis = root @ local_axis
    vertical = world_axis[:, 2]
    horizontal = torch.linalg.vector_norm(world_axis[:, :2], dim=-1)
    # Anatomical anterior tilt: anterior side moves down.
    return torch.rad2deg(torch.atan2(-vertical, horizontal))


def signed_segment_lean(
    segment: torch.Tensor, baseline_heading: torch.Tensor
) -> torch.Tensor:
    up = torch.tensor([0.0, 0.0, 1.0], dtype=segment.dtype)
    forward = (segment * baseline_heading).sum(-1)
    vertical = (segment * up).sum(-1)
    return torch.rad2deg(torch.atan2(forward, vertical))


def angular_deviation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    cosine = (unit(a) * unit(b)).sum(-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def run(args: argparse.Namespace) -> dict[str, Any]:
    row = load_row(args.metrics, args.sample_id, args.seed, args.dose)
    candidate_path = Path(row["motion_file"])
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    if sha256_file(candidate_path) != row["motion_sha256"]:
        raise ValueError("candidate motion file hash changed")
    candidate = load_motion(candidate_path)
    baseline, valid = paired_m0(
        args.reference,
        args.sample_id,
        args.seed,
        row["paired_m0_sha256"],
    )
    if candidate.shape != baseline.shape or valid.shape != candidate.shape[:1]:
        raise ValueError("candidate, M0, and valid mask shapes do not match")

    baseline = baseline[valid]
    candidate = candidate[valid]
    joints_m0, root_m0 = joints_and_root(baseline)
    joints_g, root_g = joints_and_root(candidate)

    current_delta = wrap_angle_deg(
        pelvis_angle_curve_deg(candidate[None])[0]
        - pelvis_angle_curve_deg(baseline[None])[0]
    )
    anatomical_delta = wrap_angle_deg(
        corrected_anatomical_angle(root_g, args.calibration)
        - corrected_anatomical_angle(root_m0, args.calibration)
    )

    local_forward = torch.tensor([0.0, 0.0, 1.0], dtype=root_m0.dtype)
    heading = root_m0 @ local_forward
    heading[:, 2] = 0.0
    heading = unit(heading)

    trunk_m0 = joints_m0[:, 12] - joints_m0[:, 3]
    trunk_g = joints_g[:, 12] - joints_g[:, 3]
    trunk_signed_delta = wrap_angle_deg(
        signed_segment_lean(trunk_g, heading) - signed_segment_lean(trunk_m0, heading)
    )
    trunk_world_deviation = angular_deviation(trunk_g, trunk_m0)

    foot_m0 = torch.stack(
        (joints_m0[:, 10] - joints_m0[:, 7], joints_m0[:, 11] - joints_m0[:, 8]),
        dim=1,
    )
    foot_g = torch.stack(
        (joints_g[:, 10] - joints_g[:, 7], joints_g[:, 11] - joints_g[:, 8]),
        dim=1,
    )
    foot_world_deviation = angular_deviation(foot_g, foot_m0).reshape(-1)

    centred_m0 = joints_m0[:, 1:] - joints_m0[:, :1]
    centred_g = joints_g[:, 1:] - joints_g[:, :1]
    root_local_m0 = torch.einsum("tij,tkj->tki", root_m0.transpose(-1, -2), centred_m0)
    root_local_g = torch.einsum("tij,tkj->tki", root_g.transpose(-1, -2), centred_g)
    root_local_error_mm = (
        torch.linalg.vector_norm(root_local_g - root_local_m0, dim=-1).mean(-1) * 1000.0
    )

    root_translation_error_mm = torch.linalg.vector_norm(
        candidate[:, MOTION_LAYOUT.root_translation]
        - baseline[:, MOTION_LAYOUT.root_translation],
        dim=-1,
    ) * 1000.0

    return {
        "protocol": PROTOCOL,
        "status": "CURRENT_CONTROL_SEMANTICS_FAIL",
        "sample_id": str(args.sample_id),
        "seed": int(args.seed),
        "target_dose_deg_current_semantics": float(args.dose),
        "frame_count": int(valid.sum()),
        "current_proxy_delta_deg": quantiles(current_delta),
        "corrected_anatomical_marker_delta_deg": quantiles(anatomical_delta),
        "signed_trunk_forward_lean_delta_deg": quantiles(trunk_signed_delta),
        "world_trunk_direction_deviation_deg": quantiles(trunk_world_deviation),
        "world_foot_direction_deviation_deg": quantiles(foot_world_deviation),
        "root_local_21_mean_error_mm": quantiles(root_local_error_mm),
        "root_translation_error_mm": quantiles(root_translation_error_mm),
        "interpretation": {
            "positive_signed_trunk_forward_lean": "forward lean",
            "negative_signed_trunk_forward_lean": "backward lean",
            "historical_marker_label_correction_applied": True,
        },
        "inputs": {
            "metrics": str(args.metrics.resolve()),
            "metrics_sha256": sha256_file(args.metrics),
            "reference": str(args.reference.resolve()),
            "reference_sha256": sha256_file(args.reference),
            "calibration": str(args.calibration.resolve()),
            "calibration_sha256": sha256_file(args.calibration),
            "candidate_motion": str(candidate_path.resolve()),
            "candidate_motion_sha256": sha256_file(candidate_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--sample-id", default="94")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dose", type=float, choices=(-10.0, 10.0), required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
