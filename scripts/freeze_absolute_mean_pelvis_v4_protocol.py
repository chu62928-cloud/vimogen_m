#!/usr/bin/env python3
"""Freeze v4 split manifests plus a reviewed anatomical calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_rep.anatomical_pelvis import V4_PROTOCOL, calibration_sha256, load_pelvis_calibration


SOURCE = ROOT / "results/phase6/absolute_mean_pelvis_v3"
DEST = ROOT / "results/phase6/absolute_mean_pelvis_v4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--destination", type=Path, default=DEST)
    args = parser.parse_args()
    if not args.source.joinpath("protocol.json").is_file():
        raise FileNotFoundError(args.source / "protocol.json")
    calibration = load_pelvis_calibration(args.calibration)
    if args.destination.exists() and any(args.destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.destination}")
    args.destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.source / "data", args.destination / "data")
    source_protocol = json.loads((args.source / "protocol.json").read_text(encoding="utf-8"))
    protocol = dict(source_protocol)
    protocol.update({
        "protocol": V4_PROTOCOL,
        "status": "FROZEN_BEFORE_V4_MODEL_RUNS",
        "supersedes": {
            "protocol": source_protocol.get("protocol"),
            "protocol_path": str(args.source / "protocol.json"),
            "protocol_sha256": sha256_file(args.source / "protocol.json"),
            "reason": "v4 replaces the uncalibrated root-axis proxy with reviewed project-specific LASI/RASI/LPSI/RPSI virtual marker groups and local anti-cheat control",
            "v3_results_policy": "retain unchanged as historical engineering evidence; never overwrite or reinterpret v3 results",
        },
        "angle_definition": {
            "name": "calibrated_anatomical_pelvis_local_sagittal_tilt",
            "markers": "A=(LASI+RASI)/2; P=(LPSI+RPSI)/2; v=A-P",
            "formula": "atan2(-dot(v,up), dot(v,heading))",
            "positive_direction": "anterior side down",
            "heading": "horizontal projection of v, with degenerate frames marked invalid",
        },
        "calibration": calibration.to_mapping() | {"canonical_sha256": calibration_sha256(calibration)},
        "anti_cheat": {
            "active_rotation_channels": {"root_pelvis": [258, 264], "left_hip": [0, 6], "right_hip": [6, 12], "spine1": [12, 18], "spine2": [30, 36], "spine3": [48, 54]},
            "soft_limit_deg": 2.0,
            "p95_limit_deg": 3.0,
            "loss": "mean(ReLU(abs(delta_trunk/thigh)-2deg)^2 over three segments)",
            "coefficient": 1.0,
            "local_dominance": "fixed hinge requiring absolute local pelvis-trunk change >= 0.5 of pelvis change and same sign above 0.5deg signal",
            "local_share_gate": ">0.5 and same sign when absolute pelvis change is at least 0.5deg",
            "ratio": "audit only; no standalone pass threshold",
        },
        "implementation": {
            "geometry_module": "motion_rep/anatomical_pelvis.py",
            "consistency_module": "motion_rep/consistency_v3.py",
            "guidance_module": "sampling/absolute_mean_pelvis_guidance_v4.py",
            "runner": "scripts/run_absolute_mean_pelvis_v4.py",
            "evaluator": "scripts/evaluate_absolute_mean_pelvis_v4.py",
        },
        "fixed": dict(source_protocol.get("fixed", {})) | {"targets_deg": [5.0, 10.0], "fusion_window": 9, "anti_cheat_soft_limit_deg": 2.0, "anti_cheat_p95_limit_deg": 3.0},
    })
    files = {}
    for path in sorted((args.destination / "data").rglob("*")):
        if path.is_file():
            files[str(path.relative_to(args.destination / "data")).replace("\\", "/")] = sha256_file(path)
    protocol["data"] = dict(protocol.get("data", {})) | {"files_sha256": files, "selection_reused_byte_for_byte_from_v3": True}
    (args.destination / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": V4_PROTOCOL, "path": str(args.destination / "protocol.json"), "calibration_sha256": calibration_sha256(calibration), "sha256": sha256_file(args.destination / "protocol.json")}, indent=2))


if __name__ == "__main__":
    main()
