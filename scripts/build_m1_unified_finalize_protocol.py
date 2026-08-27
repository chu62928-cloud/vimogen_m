#!/usr/bin/env python3
"""Freeze the reviewer1-only protocol for the unified-finalization revision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_protocol(manifest_path: Path, input_path: Path, output_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"FROZEN", "FROZEN_SINGLE_REVIEW_OVERRIDE"}:
        raise ValueError("unified-finalization protocol requires a frozen manifest")
    if output_path.exists():
        raise FileExistsError(output_path)
    protocol = {
        "status": "FROZEN_PROTOCOL",
        "revision": "m1_unified_finalize_v1",
        "review_protocol": manifest.get("review_protocol", "reviewer1_only_user_override"),
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "status": manifest["status"],
        },
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "samples": 20,
        "seeds": [0, 1, 2],
        "commands_degrees": [0.0, 5.0, 10.0],
        "noise": {
            "sample_protocol": "vimogen-sample-noise-v1",
            "batch_invariant": True,
            "reuse_existing_sample_level_z0": True,
            "changes_noise_semantics": False,
        },
        "common_finalizer": {
            "implementation": "motion_rep.unified_finalizer.finalize_motion_tensor",
            "boundary": "physical_Tplus1_to_Tx276",
            "input_boundary": "de-standardize_before_recovery_when_declared",
            "root_anchor": "first_root_rotation_and_first_root_translation",
            "recovery": "root_rot_velocity_and_position_velocity_cumulative_forward_difference",
            "body_rotation_last_pose": "hold_last_stored_body_pose",
            "velocity_policy": "recompute_all_joint_and_root_velocities_from_recovered_Tplus1_stream",
            "mask_policy": "input_row_mask_plus_last_row_then_pair_mask",
            "output_boundary": "physical_or_explicitly_restandardized",
        },
        "methods": {
            "M0": {"path": "M0 raw -> common finalizer -> B0"},
            "B0": {"path": "common finalizer", "status": "LEGAL_NO_EDIT_REFERENCE"},
            "B1": {
                "path": "root_rotation_only diagnostic",
                "status": "INTENTIONAL_DIAGNOSTIC_BYPASS",
                "common_finalized_derivative": "reported separately; not the primary B1 diagnostic",
            },
            "B2": {"path": "rigid edit -> common finalizer", "status": "LEGAL_BASELINE"},
            "M1": {"path": "M1 raw/official -> common finalizer -> unified output"},
        },
        "evaluation": {
            "strict_per_unit_angle_threshold_degrees": 2.0,
            "report_before_after_finalization": True,
            "keep_turning_samples": True,
            "no_posthoc_sample_exclusion": True,
            "model_rerun_required": False,
            "reason": "revision reuses existing M1 model outputs and only changes the frozen representation boundary",
        },
        "environment": {
            "python": "/root/miniconda3/envs/mdm5090/bin/python",
            "dtype": "bfloat16 source artifacts; float32 offline finalization metrics",
            "attention": "SDPA",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_protocol(args.manifest, args.input, args.output), indent=2))


if __name__ == "__main__":
    main()

