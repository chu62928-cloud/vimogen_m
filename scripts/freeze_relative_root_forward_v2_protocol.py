#!/usr/bin/env python3
"""Freeze the v2 source-noise protocol before any formal expansion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "vimogen_relative_root_forward_v2_minimal_source_noise"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"protocol already frozen: {output}")
    code_files = [
        ROOT / "sampling/differentiable_flow_sampler.py",
        ROOT / "sampling/relative_root_forward_guidance_v2.py",
        ROOT / "train_eval_vimogen.py",
    ]
    record = {
        "protocol": PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "implementation_calibration_only",
        "authority": "final_single_authority_project_from_selected_Gz",
        "mapping": "deterministic_50_step_flow_sampler_G(z)",
        "source_noise": {
            "variable": "z=z0+delta_z",
            "optimization_space": "whitened_normalized_source_noise",
            "pose_channel_edits_during_sampling": False,
            "gradient_terms": ["root_full_forward_only"],
            "external_only_terms": [
                "trunk_direction",
                "q_rigid",
                "heel_toe_height",
                "toe_dominance",
                "foot_sliding",
                "floating",
                "penetration",
                "naturalness",
            ],
            "feasibility_constraints": {
                "pitch_mae_deg_max": 1.0,
                "forward_p95_deg_max": 2.0,
                "dose_sign_correct": True,
            },
            "second_level_objective": "minimum_source_delta_rms_among_observed_feasible_candidates",
            "trust_region_max_delta_rms": 1.0,
        },
        "reproduction_stop_gate": {
            "steps": 50,
            "batch_size": 1,
            "dtype": "bfloat16",
            "outputs_must_be_bitwise_equal": [
                "raw",
                "official_pre_cast",
                "official",
            ],
            "gradient_must_be_finite_and_nonzero": True,
            "max_reserved_mib": 28672.0,
        },
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in code_files
        },
        "dynamic_environment": {
            "runner": "connect_server.py",
            "python": "/root/miniconda3/envs/mdm5090/bin/python",
            "gpu": "NVIDIA GeForce RTX 5090",
        },
        "status_boundary": "no formal MBench claim until a fresh frozen-protocol run passes all constraints",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/phase7/relative_root_forward_v2/protocol.json",
    )
    args = parser.parse_args()
    print(json.dumps(freeze(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
