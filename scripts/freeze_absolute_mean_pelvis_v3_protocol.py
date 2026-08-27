#!/usr/bin/env python3
"""Freeze the independent v3 protocol and byte-identical split manifests."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/phase6/absolute_mean_pelvis_v2"
DEST = ROOT / "results/phase6/absolute_mean_pelvis_v3"
PROTOCOL_NAME = "vimogen_absolute_mean_pelvis_v3_tail_safe"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not SOURCE.joinpath("protocol.json").is_file():
        raise FileNotFoundError(SOURCE / "protocol.json")
    if DEST.exists() and any(DEST.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {DEST}")
    DEST.mkdir(parents=True, exist_ok=True)
    source_data = SOURCE / "data"
    destination_data = DEST / "data"
    shutil.copytree(source_data, destination_data)

    source_protocol = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    protocol = dict(source_protocol)
    protocol["protocol"] = PROTOCOL_NAME
    protocol["status"] = "FROZEN_BEFORE_V3_MODEL_RUNS"
    protocol["supersedes"] = {
        "protocol": source_protocol["protocol"],
        "protocol_path": "/root/autodl-tmp/vimogen_clean/results/phase6/absolute_mean_pelvis_v2/protocol.json",
        "protocol_sha256": sha256_file(SOURCE / "protocol.json"),
        "reason": "v2 tail fusion extrapolated a hidden T+1 endpoint and replicated it into the window-9 average; v3 holds the hidden endpoint at the final physical pose and averages only physical output rows",
        "v2_results_policy": "retain unchanged as historical engineering evidence; never overwrite or reinterpret v2 results",
    }
    protocol["tail_boundary"] = {
        "name": "output_only_window_then_hidden_pose_hold_last",
        "physical_output_frames": "T",
        "hidden_pack_frame": "T+1 held equal to fused frame T",
        "endpoint_extrapolation": "disabled",
        "moving_average": "truncated_valid_window_no_endpoint_replication",
    }
    protocol["authority_pipeline"] = list(protocol["authority_pipeline"]) + [
        "v3 tail boundary: fuse only T physical poses with truncated windows, then hold hidden T+1 pose at the final fused pose",
    ]
    protocol["formulas"] = dict(protocol["formulas"])
    protocol["formulas"]["fused_root"] = "R_auth[t] = Exp(truncated_MA9(Log(R_direct[t] * transpose(R_velocity[t])))) * R_velocity[t], t in [0,T-1]; hidden R[T] = R_auth[T-1]"
    protocol["formulas"]["root_rotation_velocity"] = "dR_t = R_(t+1) * transpose(R_t), with hidden R_T held at R_(T-1)"
    protocol["formulas"]["root_translation_velocity"] = "dT_t = T_(t+1) - T_t, with hidden T_T held at T_(T-1)"
    protocol["implementation"] = {
        "consistency_module": "motion_rep/consistency_v3.py",
        "guidance_module": "sampling/absolute_mean_pelvis_guidance_v3.py",
        "runner": "scripts/run_absolute_mean_pelvis_v3.py",
        "evaluator": "scripts/evaluate_absolute_mean_pelvis_v3.py",
    }
    files = {}
    for path in sorted(destination_data.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(destination_data)).replace("\\", "/")] = sha256_file(path)
    protocol["data"] = dict(protocol["data"])
    protocol["data"]["files_sha256"] = files
    protocol["data"]["selection_reused_byte_for_byte_from_v2"] = True
    (DEST / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"protocol": PROTOCOL_NAME, "path": str(DEST / "protocol.json"), "data_files": len(files), "sha256": sha256_file(DEST / "protocol.json")}, indent=2))


if __name__ == "__main__":
    main()
