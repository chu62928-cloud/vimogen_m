#!/usr/bin/env python3
"""Render every completed v0.4 sample94 case immediately after evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_rep.pose_authority import authority_project  # noqa: E402
from sampling.pelvis_contact_flow_projection_v0_1 import write_strict_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol = json.loads((args.protocol_root / "protocol.json").read_text(encoding="utf-8"))
    mean = torch.from_numpy(np.load(protocol["inputs"]["mean"]["path"])).float()
    std = torch.from_numpy(np.load(protocol["inputs"]["std"]["path"])).float()
    valid = torch.load(args.protocol_root / "valid_mask.pt", map_location="cpu", weights_only=True).bool()[0:1]
    m0 = torch.load(args.protocol_root / "m0_physical.pt", map_location="cpu", weights_only=True).float()[0:1]
    candidate_norm = torch.load(args.run_root / "projection_artifacts" / "batch_000" / "projected_g0_norm_batch.pt", map_location="cpu", weights_only=True).float()
    candidate = authority_project(candidate_norm * std.view(1, 1, -1) + mean.view(1, 1, -1), valid_mask=valid).physical_motion
    staging = args.run_root / "videos" / "_renderer_input"
    staging.mkdir(parents=True, exist_ok=True)
    torch.save(m0, staging / "m0_physical.pt")
    torch.save(candidate, staging / "diagnostic_motion.pt")
    output_dir = args.output_dir or (args.run_root / "videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    record = json.loads((args.run_root / "run_record.json").read_text(encoding="utf-8"))
    command = [
        sys.executable,
        str(ROOT / "scripts/render_pelvis_contact_walk_diagnostic.py"),
        "--run-root", str(staging),
        "--candidate-path", str(staging / "diagnostic_motion.pt"),
        "--output-dir", str(output_dir),
        "--target-delta-deg", str(record.get("target_delta_deg", 2.0)),
        "--device", args.device,
    ]
    subprocess.run(command, check=True)
    mode = str(record.get("contact_mode", "unknown"))
    dose = float(record.get("target_delta_deg", 0.0))
    for path in output_dir.glob("sample94_walk_*.mp4"):
        renamed = output_dir / f"sample94_{mode}_dose_{dose:+g}deg_{path.name.removeprefix('sample94_')}"
        if renamed != path:
            shutil.move(path, renamed)
    write_strict_json(output_dir / "render_record.json", {
        "sample_id": "94", "contact_mode": mode, "target_delta_deg": dose,
        "status": record.get("status"), "command": command,
    })
    print(json.dumps({"output_dir": str(output_dir), "status": record.get("status")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

