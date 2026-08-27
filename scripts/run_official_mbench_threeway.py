#!/usr/bin/env python3
"""Run the official MBench motion-quality dimensions for all frozen variants.

Each method is evaluated in its own directory because the official evaluator
expects files named ``<global_id>.npy`` at the evaluation root.  The script is
resumable and refuses to overwrite a completed per-motion result.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

METHODS = ("absolute_position", "velocity_integral", "reconciled")
CONDITIONS = ("m0", "m1_plus5", "m1_plus10")
SEEDS = (0, 1, 2)
DIMENSIONS = (
    "Jitter_Degree",
    "Ground_Penetration",
    "Foot_Floating",
    "Foot_Sliding",
    "Dynamic_Degree",
    "Body_Penetration",
    "Pose_Quality",
)


def _find_per_motion(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("*_per_motion_results.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--organized-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-count", type=int, default=450)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            for method in METHODS:
                jobs.append((condition, seed, method))

    for condition, seed, method in jobs:
        eval_dir = args.organized_root / condition / f"seed_{seed:03d}" / method
        output_dir = args.output_root / condition / f"seed_{seed:03d}" / method
        output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(eval_dir.glob("*.npy")) if eval_dir.is_dir() else []
        if len(files) != args.expected_count:
            raise RuntimeError(f"{eval_dir}: expected {args.expected_count} .npy files, found {len(files)}")
        existing = _find_per_motion(output_dir)
        record_path = output_dir / "run_record.json"
        if existing is not None and record_path.exists():
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                record = {}
            if record.get("status") == "COMPLETED":
                print(f"SKIP completed {condition} seed {seed} {method}", flush=True)
                continue

        name = f"mbench_{condition}_seed{seed}_{method}"
        command = [
            sys.executable,
            "evaluate_mbench.py",
            "--evaluation_path", str(eval_dir),
            "--output_path", str(output_dir),
            "--dimension", *DIMENSIONS,
            "--device", args.device,
        ]
        record = {
            "status": "RUNNING",
            "protocol": "vimogen_publication_mbench_motion_quality_v1",
            "condition": condition,
            "seed": seed,
            "method": method,
            "dimensions": list(DIMENSIONS),
            "evaluation_path": str(eval_dir),
            "output_path": str(output_dir),
            "expected_count": args.expected_count,
            "command": command,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        log_path = output_dir / "evaluate.log"
        print(f"RUN {condition} seed {seed} {method}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        per_motion = _find_per_motion(output_dir)
        record["returncode"] = completed.returncode
        record["per_motion_path"] = str(per_motion) if per_motion else None
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["status"] = "COMPLETED" if completed.returncode == 0 and per_motion else "FAILED"
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if record["status"] != "COMPLETED":
            raise RuntimeError(f"official MBench failed for {condition} seed {seed} {method}; see {log_path}")


if __name__ == "__main__":
    main()
