#!/usr/bin/env python3
"""Run the three predeclared S1 M7 local-solver strengths on a paired M0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--code-revision", default="scale-local-pelvis-s1-20260918")
    args = parser.parse_args()
    if not args.source.is_dir():
        raise FileNotFoundError(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, step in (("half", 0.5), ("center", 1.0), ("double", 2.0)):
        config_id = f"s1_m7_{label}"
        destination = args.output / config_id / "attempt_01"
        if destination.exists():
            summary_path = destination / "summary.json"
            if not summary_path.exists():
                raise RuntimeError(f"incomplete M7 attempt; inspect before retry: {destination}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            command = [
                sys.executable, str(ROOT / "experiments/run_m7_smoke_from_cache.py"),
                "--source", f"0={args.source}",
                "--output", str(destination),
                "--code-commit", args.code_revision,
                "--checkpoint-hash", args.checkpoint_hash,
                "--task", "relative_pelvis_s1",
                "--config-id", config_id,
                "--experiment-stage", "development",
                "--iterations", "20",
                "--max-step-deg", str(step),
                "--dose", "-5", "--dose", "0", "--dose", "5",
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        rows.append({
            "config_id": config_id, "max_step_deg": step,
            "status": summary["status"],
            "angle_gate_passes": summary["angle_gate_passes"],
            "expected_runs": summary["expected_runs"],
            "output": str(destination),
        })
        (args.output / "progress.json").write_text(
            json.dumps({"status": "RUNNING", "rows": rows}, indent=2) + "\n",
            encoding="utf-8",
        )
    result = {"status": "COMPLETE", "rows": rows}
    (args.output / "progress.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
