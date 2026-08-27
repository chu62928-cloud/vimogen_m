#!/usr/bin/env python3
"""Validate the v3 smoke tail and consistency metrics without model execution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _tail_metrics(csv_path: Path, tail_frames: int = 8) -> dict:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty angle CSV: {csv_path}")
    result = {}
    for method, column in (("M0", "m0_angle_deg"), ("G0", "g0_angle_deg"), ("G1", "g1_angle_deg")):
        values = [float(row[column]) for row in rows]
        steps = [abs(values[index] - values[index - 1]) for index in range(1, len(values))]
        tail = values[-tail_frames:]
        tail_steps = [abs(tail[index] - tail[index - 1]) for index in range(1, len(tail))]
        result[method] = {
            "frame_count": len(values),
            "max_step_deg": max(steps, default=0.0),
            "tail_max_step_deg": max(tail_steps, default=0.0),
            "tail_values_deg": tail,
        }
    return result


def validate_run(run_root: Path) -> dict:
    summary = json.loads((run_root / "summaries/summary.json").read_text(encoding="utf-8"))
    tails = _tail_metrics(run_root / "angles/per_frame_angles.csv")
    g0_summary = summary["summaries"]["G0"]
    g1_summary = summary["summaries"]["G1"]
    return {
        "protocol": summary["protocol"],
        "run_root": str(run_root),
        "target_mean_deg": summary["target_mean_deg"],
        "tails": tails,
        "g0_summary": g0_summary,
        "g1_summary": g1_summary,
        "tail_spike_gate": {
            "threshold_deg": 2.0,
            "g0_pass": tails["G0"]["tail_max_step_deg"] <= 2.0,
            "g1_pass": tails["G1"]["tail_max_step_deg"] <= 2.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {"protocol": "vimogen_absolute_mean_pelvis_v3_tail_safe", "runs": [validate_run(path) for path in args.run_root]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
