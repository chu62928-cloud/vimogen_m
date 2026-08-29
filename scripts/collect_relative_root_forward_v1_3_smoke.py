"""Collect compact metrics from v1.3 server smoke artifacts.

This script is intentionally tensor-free: it only reads the JSON artifacts
written by the server runner, so it can also be used from a CPU-only checkout.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DELTA_RE = re.compile(r"delta_([+-]?\d+(?:\.\d+)?)deg$")


def _latest_attempt(delta_dir: Path) -> Path | None:
    attempts = []
    for p in delta_dir.glob("attempt_*"):
        try:
            attempts.append((int(p.name.split("_", 1)[1]), p))
        except (IndexError, ValueError):
            continue
    # A transient distributed-launch failure can create a higher-numbered
    # failed attempt after a valid generation.  Prefer the newest attempt
    # that actually contains a guidance summary, while retaining the newest
    # directory as a diagnostic fallback.
    newest = max(attempts, default=(0, None))[1]
    for _, attempt in sorted(attempts, reverse=True):
        if (attempt / "guided_artifacts" / "batch_000" / "guidance_summary.json").is_file():
            return attempt
    return newest


def _sample_record(sample: dict, *, seed: int, delta: float, sample_index: int, run_dir: Path) -> dict:
    metrics = sample.get("metrics", {})
    per_sample = (metrics.get("per_sample") or [{}])[0]
    tail = (metrics.get("tail_safety", {}).get("per_sample") or [{}])[0]
    whole_body = metrics.get("whole_body", {})
    solver = sample.get("solver", {})
    step_records = sample.get("step_records", [])
    active = [r for r in step_records if r.get("active")]
    accepted = [r for r in active if r.get("accepted")]
    last_sigma = max((r.get("sigma", 0.0) for r in accepted), default=None)
    return {
        "seed": seed,
        "sample_index": sample_index,
        "target_delta_deg": delta,
        "run_dir": str(run_dir),
        "metrics": {
            "mean_absolute_error_deg": per_sample.get("mean_absolute_error_deg"),
            "p95_absolute_error_deg": per_sample.get("p95_absolute_error_deg"),
            "forward_vector_error_p95_deg": per_sample.get("forward_vector_error_p95_deg"),
            "horizontal_heading_drift_p95_deg": per_sample.get("horizontal_heading_drift_p95_deg"),
            "dose_sign_correct": per_sample.get("dose_sign_correct"),
            "q_rigid": whole_body.get("q_rigid"),
            "trunk_p95_deg": whole_body.get("trunk_change_deg", {}).get("p95"),
            "root_change_mean_deg": whole_body.get("root_change_deg", {}).get("mean"),
            "root_change_p95_deg": whole_body.get("root_change_deg", {}).get("p95"),
            "tail_extra_so3_jump_max_deg": tail.get("tail_extra_so3_jump_max_deg"),
            "tail_extra_pitch_step_max_deg": tail.get("tail_extra_pitch_step_max_deg"),
            "consistency_pass": (metrics.get("consistency", {}).get("candidate") or [{}])[0].get("passed"),
        },
        "solver": {
            "step_record_count": len(step_records),
            "active_step_count": len(active),
            "accepted_step_count": len(accepted),
            "last_accepted_sigma": last_sigma,
            "solver_summary": solver,
        },
    }


def collect(root: Path, seed: int) -> dict:
    seed_root = root / "runs" / "smoke" / f"seed_{seed:03d}"
    rows = []
    for delta_dir in sorted(seed_root.glob("*/delta_*deg")):
        match = DELTA_RE.search(delta_dir.name)
        if not match:
            continue
        delta = float(match.group(1))
        attempt = _latest_attempt(delta_dir)
        if attempt is None:
            continue
        summary_path = attempt / "guided_artifacts" / "batch_000" / "guidance_summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        records = summary.get("records", [])
        samples = records[0].get("samples", []) if records else []
        for sample_index, sample in enumerate(samples):
            row = _sample_record(
                sample,
                seed=seed,
                delta=delta,
                sample_index=sample_index,
                run_dir=attempt,
            )
            rows.append(row)
    return {"protocol": "vimogen_relative_root_forward_v1_3_shadow_pose_hierarchical", "seed": seed, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.root, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({"rows": len(result["rows"]), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
