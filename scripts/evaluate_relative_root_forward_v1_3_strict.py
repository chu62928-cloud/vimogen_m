"""Apply the frozen v1.3 smoke gates to per-seed JSON summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


SEEDS = (0, 42, 464229750, 1057660199, 1386772747)
DELTAS = (-10.0, -5.0, 5.0, 10.0)
GATES = {
    "mean_absolute_error_deg": 1.0,
    "p95_absolute_error_deg": 2.0,
    "forward_vector_error_p95_deg": 2.0,
    "horizontal_heading_drift_p95_deg": 2.0,
    "trunk_p95_deg": 2.0,
    "q_rigid": 0.2,
    "tail_extra_so3_jump_max_deg": 2.0,
    "tail_extra_pitch_step_max_deg": 2.0,
}


def _row_failures(row: dict) -> list[str]:
    metrics = row.get("metrics", {})
    failures = []
    for name, limit in GATES.items():
        value = metrics.get(name)
        if value is None or float(value) > limit:
            failures.append(f"{name}>{limit}")
    for name in ("dose_sign_correct", "consistency_pass"):
        if metrics.get(name) is not True:
            failures.append(name)
    return failures


def evaluate(rows: list[dict]) -> dict:
    failures = []
    for row in rows:
        row["gate_failures"] = _row_failures(row)
        if row["gate_failures"]:
            failures.append({"seed": row["seed"], "sample_index": row["sample_index"], "target_delta_deg": row["target_delta_deg"], "failures": row["gate_failures"]})

    # Root-change magnitudes are stored as non-negative geodesic changes.
    # Compare +10 vs +5 and -10 vs -5 within every seed/sample pair.
    indexed = {(r["seed"], r["sample_index"], r["target_delta_deg"]): r for r in rows}
    monotonic_failures = []
    for seed in SEEDS:
        for sample_index in (0, 1):
            for small, large in ((5.0, 10.0), (-5.0, -10.0)):
                small_row = indexed.get((seed, sample_index, small))
                large_row = indexed.get((seed, sample_index, large))
                if small_row is None or large_row is None:
                    monotonic_failures.append({"seed": seed, "sample_index": sample_index, "pair": [small, large], "reason": "missing"})
                    continue
                small_value = small_row["metrics"].get("root_change_mean_deg")
                large_value = large_row["metrics"].get("root_change_mean_deg")
                if small_value is None or large_value is None or not abs(float(large_value)) > abs(float(small_value)):
                    monotonic_failures.append({"seed": seed, "sample_index": sample_index, "pair": [small, large], "small": small_value, "large": large_value})

    values = {name: [float(r["metrics"][name]) for r in rows if r.get("metrics", {}).get(name) is not None] for name in GATES}
    aggregate = {
        name: {"mean": mean(value), "std": pstdev(value), "max": max(value)}
        for name, value in values.items() if value
    }
    expected = len(SEEDS) * 2 * len(DELTAS)
    return {
        "protocol": "vimogen_relative_root_forward_v1_3_shadow_pose_hierarchical",
        "expected_rows": expected,
        "actual_rows": len(rows),
        "row_failures": failures,
        "monotonicity_failures": monotonic_failures,
        "aggregate": aggregate,
        "strict_pass": len(rows) == expected and not failures and not monotonic_failures,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.input:
        rows.extend(json.loads(path.read_text(encoding="utf-8")).get("rows", []))
    result = evaluate(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({"strict_pass": result["strict_pass"], "actual_rows": result["actual_rows"], "row_failures": len(result["row_failures"]), "monotonicity_failures": len(result["monotonicity_failures"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
