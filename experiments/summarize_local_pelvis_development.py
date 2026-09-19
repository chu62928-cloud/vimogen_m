#!/usr/bin/env python3
"""Read-only audit of the staged local-pelvis development results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


METHODS = tuple(f"M{index}" for index in range(1, 8))
DOSES = (-5.0, 0.0, 5.0)


def _attempt_number(path: Path) -> int:
    return int(next(part.split("_")[1] for part in path.parts if part.startswith("attempt_")))


def summarize(root: Path) -> dict:
    latest: dict[tuple[str, str, int, str, float], tuple[int, Path, dict]] = {}
    record_paths = list((root / "evaluation").glob("*/attempt_*/*/run_record.json"))
    record_paths.extend((root / "m7_development").glob("*/attempt_*/*/run_record.json"))
    for path in record_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("experiment_stage") != "development":
            continue
        key = (
            str(record["method_name"]), str(record["config_id"]),
            int(record["seed"]), str(record["prompt_id"]),
            float(record["target_dose_deg"]),
        )
        number = _attempt_number(path)
        if key not in latest or number > latest[key][0]:
            latest[key] = (number, path, record)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    baseline_hashes: dict[tuple[int, str], set[str]] = defaultdict(set)
    rows = []
    for key, (_, path, record) in sorted(latest.items()):
        method, config_id, seed, sample, dose = key
        control = record.get("all_metrics", {}).get("control", {})
        sequences = control.get("per_sequence", [])
        metric = sequences[0] if len(sequences) == 1 else None
        diagnostics = record.get("all_diagnostics", {})
        valid = bool(
            metric is not None
            and record.get("status") in {"COMPLETED", "ANGLE_GATE_FAIL"}
            and not diagnostics.get("fallback_used", False)
            and diagnostics.get("nonfinite_count", 0) == 0
            and (dose != 0.0 or record.get("zero_dose_identity") is True)
        )
        checks = (
            "target_hit", "world_pelvis_gate", "trunk_gate",
            "root_trajectory_gate", "temporal_gate",
        )
        measured_pass = bool(
            valid and all(metric.get(name) is True for name in checks)
            and metric["foot_direction_change_deg"]["p95"] <= 3.0
        )
        row = {
            "method": method, "config_id": config_id, "seed": seed,
            "sample": sample, "dose": dose, "record": str(path),
            "valid": valid,
            "target_hit": bool(valid and metric["target_hit"]),
            "measured_noncontact_pass": measured_pass,
            "relative_mae_deg": metric["relative_angle_error_deg"]["mean"] if metric else None,
            "relative_p95_deg": metric["relative_angle_error_deg"]["p95"] if metric else None,
            "relative_response_deg": metric["relative_angle_delta_mean_deg"] if metric else None,
            "world_pelvis_response_deg": metric["world_pelvis_delta_mean_deg"] if metric else None,
            "trunk_rotation_p95_deg": metric["trunk_rotation_change_deg"]["p95"] if metric else None,
            "foot_direction_p95_deg": metric["foot_direction_change_deg"]["p95"] if metric else None,
            "root_translation_p95_mm": metric["root_translation_deviation_mm"]["p95"] if metric else None,
            "wall_time_sec": diagnostics.get("wall_time_sec"),
            "zero_dose_identity": record.get("zero_dose_identity"),
            "source_m0_sha256": record.get("source_m0_sha256"),
            "code_commit": record.get("code_commit"),
            "physical_status": record.get("all_metrics", {}).get("physical", {}).get("status"),
        }
        rows.append(row)
        groups[(method, config_id)].append(row)
        baseline_hashes[(seed, sample)].add(row["source_m0_sha256"])

    configurations = []
    for (method, config_id), values in sorted(groups.items()):
        dev = [row for row in values if row["seed"] == 0 and row["sample"] == "94"]
        doses = {row["dose"] for row in dev}
        active = [row for row in dev if row["dose"] != 0.0]
        configurations.append({
            "method": method, "config_id": config_id,
            "complete_development_slots": doses == set(DOSES),
            "valid_slots": sum(row["valid"] for row in dev),
            "target_hit_active_slots": sum(row["target_hit"] for row in active),
            "measured_noncontact_pass_active_slots": sum(row["measured_noncontact_pass"] for row in active),
            "active_mean_relative_mae_deg": (
                sum(row["relative_mae_deg"] for row in active if row["relative_mae_deg"] is not None)
                / len(active) if len(active) == 2 else None
            ),
            "total_wall_time_sec": sum(row["wall_time_sec"] or 0 for row in dev),
            "contact_and_visual_status": "NOT_EVALUATED",
        })
    return {
        "status": "DEVELOPMENT_AUDIT_ONLY",
        "full_numeric_pass_claim_allowed": False,
        "reason": "contact markers and user video review remain unavailable",
        "methods_with_results": sorted({row["method"] for row in rows}),
        "methods_missing": [method for method in METHODS if method not in {row["method"] for row in rows}],
        "paired_m0_hash_consistent": all(len(hashes) == 1 for hashes in baseline_hashes.values()),
        "paired_m0_hashes": {f"{seed}:{sample}": sorted(hashes) for (seed, sample), hashes in baseline_hashes.items()},
        "configurations": configurations,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
