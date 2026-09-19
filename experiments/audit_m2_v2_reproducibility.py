#!/usr/bin/env python3
"""Compare repeated M2-v2 batch runs with identical frozen inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from experiments.audit_m2_v2_single_batch import _load_run, _metrics


TOLERANCES = {
    "initial_noise_max_abs": 1.0e-6,
    "candidate_norm_max_abs": 1.0e-6,
    "selected_source_noise_max_abs": 1.0e-6,
    "angle_mae_abs_deg": 1.0e-5,
    "angle_p95_abs_deg": 1.0e-5,
    "mpjpe_abs_mm": 1.0e-3,
    "root_translation_p95_abs_mm": 1.0e-3,
    "iteration_history_max_abs": 1.0e-6,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compare(reference: dict[str, Any], other: dict[str, Any]) -> list[dict[str, Any]]:
    if reference["sample_ids"] != other["sample_ids"]:
        raise ValueError("repeated runs have different sample IDs")
    rows = []
    for index, sample in enumerate(reference["sample_ids"]):
        ref_metrics = _metrics(reference["records"][sample])
        other_metrics = _metrics(other["records"][sample])
        history_values: list[float] = []
        reference_history = reference["iteration_history"][index]
        other_history = other["iteration_history"][index]
        if len(reference_history) != len(other_history):
            history_values.append(float("inf"))
        else:
            for ref_step, other_step in zip(reference_history, other_history):
                if ref_step.keys() != other_step.keys():
                    history_values.append(float("inf"))
                    continue
                for key in ref_step:
                    ref_value, other_value = ref_step[key], other_step[key]
                    if isinstance(ref_value, (int, float)) and isinstance(
                        other_value, (int, float)
                    ):
                        history_values.append(abs(float(ref_value) - float(other_value)))
                    elif ref_value != other_value:
                        history_values.append(float("inf"))
        observed = {
            "initial_noise_max_abs": float(
                (reference["initial_noise"][index] - other["initial_noise"][index])
                .abs()
                .max()
            ),
            "candidate_norm_max_abs": float(
                (reference["candidate"][index] - other["candidate"][index]).abs().max()
            ),
            "selected_source_noise_max_abs": float(
                (reference["source"][index] - other["source"][index]).abs().max()
            ),
            "angle_mae_abs_deg": abs(
                ref_metrics["angle_mae_deg"] - other_metrics["angle_mae_deg"]
            ),
            "angle_p95_abs_deg": abs(
                ref_metrics["angle_p95_deg"] - other_metrics["angle_p95_deg"]
            ),
            "mpjpe_abs_mm": abs(ref_metrics["mpjpe_mm"] - other_metrics["mpjpe_mm"]),
            "root_translation_p95_abs_mm": abs(
                ref_metrics["root_translation_p95_mm"]
                - other_metrics["root_translation_p95_mm"]
            ),
            "iteration_history_max_abs": max(history_values, default=0.0),
        }
        checks = {
            name: value <= TOLERANCES[name] for name, value in observed.items()
        }
        checks["best_iteration_equal"] = (
            reference["best_iterations"][index] == other["best_iterations"][index]
        )
        rows.append(
            {
                "sample_id": sample,
                "reference_best_iteration": reference["best_iterations"][index],
                "other_best_iteration": other["best_iterations"][index],
                "observed_differences": observed,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return rows


def audit(run_roots: list[Path]) -> dict[str, Any]:
    if len(run_roots) < 2:
        raise ValueError("at least two repeated runs are required")
    runs = [_load_run(root) for root in run_roots]
    reference = runs[0]
    rows = []
    for index, other in enumerate(runs[1:], start=1):
        rows.append(
            {
                "comparison_index": index,
                "other_root": str(run_roots[index]),
                "rows": _compare(reference, other),
            }
        )
    passed = all(
        row["passed"]
        for comparison in rows
        for row in comparison["rows"]
    )
    return {
        "protocol": "vimogen_m2_v2_reproducibility_v1",
        "status": "PASS" if passed else "FAIL",
        "tolerances": dict(TOLERANCES),
        "reference_root": str(run_roots[0]),
        "reference_archive_sha256": _sha256(reference["archive_path"]),
        "run_roots": [str(root) for root in run_roots],
        "comparisons": rows,
        "s1_tuning_allowed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite reproducibility report: {args.output}")
    result = audit(args.run)
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "comparisons": len(result["comparisons"])}))


if __name__ == "__main__":
    main()
