#!/usr/bin/env python3
"""Audit one M2-v2 batch against matching singleton replays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


TOLERANCES = {
    "candidate_norm_max_abs": 1.0e-5,
    "selected_source_noise_max_abs": 1.0e-6,
    "angle_mae_abs_deg": 1.0e-4,
    "angle_p95_abs_deg": 1.0e-4,
    "mpjpe_abs_mm": 1.0e-3,
    "root_translation_p95_abs_mm": 1.0e-3,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _one(root: Path, pattern: str) -> Path:
    paths = list(root.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {pattern!r} below {root}, found {len(paths)}")
    return paths[0]


def _load_run(root: Path) -> dict[str, Any]:
    artifact = root / "guided_artifacts/batch_000"
    archive_path = _one(root, "trainer/**/batch_000/mbench_raw_norm_batch.pt")
    archive = torch.load(archive_path, map_location="cpu", weights_only=True)
    sample_ids = [str(value) for value in archive["sample_ids"]]
    candidate = torch.load(
        artifact / "g0_norm_batch.pt", map_location="cpu", weights_only=True
    ).float()
    source = torch.load(
        artifact / "selected_source_noise_batch.pt",
        map_location="cpu",
        weights_only=True,
    ).float()
    initial_noise = torch.load(
        root / "m0_artifacts/batch_000/z0_replayed.pt",
        map_location="cpu",
        weights_only=True,
    ).float()
    if candidate.shape[0] != len(sample_ids) or source.shape[0] != len(sample_ids):
        raise ValueError(f"batch/sample metadata mismatch below {root}")
    summary = json.loads((artifact / "guidance_summary.json").read_text(encoding="utf-8"))
    outer = summary.get("records", [])
    diagnostics = outer[0].get("diagnostics", {}) if outer else {}
    best_iterations = diagnostics.get("per_sample_best_iteration", [])
    if len(best_iterations) != len(sample_ids):
        raise ValueError(f"missing per-sample best iterations below {root}")
    iteration_history = diagnostics.get("iteration_history", [])
    if len(iteration_history) != len(sample_ids):
        raise ValueError(f"missing per-sample iteration history below {root}")
    records: dict[str, dict[str, Any]] = {}
    for path in (root / "evaluation").glob("*/run_record.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        records[str(value["prompt_id"])] = value
    if set(records) != set(sample_ids):
        raise ValueError(f"evaluation records do not match archive below {root}")
    return {
        "root": root,
        "archive_path": archive_path,
        "sample_ids": sample_ids,
        "candidate": candidate,
        "source": source,
        "initial_noise": initial_noise,
        "best_iterations": [int(value) for value in best_iterations],
        "iteration_history": iteration_history,
        "records": records,
    }


def _metrics(record: dict[str, Any]) -> dict[str, float]:
    control = record["all_metrics"]["control"]["per_sequence"][0]
    content = record["all_metrics"]["content"]["per_sequence"][0]
    return {
        "angle_mae_deg": float(control["angle_mae_deg"]),
        "angle_p95_deg": float(control["angle_p95_deg"]),
        "mpjpe_mm": float(content["mpjpe_vs_m0_mm"]),
        "root_translation_p95_mm": float(
            content["root_translation_deviation_p95_mm"]
        ),
    }


def audit(batch_root: Path, singleton_roots: list[Path]) -> dict[str, Any]:
    batch = _load_run(batch_root)
    singletons = [_load_run(root) for root in singleton_roots]
    by_sample: dict[str, tuple[dict[str, Any], int]] = {}
    for singleton in singletons:
        if len(singleton["sample_ids"]) != 1:
            raise ValueError(f"singleton run has more than one sample: {singleton['root']}")
        sample = singleton["sample_ids"][0]
        if sample in by_sample:
            raise ValueError(f"duplicate singleton sample {sample}")
        by_sample[sample] = (singleton, 0)
    if set(by_sample) != set(batch["sample_ids"]):
        raise ValueError("singleton sample IDs do not match the batch")

    rows = []
    for batch_index, sample in enumerate(batch["sample_ids"]):
        singleton, single_index = by_sample[sample]
        batch_metrics = _metrics(batch["records"][sample])
        single_metrics = _metrics(singleton["records"][sample])
        observed = {
            "candidate_norm_max_abs": float(
                (batch["candidate"][batch_index] - singleton["candidate"][single_index])
                .abs()
                .max()
            ),
            "selected_source_noise_max_abs": float(
                (batch["source"][batch_index] - singleton["source"][single_index])
                .abs()
                .max()
            ),
            "angle_mae_abs_deg": abs(
                batch_metrics["angle_mae_deg"] - single_metrics["angle_mae_deg"]
            ),
            "angle_p95_abs_deg": abs(
                batch_metrics["angle_p95_deg"] - single_metrics["angle_p95_deg"]
            ),
            "mpjpe_abs_mm": abs(
                batch_metrics["mpjpe_mm"] - single_metrics["mpjpe_mm"]
            ),
            "root_translation_p95_abs_mm": abs(
                batch_metrics["root_translation_p95_mm"]
                - single_metrics["root_translation_p95_mm"]
            ),
        }
        checks = {
            name: value <= TOLERANCES[name] for name, value in observed.items()
        }
        checks["best_iteration_equal"] = (
            batch["best_iterations"][batch_index]
            == singleton["best_iterations"][single_index]
        )
        rows.append(
            {
                "sample_id": sample,
                "batch_index": batch_index,
                "batch_best_iteration": batch["best_iterations"][batch_index],
                "singleton_best_iteration": singleton["best_iterations"][single_index],
                "observed_differences": observed,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return {
        "protocol": "vimogen_m2_v2_single_batch_consistency_v1",
        "status": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "tolerances": dict(TOLERANCES),
        "batch_root": str(batch_root),
        "batch_archive_sha256": _sha256(batch["archive_path"]),
        "singleton_roots": [str(root) for root in singleton_roots],
        "rows": rows,
        "s1_tuning_allowed": all(row["passed"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-run", type=Path, required=True)
    parser.add_argument("--singleton-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite consistency report: {args.output}")
    result = audit(args.batch_run, args.singleton_run)
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
