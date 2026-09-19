#!/usr/bin/env python3
"""Evaluate existing S0 outputs against one frozen paired-M0 cache."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.physical_metrics import (
    EVAL_ERROR,
    NOT_EVALUATED,
    PHYSICAL_DECISION_PROFILE_V1,
    REFERENCE_MISSING,
    EVALUATOR_VERSION_V3,
    evaluate_physical_metrics_v3,
)
from evaluation.physical_reference import (
    REFERENCE_CACHE_VERSION,
    marker_positions_from_motion,
    reference_contact_masks,
    reference_markers,
)
from scripts.freeze_s0_v1 import collect_sequence_records


DEFAULT_SAMPLING_ROOT = ROOT / "results/phase9/pelvis_m1_m7/s0_sampling"
DEFAULT_M7_ROOT = ROOT / "results/phase9/pelvis_m1_m7/s0_m7/attempt_01"
DEFAULT_REFERENCE = ROOT / "results/phase9/pelvis_m1_m7/physical_reference_v2/physical_reference.pt"
DEFAULT_OUTPUT = ROOT / "results/phase9/pelvis_m1_m7/s0_physical_evaluation_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_input_file(path: Path, filename: str) -> Path:
    """Accept either an immutable artifact directory or its exact file."""
    resolved = path / filename if path.is_dir() else path
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _strict_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_thresholds(
    path: Path | None,
) -> tuple[dict[str, float] | None, str, str]:
    if path is None:
        return None, "not_frozen", PHYSICAL_DECISION_PROFILE_V1
    path = resolve_input_file(path, "thresholds.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") not in {
        "FROZEN_PHYSICAL_THRESHOLDS",
        "FROZEN_PHYSICAL_THRESHOLDS_V2",
    }:
        raise ValueError("threshold file must contain a frozen physical status")
    thresholds = {str(key): float(limit) for key, limit in value["thresholds"].items()}
    return (
        thresholds,
        str(value.get("protocol", path.name)),
        str(value.get("decision_profile", PHYSICAL_DECISION_PROFILE_V1)),
    )


def _reference_index(reference: Mapping[str, Any]) -> dict[tuple[int, str], int]:
    result = {
        (int(seed), str(sample)): index
        for index, (seed, sample) in enumerate(
            zip(reference["seeds"], reference["sample_ids"])
        )
    }
    if len(result) != len(reference["sample_ids"]):
        raise ValueError("physical reference contains duplicate seed/sample keys")
    return result


def _spot_checks(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    selected: list[dict[str, Any]] = []
    for method in tuple(f"M{index}" for index in range(1, 8)):
        candidates = [row for row in rows if row["method"] == method]
        if candidates:
            selected.append(
                max(candidates, key=lambda row: float(row.get("mpjpe_vs_m0_mm") or -1.0))
            )
    remaining = sorted(
        (row for row in rows if row not in selected),
        key=lambda row: float(row.get("mpjpe_vs_m0_mm") or -1.0),
        reverse=True,
    )
    selected.extend(remaining[: max(0, count - len(selected))])
    return [
        {
            "run_id": row["run_id"],
            "method": row["method"],
            "seed": row["seed"],
            "sample_id": row["sample_id"],
            "dose_deg": row["dose_deg"],
            "candidate_motion_sha256": row.get("candidate_motion_sha256"),
            "physical_status": row["physical"]["status"],
            "metrics": row["physical"].get("per_sequence", [{}])[0],
            "physical_v2_status": (
                row.get("physical_v2") or {}
            ).get("status"),
            "physical_v2_metrics": (
                (row.get("physical_v2") or {}).get("per_sequence", [{}])[0]
            ),
            "programmatic_checks": {
                "reference_key_matches": row["physical_reference_status"] == "MATERIALIZED",
                "candidate_hash_recorded": bool(row.get("candidate_motion_sha256")),
                "marker_metrics_present": bool(
                    row["physical"].get("per_sequence", [{}])[0].get("marker_metrics")
                ),
            },
        }
        for row in selected[:count]
    ]


def run(
    *,
    source_paths: list[Path],
    reference_path: Path,
    output: Path,
    thresholds_path: Path | None = None,
    thresholds_v2_path: Path | None = None,
    spot_check_count: int = 8,
    expected_count: int = 84,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite physical evaluation: {output}")
    reference_path = resolve_input_file(reference_path, "physical_reference.pt")
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    if reference.get("cache_version") != REFERENCE_CACHE_VERSION:
        raise ValueError("unsupported physical reference cache version")
    thresholds, threshold_version, decision_profile = _load_thresholds(thresholds_path)
    thresholds_v2, threshold_version_v2, decision_profile_v2 = _load_thresholds(
        thresholds_v2_path
    )
    reference_lookup = _reference_index(reference)
    records = collect_sequence_records(source_paths)
    if len(records) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} canonical records, found {len(records)}"
        )
    rows: list[dict[str, Any]] = []

    for key in sorted(records):
        record, record_path = records[key]
        method, seed, dose, sample_id = key
        candidate_path = record_path.parent / "motion_physical.pt"
        content = record.get("all_metrics", {}).get("content", {}).get("per_sequence", [{}])[0]
        row: dict[str, Any] = {
            "run_id": str(record["run_id"]),
            "method": method,
            "seed": seed,
            "dose_deg": dose,
            "sample_id": sample_id,
            "original_record": str(record_path),
            "candidate_motion": str(candidate_path),
            "candidate_motion_sha256": sha256(candidate_path) if candidate_path.is_file() else None,
            "baseline_motion_id": record.get("baseline_motion_id"),
            "mpjpe_vs_m0_mm": content.get("mpjpe_vs_m0_mm"),
        }
        reference_key = (seed, sample_id)
        if reference_key not in reference_lookup:
            physical = {
                "evaluator_version": EVALUATOR_VERSION_V3,
                "status": REFERENCE_MISSING,
                "reason": "PAIRED_M0_REFERENCE_KEY_MISSING",
                "physical_pass": None,
                "per_sequence": [],
            }
            physical_v2 = dict(physical)
            row["physical_reference_status"] = REFERENCE_MISSING
        elif not candidate_path.is_file():
            physical = {
                "evaluator_version": EVALUATOR_VERSION_V3,
                "status": EVAL_ERROR,
                "reason": "CANDIDATE_MOTION_MISSING",
                "physical_pass": False,
                "per_sequence": [],
            }
            physical_v2 = dict(physical)
            row["physical_reference_status"] = "MATERIALIZED"
        else:
            index = reference_lookup[reference_key]
            valid = reference["valid_frame_mask"][index : index + 1].bool()
            try:
                candidate_motion = torch.load(
                    candidate_path, map_location="cpu", weights_only=True
                ).float()
                if candidate_motion.ndim == 2:
                    candidate_motion = candidate_motion.unsqueeze(0)
                candidate_markers, _, _ = marker_positions_from_motion(
                    candidate_motion, valid
                )
                baseline_markers = reference_markers(reference, index)
                contacts, pairs = reference_contact_masks(reference, index)
                physical = evaluate_physical_metrics_v3(
                    candidate_markers,
                    baseline_markers,
                    valid,
                    contacts,
                    reference["ground_height_m"][index : index + 1],
                    contact_pair_masks=pairs,
                    up_axis=int(reference["world_up_axis"]),
                    thresholds=thresholds,
                    decision_profile=decision_profile,
                )
                physical_v2 = (
                    evaluate_physical_metrics_v3(
                        candidate_markers,
                        baseline_markers,
                        valid,
                        contacts,
                        reference["ground_height_m"][index : index + 1],
                        contact_pair_masks=pairs,
                        up_axis=int(reference["world_up_axis"]),
                        thresholds=thresholds_v2,
                        decision_profile=decision_profile_v2,
                    )
                    if thresholds_v2 is not None
                    else None
                )
                row["physical_reference_status"] = "MATERIALIZED"
                row["paired_m0_id"] = reference["paired_m0_ids"][index]
            except Exception as error:
                physical = {
                    "evaluator_version": EVALUATOR_VERSION_V3,
                    "status": EVAL_ERROR,
                    "reason": repr(error),
                    "physical_pass": False,
                    "per_sequence": [],
                }
                physical_v2 = {
                    "evaluator_version": EVALUATOR_VERSION_V3,
                    "status": EVAL_ERROR,
                    "reason": repr(error),
                    "physical_pass": False,
                    "per_sequence": [],
                }
                row["physical_reference_status"] = "MATERIALIZED"
        row["physical"] = physical
        row["physical_v1"] = physical
        row["physical_v2"] = physical_v2
        rows.append(row)
        _strict_json(output / "records" / row["run_id"] / "physical_evaluation.json", row)

    counts = Counter(row["physical"]["status"] for row in rows)
    counts_v2 = Counter(
        row["physical_v2"]["status"]
        for row in rows
        if row.get("physical_v2") is not None
    )
    spot_checks = _spot_checks(rows, min(max(spot_check_count, 0), 8))
    summary = {
        "protocol": (
            "vimogen_m1_m7_s0_physical_evaluation_v2"
            if expected_count == 84
            else "vimogen_m1_m7_physical_record_set_v1"
        ),
        "status": (
            "S0_PHYSICAL_RAW_COMPLETE_THRESHOLDS_PENDING"
            if thresholds is None
            and len(rows) == expected_count
            and counts.get(NOT_EVALUATED, 0) == expected_count
            else "S0_PHYSICAL_EVALUATED"
            if expected_count == 84
            else "PHYSICAL_RECORD_SET_EVALUATED"
        ),
        "sequence_count": len(rows),
        "reference_cache": str(reference_path),
        "reference_cache_sha256": sha256(reference_path),
        "threshold_version": threshold_version,
        "decision_profile": decision_profile,
        "threshold_version_v2": threshold_version_v2
        if thresholds_v2 is not None
        else None,
        "decision_profile_v2": decision_profile_v2 if thresholds_v2 is not None else None,
        "physical_status_counts": dict(sorted(counts.items())),
        "physical_v2_status_counts": dict(sorted(counts_v2.items())),
        "records": rows,
        "spot_check_count": len(spot_checks),
        "s2_allowed": False,
        "s2_blockers": [
            "PHYSICAL_THRESHOLDS_NOT_FROZEN" if thresholds is None else None,
            "S1_UNIQUE_CONFIGS_NOT_FROZEN",
            "FOUR_GATE_DECISION_NOT_COMPLETE",
        ],
    }
    summary["s2_blockers"] = [value for value in summary["s2_blockers"] if value]
    _strict_json(output / "summary.json", summary)
    _strict_json(
        output / "spot_check_manifest.json",
        {
            "status": "PROGRAMMATIC_SPOT_CHECK_COMPLETE",
            "count": len(spot_checks),
            "items": spot_checks,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=Path)
    parser.add_argument("--sampling-root", type=Path, default=DEFAULT_SAMPLING_ROOT)
    parser.add_argument("--m7-root", type=Path, default=DEFAULT_M7_ROOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--thresholds-v2", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spot-check-count", type=int, default=8)
    parser.add_argument("--expected-count", type=int, default=84)
    args = parser.parse_args()
    sources = args.source or [args.sampling_root, args.m7_root]
    summary = run(
        source_paths=sources,
        reference_path=args.reference,
        thresholds_path=args.thresholds,
        thresholds_v2_path=args.thresholds_v2,
        output=args.output,
        spot_check_count=args.spot_check_count,
        expected_count=args.expected_count,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "sequence_count": summary["sequence_count"],
                "physical_status_counts": summary["physical_status_counts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
