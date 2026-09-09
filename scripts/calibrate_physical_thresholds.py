#!/usr/bin/env python3
"""Freeze S0 physical gates from paired M0 and synthetic perturbations only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.physical_metrics import EVALUATED_FAIL, evaluate_physical_metrics_v3
from evaluation.physical_reference import (
    REFERENCE_CACHE_VERSION,
    reference_contact_masks,
    reference_markers,
)


PROTOCOL = "vimogen_m1_m7_physical_thresholds_v1"
DEFAULT_REFERENCE = (
    ROOT
    / "results/phase9/pelvis_m1_m7/physical_reference_v2/physical_reference.pt"
)
DEFAULT_OUTPUT = ROOT / "results/phase9/pelvis_m1_m7/physical_thresholds_v1"
EVENT_TOLERANCES = {
    "penetration_tolerance_mm": 1.0,
    "floating_height_threshold_mm": 25.0,
}
METRIC_NAMES = (
    "penetration_p95_mm",
    "penetration_max_mm",
    "penetration_frame_rate",
    "contact_tangent_speed_p95_mm_per_frame",
    "contact_tangent_speed_max_mm_per_frame",
    "total_slide_distance_mm",
    "max_segment_slide_distance_mm",
    "contact_height_p95_mm",
    "contact_height_max_mm",
    "floating_frame_rate",
    "support_height_error_p95_mm",
)
RATE_METRICS = {"penetration_frame_rate", "floating_frame_rate"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _baseline_rows(reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    event_only = dict(EVENT_TOLERANCES)
    for index in range(len(reference["sample_ids"])):
        markers = reference_markers(reference, index)
        contacts, pairs = reference_contact_masks(reference, index)
        result = evaluate_physical_metrics_v3(
            markers,
            markers,
            torch.as_tensor(reference["valid_frame_mask"])[index : index + 1].bool(),
            contacts,
            torch.as_tensor(reference["ground_height_m"])[index : index + 1],
            contact_pair_masks=pairs,
            up_axis=int(reference["world_up_axis"]),
            thresholds=event_only,
        )
        row = dict(result["per_sequence"][0])
        if not row["contact_evaluable"]:
            raise RuntimeError("every calibration M0 must have continuous contact evidence")
        row.update(
            seed=int(reference["seeds"][index]),
            sample_id=str(reference["sample_ids"][index]),
            paired_m0_id=str(reference["paired_m0_ids"][index]),
        )
        rows.append(row)
    return rows


def _margin(metric: str, baseline: float) -> float:
    absolute = 0.01 if metric in RATE_METRICS else 1.0
    return max(abs(float(baseline)) * 0.05, absolute)


def _derive_thresholds(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    thresholds = dict(EVENT_TOLERANCES)
    for metric in METRIC_NAMES:
        values = [row.get(metric) for row in rows]
        if not values or any(value is None for value in values):
            raise RuntimeError(f"M0 calibration metric unavailable: {metric}")
        baseline_max = max(float(value) for value in values)
        thresholds[metric] = baseline_max + _margin(metric, baseline_max)
    return thresholds


def _clone_markers(
    markers: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        side: {marker: value.clone() for marker, value in side_markers.items()}
        for side, side_markers in markers.items()
    }


def _synthetic_sanity(
    reference: Mapping[str, Any], thresholds: Mapping[str, float]
) -> dict[str, Any]:
    index = max(
        range(len(reference["sample_ids"])),
        key=lambda item: int(
            torch.as_tensor(reference["contact_mask"])[item].sum().item()
        ),
    )
    baseline = reference_markers(reference, index)
    contacts, pairs = reference_contact_masks(reference, index)
    valid = torch.as_tensor(reference["valid_frame_mask"])[index : index + 1].bool()
    ground = torch.as_tensor(reference["ground_height_m"])[index : index + 1]
    up_axis = int(reference["world_up_axis"])
    horizontal_axis = 0 if up_axis != 0 else 1

    cases: dict[str, dict[str, Any]] = {}
    for name in ("penetration_20mm", "floating_50mm", "sliding_50mm_per_frame"):
        candidate = _clone_markers(baseline)
        if name == "penetration_20mm":
            for side in candidate.values():
                for marker in side.values():
                    marker[..., up_axis] -= 0.020
        elif name == "floating_50mm":
            for side in candidate.values():
                for marker in side.values():
                    marker[..., up_axis] += 0.050
        else:
            ramp = torch.arange(
                valid.shape[1], dtype=ground.dtype, device=ground.device
            ).reshape(1, -1) * 0.050
            for side in candidate.values():
                for marker in side.values():
                    marker[..., horizontal_axis] += ramp
        result = evaluate_physical_metrics_v3(
            candidate,
            baseline,
            valid,
            contacts,
            ground,
            contact_pair_masks=pairs,
            up_axis=up_axis,
            thresholds=thresholds,
        )
        cases[name] = {
            "status": result["status"],
            "physical_fail_reasons": result["per_sequence"][0][
                "physical_fail_reasons"
            ],
        }
    all_failed = all(row["status"] == EVALUATED_FAIL for row in cases.values())
    return {
        "source": "PROGRAMMATIC_PAIRED_M0_PERTURBATIONS",
        "selected_paired_m0_id": reference["paired_m0_ids"][index],
        "cases": cases,
        "all_expected_failures_observed": all_failed,
    }


def freeze_physical_thresholds(
    *,
    reference_path: Path,
    output: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Create one immutable threshold protocol without reading candidates."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite physical thresholds: {output}")
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    if reference.get("cache_version") != REFERENCE_CACHE_VERSION:
        raise ValueError("unsupported physical reference cache version")
    rows = _baseline_rows(reference)
    thresholds = _derive_thresholds(rows)
    sanity = _synthetic_sanity(reference, thresholds)
    if not sanity["all_expected_failures_observed"]:
        raise RuntimeError("synthetic physical sanity checks did not all fail")

    output.mkdir(parents=True)
    baseline_maxima = {
        metric: max(float(row[metric]) for row in rows) for metric in METRIC_NAMES
    }
    protocol = {
        "protocol": PROTOCOL,
        "status": "FROZEN_PHYSICAL_THRESHOLDS",
        "code_commit": str(code_commit),
        "reference_cache_version": REFERENCE_CACHE_VERSION,
        "reference_cache": str(reference_path),
        "reference_cache_sha256": _sha256(reference_path),
        "calibration_sources": ["PAIRED_M0_SELF_EVALUATION"],
        "candidate_results_read": False,
        "blind_s2_results_read": False,
        "decision_profile": "all_metrics",
        "hard_metrics": list(METRIC_NAMES),
        "derivation": {
            "sequence_rule": "all listed sequence metrics must be <= their limits",
            "scalar_limit": "max paired-M0 + max(5% of max paired-M0, 1 mm)",
            "rate_limit": "max paired-M0 + max(5% of max paired-M0, 0.01)",
            "penetration_event": "depth > 1 mm",
            "floating_event": "frozen-contact marker height > 25 mm",
        },
        "thresholds": thresholds,
        "baseline_maxima": baseline_maxima,
        "m0_rows": rows,
        "synthetic_sanity": sanity,
        "invariants": {
            "paired_m0_only": True,
            "candidate_tuning_forbidden": True,
            "candidate_contact_reclassification_forbidden": True,
            "overwrite_forbidden": True,
        },
    }
    _strict_json(output / "thresholds.json", protocol)
    _strict_json(
        output / "calibration_evidence.json",
        {
            key: protocol[key]
            for key in (
                "protocol",
                "code_commit",
                "reference_cache_sha256",
                "calibration_sources",
            "candidate_results_read",
            "decision_profile",
            "hard_metrics",
            "derivation",
                "baseline_maxima",
                "m0_rows",
                "synthetic_sanity",
            )
        },
    )
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--code-commit", default=None)
    args = parser.parse_args()
    protocol = freeze_physical_thresholds(
        reference_path=args.reference,
        output=args.output,
        code_commit=args.code_commit or _git_commit(),
    )
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "protocol": protocol["protocol"],
                "synthetic_sanity": protocol["synthetic_sanity"][
                    "all_expected_failures_observed"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
