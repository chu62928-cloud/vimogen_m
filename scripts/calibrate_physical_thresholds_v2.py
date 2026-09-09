#!/usr/bin/env python3
"""Freeze an independent, dual-track physical-gate v2 protocol.

The v2 protocol never reads candidate-method results.  It starts from the
existing paired-M0-derived scalar limits, changes only the two suspected
overly-sensitive limits, and removes duplicated contact-height hard gates.
Synthetic perturbation ladders determine whether the predeclared candidate
limits detect light versus severe physical defects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.physical_metrics import (  # noqa: E402
    EVALUATED_FAIL,
    EVALUATED_PASS,
    PHYSICAL_DECISION_PROFILE_V2,
    V2_HARD_METRICS,
    evaluate_physical_metrics_v3,
)
from evaluation.physical_reference import (  # noqa: E402
    REFERENCE_CACHE_VERSION,
    reference_contact_masks,
    reference_markers,
)
from scripts.calibrate_physical_thresholds import (  # noqa: E402
    EVENT_TOLERANCES,
    METRIC_NAMES,
    _baseline_rows,
    _derive_thresholds,
    _sha256,
)


PROTOCOL = "vimogen_m1_m7_physical_thresholds_v2"
DEFAULT_REFERENCE = (
    ROOT / "results/phase9/pelvis_m1_m7/physical_reference_v2/physical_reference.pt"
)
DEFAULT_OUTPUT = ROOT / "results/phase9/pelvis_m1_m7/physical_thresholds_v2"

# Ordered from the least relaxation to the largest predeclared relaxation.
# The selection rule is deterministic and is evaluated only on synthetic M0
# perturbations, never on candidate outputs.
CANDIDATE_LIMITS = (
    {"support_height_error_p95_mm": 2.0, "floating_frame_rate": 0.025},
    {"support_height_error_p95_mm": 3.0, "floating_frame_rate": 0.025},
    {"support_height_error_p95_mm": 3.0, "floating_frame_rate": 0.05},
    {"support_height_error_p95_mm": 5.0, "floating_frame_rate": 0.05},
)


def _strict_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _clone_markers(
    markers: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        side: {marker: value.clone() for marker, value in values.items()}
        for side, values in markers.items()
    }


def _contact_indices(mask: torch.Tensor, fraction: float) -> torch.Tensor:
    indices = torch.where(mask)[0]
    if not indices.numel():
        return indices
    count = max(1, int(round(float(indices.numel()) * fraction)))
    return indices[: min(count, indices.numel())]


def _perturb(
    baseline: Mapping[str, Mapping[str, torch.Tensor]],
    contacts: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    up_axis: int,
    horizontal_axis: int,
    kind: str,
    amount: float,
    fraction: float = 1.0,
) -> dict[str, dict[str, torch.Tensor]]:
    candidate = _clone_markers(baseline)
    for side in ("left", "right"):
        for marker in ("heel", "toe"):
            value = candidate[side][marker]
            contact = contacts[side][marker][0]
            selected = _contact_indices(contact, fraction)
            if kind == "support":
                value[0, selected, up_axis] += amount / 1000.0
            elif kind == "float":
                value[0, selected, up_axis] += amount / 1000.0
            elif kind == "penetration":
                value[0, :, up_axis] -= amount / 1000.0
            elif kind == "slide":
                ramp = torch.arange(
                    value.shape[1], dtype=value.dtype, device=value.device
                )
                value[0, :, horizontal_axis] += ramp * amount / 1000.0
            else:
                raise ValueError(f"unknown perturbation kind: {kind}")
    return candidate


def _evaluate_case(
    reference: Mapping[str, Any],
    index: int,
    thresholds: Mapping[str, float],
    *,
    kind: str | None = None,
    amount: float = 0.0,
    fraction: float = 1.0,
) -> dict[str, Any]:
    baseline = reference_markers(reference, index)
    contacts, pairs = reference_contact_masks(reference, index)
    valid = torch.as_tensor(reference["valid_frame_mask"])[index : index + 1].bool()
    ground = torch.as_tensor(reference["ground_height_m"])[index : index + 1]
    up_axis = int(reference["world_up_axis"])
    horizontal_axis = 0 if up_axis != 0 else 1
    candidate = baseline if kind is None else _perturb(
        baseline,
        contacts,
        up_axis=up_axis,
        horizontal_axis=horizontal_axis,
        kind=kind,
        amount=amount,
        fraction=fraction,
    )
    result = evaluate_physical_metrics_v3(
        candidate,
        baseline,
        valid,
        contacts,
        ground,
        contact_pair_masks=pairs,
        up_axis=up_axis,
        thresholds=thresholds,
        decision_profile=PHYSICAL_DECISION_PROFILE_V2,
    )
    row = result["per_sequence"][0]
    return {
        "status": row["status"],
        "physical_pass": row["physical_pass"],
        "physical_fail_reasons": row["physical_fail_reasons"],
        "metrics": {
            name: row.get(name)
            for name in METRIC_NAMES
        },
    }


def _calibration_cases(
    reference: Mapping[str, Any],
    index: int,
    thresholds: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    return {
        "clean_m0": _evaluate_case(reference, index, thresholds),
        "light_support_3mm": _evaluate_case(
            reference, index, thresholds, kind="support", amount=3.0
        ),
        "light_float_2pct": _evaluate_case(
            reference, index, thresholds, kind="float", amount=30.0, fraction=0.02
        ),
        "severe_support_10mm": _evaluate_case(
            reference, index, thresholds, kind="support", amount=10.0
        ),
        "severe_float_30mm_10pct": _evaluate_case(
            reference, index, thresholds, kind="float", amount=30.0, fraction=0.10
        ),
        "severe_penetration_20mm": _evaluate_case(
            reference, index, thresholds, kind="penetration", amount=20.0
        ),
        "severe_sliding_50mm_per_frame": _evaluate_case(
            reference, index, thresholds, kind="slide", amount=50.0
        ),
    }


def _candidate_is_valid(cases: Mapping[str, Mapping[str, Any]]) -> bool:
    light_pass = all(
        cases[name]["status"] == EVALUATED_PASS
        for name in ("clean_m0", "light_support_3mm", "light_float_2pct")
    )
    severe_fail = all(
        cases[name]["status"] == EVALUATED_FAIL
        for name in (
            "severe_support_10mm",
            "severe_float_30mm_10pct",
            "severe_penetration_20mm",
            "severe_sliding_50mm_per_frame",
        )
    )
    return light_pass and severe_fail


def freeze_physical_thresholds_v2(
    *, reference_path: Path, output: Path, code_commit: str
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite physical thresholds: {output}")
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    if reference.get("cache_version") != REFERENCE_CACHE_VERSION:
        raise ValueError("unsupported physical reference cache version")
    baseline_rows = _baseline_rows(reference)
    base_thresholds = _derive_thresholds(baseline_rows)
    index = max(
        range(len(reference["sample_ids"])),
        key=lambda item: int(
            torch.as_tensor(reference["contact_mask"])[item].sum().item()
        ),
    )

    candidate_results: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for candidate_index, overrides in enumerate(CANDIDATE_LIMITS):
        thresholds = dict(base_thresholds)
        thresholds.update(overrides)
        cases = _calibration_cases(reference, index, thresholds)
        result = {
            "candidate_index": candidate_index,
            "overrides": overrides,
            "thresholds": thresholds,
            "cases": cases,
            "valid": _candidate_is_valid(cases),
        }
        candidate_results.append(result)
        if selected is None and result["valid"]:
            selected = result

    if selected is None:
        raise RuntimeError("no v2 threshold candidate passed independent calibration")

    thresholds = selected["thresholds"]
    output.mkdir(parents=True)
    protocol = {
        "protocol": PROTOCOL,
        "status": "FROZEN_PHYSICAL_THRESHOLDS_V2",
        "code_commit": str(code_commit),
        "reference_cache_version": REFERENCE_CACHE_VERSION,
        "reference_cache_sha256": _sha256(reference_path),
        "calibration_sources": [
            "PAIRED_M0_SELF_EVALUATION",
            "PROGRAMMATIC_M0_PERTURBATION_LADDER",
        ],
        "candidate_results_read": False,
        "blind_s2_results_read": False,
        "decision_profile": PHYSICAL_DECISION_PROFILE_V2,
        "hard_metrics": sorted(V2_HARD_METRICS),
        "derivation": {
            "base_protocol": "vimogen_m1_m7_physical_thresholds_v1",
            "selection": "strictest predeclared candidate passing light controls and detecting severe controls",
            "support_light_control": "3 mm on frozen contact frames",
            "floating_light_control": "30 mm lift on <=2% frozen contact frames",
            "severe_controls": "10 mm support error, 30 mm lift on 10% contact frames, 20 mm penetration, 50 mm/frame sliding",
            "contact_height_hard_gate": False,
        },
        "selected_candidate_index": selected["candidate_index"],
        "thresholds": thresholds,
        "baseline_maxima": {
            metric: max(float(row[metric]) for row in baseline_rows)
            for metric in METRIC_NAMES
        },
        "candidate_ladder": candidate_results,
        "calibration_reference": {
            "selected_paired_m0_id": str(reference["paired_m0_ids"][index]),
            "seed": int(reference["seeds"][index]),
            "sample_id": str(reference["sample_ids"][index]),
        },
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
            "protocol": PROTOCOL,
            "reference_cache_sha256": protocol["reference_cache_sha256"],
            "candidate_results_read": False,
            "selected_candidate_index": selected["candidate_index"],
            "candidate_ladder": candidate_results,
        },
    )
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    protocol = freeze_physical_thresholds_v2(
        reference_path=args.reference,
        output=args.output,
        code_commit=args.code_commit,
    )
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "selected_candidate_index": protocol["selected_candidate_index"],
                "thresholds": protocol["thresholds"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
