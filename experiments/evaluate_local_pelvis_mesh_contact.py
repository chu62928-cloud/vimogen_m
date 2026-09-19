#!/usr/bin/env python3
"""Materialize frozen-M0 sole markers and evaluate S1 candidate contact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.local_pelvis_contact_metrics import (
    evaluate_frozen_contact,
    frozen_contact_reference,
)
from evaluation.smplx_foot_markers import SMPLXFootMarkerExtractor
from experiments.summarize_local_pelvis_development import summarize
from guidance.base import tensor_sha256


def _one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} below {root}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--paired-m0", type=Path, required=True)
    parser.add_argument("--source-generation", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline = torch.load(args.paired_m0, map_location="cpu", weights_only=True).float()
    if baseline.ndim != 3 or baseline.shape[0] != 1 or baseline.shape[-1] != 276:
        raise ValueError("paired M0 must have shape [1,T,276]")
    archive = torch.load(
        _one(args.source_generation, "trainer/**/batch_000/mbench_raw_norm_batch.pt"),
        map_location="cpu", weights_only=True,
    )
    valid = archive["motion_mask"].bool()
    if valid.shape != baseline.shape[:2]:
        raise ValueError("M0 valid mask and motion shape disagree")
    audit = summarize(args.development_root)
    development_rows = [
        row for row in audit["rows"] if row["seed"] == 0 and row["sample"] == "94"
    ]
    # The target gate is a prerequisite for downstream diagnosis.  Zero-dose
    # identity has its own gate and needs no mesh-contact interpretation here.
    rows = [
        row for row in development_rows
        if row["dose"] != 0.0 and row["target_hit"]
    ]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("no S1 development records available")
    baseline_hash = tensor_sha256(baseline)
    if any(row["source_m0_sha256"] != baseline_hash for row in rows):
        raise ValueError("S1 records do not share the provided M0")
    extractor = SMPLXFootMarkerExtractor(
        args.smplx_model, frame_count=baseline.shape[1], device=args.device
    )
    m0_markers = extractor(baseline)
    frozen = frozen_contact_reference(m0_markers, valid[0])
    args.output.mkdir(parents=True)
    torch.save(frozen, args.output / "frozen_m0_contact.pt")
    marker_definition = extractor.definition()
    records = []
    for row in rows:
        record_path = Path(row["record"])
        candidate_path = record_path.parent / "motion_physical.pt"
        candidate = torch.load(candidate_path, map_location="cpu", weights_only=True).float()
        if candidate.shape != baseline.shape:
            raise ValueError(f"candidate shape differs from M0: {candidate_path}")
        candidate_markers = extractor(candidate)
        metrics = evaluate_frozen_contact(candidate_markers, frozen)
        destination = (
            args.output / "records" / row["method"] / row["config_id"]
            / f"dose_{row['dose']:+g}"
        )
        destination.mkdir(parents=True)
        torch.save(candidate_markers, destination / "markers.pt")
        result = {
            "method": row["method"],
            "config_id": row["config_id"],
            "sample": row["sample"],
            "seed": row["seed"],
            "dose": row["dose"],
            "record": str(record_path),
            "paired_m0_sha256": baseline_hash,
            "candidate_sha256": tensor_sha256(candidate),
            "marker_definition_sha256": marker_definition["patch_sha256"],
            "contact": metrics,
            "measured_noncontact_pass": row["measured_noncontact_pass"],
            "full_numeric_pass": bool(
                row["measured_noncontact_pass"] and metrics["full_contact_pass"]
            ),
            "visual_review_status": "PENDING_USER_FINAL_REVIEW",
        }
        (destination / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        records.append({
            key: value for key, value in result.items() if key != "contact"
        } | {"contact_status": metrics["status"], "contact_gates": metrics["gates"]})
        (args.output / "progress.json").write_text(
            json.dumps({"status": "RUNNING", "completed": len(records),
                        "expected": len(rows), "rows": records}, indent=2) + "\n",
            encoding="utf-8",
        )
    summary = {
        "status": "COMPLETED",
        "target_gate_screened_records": len(development_rows),
        "target_gate_skipped_records": len(development_rows) - len(rows),
        "paired_m0_sha256": baseline_hash,
        "marker_definition": marker_definition,
        "source_m0_only": True,
        "rows": records,
        "full_numeric_pass_count": sum(item["full_numeric_pass"] for item in records),
        "contact_not_evaluable_count": sum(item["contact_status"] != "EVALUATED" for item in records),
        "visual_review_status": "PENDING_USER_FINAL_REVIEW",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
