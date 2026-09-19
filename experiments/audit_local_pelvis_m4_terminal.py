#!/usr/bin/env python3
"""Separate M4's propagated sampling endpoint from its terminal correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.local_pelvis_metrics import evaluate_local_pelvis_metrics
from guidance.base import tensor_sha256
from motion_rep.pose_authority import authority_project


def _one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} below {root}, found {len(matches)}")
    return matches[0]


def _physical(motion: torch.Tensor, valid: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return authority_project(
        motion.float(), valid_mask=valid, mean=mean, std=std,
        input_standardized=True, output_standardized=False,
    ).motion


def audit_one(run_root: Path, output: Path) -> list[dict]:
    meta = json.loads((run_root / "run_record.json").read_text(encoding="utf-8"))
    if meta.get("method") != "M4" or meta.get("task") != "relative_pelvis_s1":
        raise ValueError(f"not an M4 S1 generation: {run_root}")
    archive = torch.load(
        _one(run_root, "trainer/**/batch_000/mbench_raw_norm_batch.pt"),
        map_location="cpu", weights_only=True,
    )
    artifact = run_root / "guided_artifacts/batch_000"
    valid = archive["motion_mask"].bool()
    mean, std = archive["motion_mean"].float(), archive["motion_std"].float()
    baseline = _physical(
        torch.load(artifact / "m0_authority_norm_batch.pt", map_location="cpu", weights_only=True),
        valid, mean, std,
    )
    before = _physical(
        torch.load(artifact / "guided_official_norm_batch.pt", map_location="cpu", weights_only=True),
        valid, mean, std,
    )
    after = _physical(
        torch.load(artifact / "g0_norm_batch.pt", map_location="cpu", weights_only=True),
        valid, mean, std,
    )
    dose = float(meta["target_dose_deg"])
    pre_metrics = evaluate_local_pelvis_metrics(before, baseline, valid, dose)
    post_metrics = evaluate_local_pelvis_metrics(after, baseline, valid, dose)
    ids = [str(item) for item in archive["sample_ids"]]
    output.mkdir(parents=True)
    rows = []
    for index, sample_id in enumerate(ids):
        before_row = pre_metrics["per_sequence"][index]
        after_row = post_metrics["per_sequence"][index]
        stem = f"p{sample_id}_s{meta['seed']}_d{dose:+g}"
        before_path = output / f"{stem}_before_terminal.pt"
        after_path = output / f"{stem}_after_terminal.pt"
        torch.save(before[index:index + 1], before_path)
        torch.save(after[index:index + 1], after_path)
        rows.append({
            "sample_id": sample_id,
            "seed": int(meta["seed"]),
            "dose_deg": dose,
            "config_id": meta["config_id"],
            "source_run": str(run_root),
            "source_m0_sha256": tensor_sha256(baseline[index:index + 1]),
            "before_terminal_motion": str(before_path),
            "after_terminal_motion": str(after_path),
            "before_terminal": before_row,
            "after_terminal": after_row,
            "sampling_endpoint_target_hit": before_row["target_hit"],
            "post_terminal_target_hit": after_row["target_hit"],
            "terminal_only_success": bool(
                not before_row["target_hit"] and after_row["target_hit"]
            ),
        })
    (output / "audit.json").write_text(
        json.dumps({"status": "COMPLETED", "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = sorted(args.generation_root.glob("M4/*/seed_*/dose_*/attempt_*/run_record.json"))
    if not records:
        raise FileNotFoundError(f"no M4 generation records below {args.generation_root}")
    args.output.mkdir(parents=True)
    rows = []
    for path in records:
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("status") != "COMPLETED_GENERATION_PENDING_EVALUATION":
            continue
        destination = (
            args.output / meta["config_id"] / path.parent.parent.parent.name
            / path.parent.parent.name / path.parent.name
        )
        rows.extend(audit_one(path.parent, destination))
    summary = {
        "status": "COMPLETED",
        "rows": rows,
        "sampling_endpoint_hit_count": sum(row["sampling_endpoint_target_hit"] for row in rows),
        "post_terminal_hit_count": sum(row["post_terminal_target_hit"] for row in rows),
        "terminal_only_success_count": sum(row["terminal_only_success"] for row in rows),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
