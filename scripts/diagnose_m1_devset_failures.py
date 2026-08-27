#!/usr/bin/env python3
"""Read-only diagnostics for valid M1 development-set failures.

This script never reruns a model, edits a tensor, or removes a sample.  It
summarises the frame-wise model-space proxy-angle error for entries that
already failed the pre-registered per-unit gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_m1_pilot import angle_curve, load_norm, tensor_sha256  # noqa: E402


def _collect(root: Path, subdir: str, name: str) -> torch.Tensor:
    paths = sorted((root / subdir).glob(f"batch_*/{name}"))
    if not paths:
        raise FileNotFoundError(f"missing {name} under {root / subdir}")
    return torch.cat([torch.load(path, weights_only=True, map_location="cpu") for path in paths])


def _summarise_curve(baseline: torch.Tensor, candidate: torch.Tensor, target: float) -> dict[str, Any]:
    base_angle, base_valid = angle_curve(baseline, heading_mode="canonical_y")
    cand_angle, cand_valid = angle_curve(candidate, heading_mode="canonical_y")
    valid = base_valid & cand_valid
    shift = cand_angle - base_angle
    error = shift - float(target)
    selected_shift = shift[valid].float()
    selected_error = error[valid].float()
    if selected_error.numel() == 0:
        raise ValueError("failure diagnostic found no valid frames")
    return {
        "valid_frame_count": int(selected_error.numel()),
        "target_delta_degrees": float(target),
        "baseline_angle_median_degrees": float(base_angle[valid].median().item()),
        "baseline_angle_p05_p95_degrees": [
            float(torch.quantile(base_angle[valid], 0.05).item()),
            float(torch.quantile(base_angle[valid], 0.95).item()),
        ],
        "shift_median_degrees": float(selected_shift.median().item()),
        "shift_min_max_degrees": [
            float(selected_shift.min().item()),
            float(selected_shift.max().item()),
        ],
        "absolute_error_median_degrees": float(selected_error.abs().median().item()),
        "absolute_error_p95_degrees": float(torch.quantile(selected_error.abs(), 0.95).item()),
        "absolute_error_max_degrees": float(selected_error.abs().max().item()),
        "frames_absolute_error_gt_2_degrees": int((selected_error.abs() > 2.0).sum().item()),
        "frames_absolute_error_gt_5_degrees": int((selected_error.abs() > 5.0).sum().item()),
    }


def diagnose(
    metrics_path: Path,
    input_path: Path,
    m0_root: Path,
    m1_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = {str(row["sample_id"]): row for row in json.loads(input_path.read_text(encoding="utf-8"))}
    mean = torch.from_numpy(np.load(ROOT / "data/meta_info/mean.npy")).float()
    std = torch.from_numpy(np.load(ROOT / "data/meta_info/std.npy")).float()
    failures: list[dict[str, Any]] = []
    for label, run_metrics in metrics["runs"].items():
        for sample_id, angle in zip(run_metrics["sample_ids"], run_metrics["m1_official"]["angle"]):
            if float(angle["median_absolute_target_error_degrees"]) <= 2.0:
                continue
            seed = int(run_metrics["seed"])
            delta = float(run_metrics["target_delta_degrees"])
            run = m1_root / f"seed_{seed:03d}" / f"delta_{int(delta):02d}deg"
            m0 = _collect(m0_root / f"seed_{seed:03d}", "artifacts", "m0_raw_norm_batch.pt").float() * std + mean
            m1_raw = _collect(run, "m1_artifacts", "m1_raw_norm_batch.pt").float() * std + mean
            m1_official = _collect(run, "m1_artifacts", "m1_official_norm_batch.pt").float() * std + mean
            index = list(run_metrics["sample_ids"]).index(sample_id)
            row = rows[str(sample_id)]
            item = {
                "run_label": label,
                "seed": seed,
                "target_delta_degrees": delta,
                "sample_id": str(sample_id),
                "category": row.get("category"),
                "prompt": row.get("prompt", row.get("motion_text_annot")),
                "source_motion_path": row.get("source_motion_path"),
                "official_gate_error_degrees": float(angle["median_absolute_target_error_degrees"]),
                "m0_raw_sha256": tensor_sha256(m0[index]),
                "m1_raw_curve": _summarise_curve(m0[index], m1_raw[index], delta),
                "m1_official_curve": _summarise_curve(m0[index], m1_official[index], delta),
                "video_paths": sorted(str(path) for path in run.rglob(f"{sample_id}/*.mp4")),
            }
            failures.append(item)
    report = {
        "status": "VERIFIED_M1_FAILURES_READ_ONLY",
        "threshold_degrees": 2.0,
        "failure_count": len(failures),
        "rerun_performed": False,
        "parameter_change": False,
        "failures": failures,
    }
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--m1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.metrics, args.input, args.m0_root, args.m1_root, args.output), indent=2))


if __name__ == "__main__":
    main()
