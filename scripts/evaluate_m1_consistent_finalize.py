#!/usr/bin/env python3
"""Evaluate the pose-authoritative M1 finalization on existing tensors only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from smplx import SMPLX

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_rep.consistent_finalizer import finalize_consistent_motion_tensor  # noqa: E402
from scripts.evaluate_m1_pilot import one_condition  # noqa: E402


def _collect(root: Path, subdir: str, filename: str) -> torch.Tensor:
    paths = sorted(root.glob(f"{subdir}/batch_*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"no {filename} below {root}")
    return torch.cat([torch.load(path, map_location="cpu", weights_only=True) for path in paths]).float()


def _finalize_batch(motion: torch.Tensor) -> torch.Tensor:
    return torch.stack([finalize_consistent_motion_tensor(item).motion for item in motion])


def evaluate(m1_root: Path, output_root: Path, *, seeds: tuple[int, ...] = (0, 1, 2), deltas: tuple[float, ...] = (5.0, 10.0)) -> dict[str, object]:
    mean = torch.from_numpy(np.load(ROOT / "data/meta_info/mean.npy")).float()
    std = torch.from_numpy(np.load(ROOT / "data/meta_info/std.npy")).float()
    model = SMPLX(model_path=str(ROOT / "data/body_models/smplx"), gender="neutral", use_pca=False, num_betas=10, batch_size=100).eval()
    records: list[dict[str, object]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for delta in deltas:
            run = m1_root / f"seed_{seed:03d}" / f"delta_{int(delta):02d}deg"
            m0 = _collect(run, "m0_artifacts", "m0_raw_norm_batch.pt") * std + mean
            m1 = _collect(run, "m1_artifacts", "m1_official_norm_batch.pt") * std + mean
            b0 = _finalize_batch(m0)
            m1_consistent = _finalize_batch(m1)
            save_dir = output_root / f"seed_{seed:03d}" / f"delta_{int(delta):02d}deg"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(b0, save_dir / "b0_consistent_physical.pt")
            torch.save(m1_consistent, save_dir / "m1_consistent_physical.pt")
            evaluated = one_condition(b0, m1_consistent, delta, model, heading_mode="canonical_y")
            for index, item in enumerate(evaluated["angle"]):
                records.append({
                    "seed": seed,
                    "delta": delta,
                    "sample_index": index,
                    "median_absolute_target_error_degrees": float(item["median_absolute_target_error_degrees"]),
                    "median_shift_degrees": float(item["median_shift_degrees"]),
                    "valid_frame_count": int(item["count"]),
                })
    errors = [float(item["median_absolute_target_error_degrees"]) for item in records]
    failures = [item for item in records if item["median_absolute_target_error_degrees"] > 2.0]
    report = {
        "status": "M1_CONSISTENT_FINALIZE_ENTRY_GATE_PASSED" if not failures else "M1_CONSISTENT_FINALIZE_ENTRY_GATE_FAILED",
        "protocol": {
            "revision": "m1_consistent_finalize_v1",
            "boundary": "direct_pose_authoritative_Tplus1_to_Tx276",
            "model_rerun": False,
            "same_existing_m1_tensors": True,
            "turning_samples_kept": True,
            "threshold_degrees": 2.0,
        },
        "summary": {
            "unit_count": len(records),
            "fail_count": len(failures),
            "max_error_degrees": max(errors),
            "mean_error_degrees": float(np.mean(errors)),
            "median_error_degrees": float(np.median(errors)),
        },
        "failures": failures,
        "records": records,
    }
    (output_root / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.m1_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
