#!/usr/bin/env python3
"""Evaluate all frozen-development-set window_mid M1 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from smplx import SMPLX


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_rep.baselines import build_b0  # noqa: E402
from scripts.evaluate_m1_pilot import load_norm, one_condition, tensor_sha256  # noqa: E402


def _collect(root: Path, subdir: str, name: str) -> torch.Tensor:
    paths = sorted((root / subdir).glob(f"batch_*/{name}"))
    if not paths:
        raise FileNotFoundError(f"missing {name} under {root / subdir}")
    return torch.cat([torch.load(path, weights_only=True, map_location="cpu") for path in paths])


def _sample_ids(input_path: Path) -> list[str]:
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    return [str(row["sample_id"]) for row in rows]


def evaluate(
    input_path: Path,
    m0_root: Path,
    m1_root: Path,
    output_path: Path,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    deltas: tuple[float, ...] = (5.0, 10.0),
) -> dict[str, Any]:
    mean = torch.from_numpy(np.load(ROOT / "data/meta_info/mean.npy")).float()
    std = torch.from_numpy(np.load(ROOT / "data/meta_info/std.npy")).float()
    ids = _sample_ids(input_path)
    model = SMPLX(
        model_path=str(ROOT / "data/body_models/smplx"),
        gender="neutral",
        use_pca=False,
        num_betas=10,
        batch_size=100,
    ).eval()
    report: dict[str, Any] = {
        "status": "VERIFIED_M1_DEVSET_WINDOW_MID",
        "protocol": {
            "variant": "window_mid",
            "samples": len(ids),
            "seeds": list(seeds),
            "target_deltas_degrees": list(deltas),
            "same_sample_level_z0": True,
            "m0_frozen": True,
            "heading_mode": "canonical_y",
            "sigma_window": [0.25, 0.65],
            "max_correction_rms": 0.05,
            "strict_per_sample_error_threshold_degrees": 2.0,
            "official_metrics": {"fid": False, "r_precision": False, "mbench_physics": False},
        },
        "runs": {},
    }
    all_gate_errors: list[float] = []
    for seed in seeds:
        m0_seed = m0_root / f"seed_{seed:03d}"
        m0_official_norm = _collect(m0_seed, "artifacts", "m0_official_norm_batch.pt")
        m0_raw_norm = _collect(m0_seed, "artifacts", "m0_raw_norm_batch.pt")
        z0 = _collect(m0_seed, "artifacts", "z0_replayed.pt")
        m0_raw = m0_raw_norm.float() * std + mean
        baseline = torch.stack([build_b0(sample).motion for sample in m0_raw], dim=0)
        for delta in deltas:
            label = f"seed_{seed:03d}_delta_{int(delta):02d}deg"
            run = m1_root / f"seed_{seed:03d}" / f"delta_{int(delta):02d}deg"
            m1_raw_norm = _collect(run, "m1_artifacts", "m1_raw_norm_batch.pt")
            m1_official_norm = _collect(run, "m1_artifacts", "m1_official_norm_batch.pt")
            m1_raw = m1_raw_norm.float() * std + mean
            m1_official = m1_official_norm.float() * std + mean
            m0_run_official = _collect(run, "m0_artifacts", "m0_official_norm_batch.pt")
            m0_run_raw = _collect(run, "m0_artifacts", "m0_raw_norm_batch.pt")
            z0_run = _collect(run, "m0_artifacts", "z0_replayed.pt")
            result = {
                "path": str(run),
                "seed": seed,
                "target_delta_degrees": delta,
                "sample_ids": ids,
                "z0_sha256": tensor_sha256(z0_run),
                "z0_bitwise_equal_m0_run": bool(torch.equal(z0_run, z0)),
                "m0_raw_bitwise_equal_m0_run": bool(torch.equal(m0_run_raw, m0_raw_norm)),
                "m0_official_bitwise_equal_m0_run": bool(torch.equal(m0_run_official, m0_official_norm)),
                "m0_baseline": one_condition(baseline, baseline, 0.0, model, heading_mode="canonical_y"),
                "m1_raw": one_condition(baseline, m1_raw, delta, model, heading_mode="canonical_y"),
                "m1_official": one_condition(baseline, m1_official, delta, model, heading_mode="canonical_y"),
            }
            errors = [
                float(item["median_absolute_target_error_degrees"])
                for item in result["m1_official"]["angle"]
            ]
            all_gate_errors.extend(errors)
            result["strict_per_sample_gate"] = {
                "threshold_degrees": 2.0,
                "max_median_absolute_target_error_degrees": max(errors),
                "mean_median_absolute_target_error_degrees": float(np.mean(errors)),
                "passed": bool(max(errors) <= 2.0),
            }
            report["runs"][label] = result
    report["entry_gate"] = {
        "threshold_degrees": 2.0,
        "unit_count": len(all_gate_errors),
        "max_median_absolute_target_error_degrees": max(all_gate_errors),
        "mean_median_absolute_target_error_degrees": float(np.mean(all_gate_errors)),
        "passed": bool(max(all_gate_errors) <= 2.0),
        "status": "M1_ENTRY_GATE_PASSED" if max(all_gate_errors) <= 2.0 else "M1_ENTRY_GATE_FAILED",
    }
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--m1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.input, args.m0_root, args.m1_root, args.output), indent=2))


if __name__ == "__main__":
    main()
