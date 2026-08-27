#!/usr/bin/env python3
"""Offline before/after evaluation for the unified M1 finalization boundary."""

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
from motion_rep.unified_finalizer import finalize_motion_tensor  # noqa: E402
from scripts.evaluate_m1_pilot import load_norm, one_condition, tensor_sha256  # noqa: E402


def _collect(root: Path, subdir: str, name: str) -> torch.Tensor:
    paths = sorted((root / subdir).glob(f"batch_*/{name}"))
    if not paths:
        raise FileNotFoundError(f"missing {name} under {root / subdir}")
    return torch.cat([torch.load(path, weights_only=True, map_location="cpu") for path in paths])


def evaluate(
    m0_root: Path,
    m1_root: Path,
    output_path: Path,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    deltas: tuple[float, ...] = (5.0, 10.0),
) -> dict[str, Any]:
    mean = torch.from_numpy(np.load(ROOT / "data/meta_info/mean.npy")).float()
    std = torch.from_numpy(np.load(ROOT / "data/meta_info/std.npy")).float()
    model = SMPLX(
        model_path=str(ROOT / "data/body_models/smplx"),
        gender="neutral",
        use_pca=False,
        num_betas=10,
        batch_size=100,
    ).eval()
    report: dict[str, Any] = {
        "status": "VERIFIED_M1_UNIFIED_FINALIZE_OFFLINE",
        "protocol": {
            "revision": "m1_unified_finalize_v1",
            "common_finalizer": "motion_rep.unified_finalizer.finalize_motion_tensor",
            "boundary": "physical_Tplus1_to_Tx276",
            "seeds": list(seeds),
            "target_deltas_degrees": list(deltas),
            "same_sample_level_z0": True,
            "model_rerun": False,
            "strict_per_sample_threshold_degrees": 2.0,
            "turning_samples_kept": True,
        },
        "runs": {},
    }
    before_errors: list[float] = []
    after_errors: list[float] = []
    finalizer_deltas: list[float] = []
    for seed in seeds:
        m0_seed = m0_root / f"seed_{seed:03d}"
        m0_raw_norm = _collect(m0_seed, "artifacts", "m0_raw_norm_batch.pt")
        m0_raw = m0_raw_norm.float() * std + mean
        common_b0 = torch.stack([finalize_motion_tensor(sample).motion for sample in m0_raw])
        legacy_b0 = torch.stack([build_b0(sample).motion for sample in m0_raw])
        legacy_vs_common = torch.abs(legacy_b0 - common_b0)
        for delta in deltas:
            label = f"seed_{seed:03d}_delta_{int(delta):02d}deg"
            run = m1_root / f"seed_{seed:03d}" / f"delta_{int(delta):02d}deg"
            m1_raw = _collect(run, "m1_artifacts", "m1_raw_norm_batch.pt").float() * std + mean
            m1_official = _collect(run, "m1_artifacts", "m1_official_norm_batch.pt").float() * std + mean
            unified_raw = torch.stack([finalize_motion_tensor(sample).motion for sample in m1_raw])
            unified_official = torch.stack([finalize_motion_tensor(sample).motion for sample in m1_official])
            before = one_condition(legacy_b0, m1_official, delta, model, heading_mode="canonical_y")
            after = one_condition(common_b0, unified_official, delta, model, heading_mode="canonical_y")
            raw_after = one_condition(common_b0, unified_raw, delta, model, heading_mode="canonical_y")
            before_values = [float(x["median_absolute_target_error_degrees"]) for x in before["angle"]]
            after_values = [float(x["median_absolute_target_error_degrees"]) for x in after["angle"]]
            before_errors.extend(before_values)
            after_errors.extend(after_values)
            finalizer_deltas.extend(
                float(torch.sqrt((unified_official[i] - m1_official[i]).square().mean()).item())
                for i in range(unified_official.shape[0])
            )
            report["runs"][label] = {
                "m0_z0_reused": True,
                "legacy_b0_vs_common_max_abs": float(legacy_vs_common.max().item()),
                "legacy_b0_vs_common_rms": float(torch.sqrt(legacy_vs_common.square().mean()).item()),
                "before_legacy_b0_m1_official": before,
                "after_common_b0_m1_unified_official": after,
                "after_common_b0_m1_unified_raw": raw_after,
                "before_gate": {
                    "max_error_degrees": max(before_values),
                    "mean_error_degrees": float(np.mean(before_values)),
                    "passed": bool(max(before_values) <= 2.0),
                },
                "after_gate": {
                    "max_error_degrees": max(after_values),
                    "mean_error_degrees": float(np.mean(after_values)),
                    "passed": bool(max(after_values) <= 2.0),
                },
                "m1_official_unified_change_rms_mean": float(np.mean(finalizer_deltas[-len(after_values):])),
            }
    report["summary"] = {
        "unit_count": len(before_errors),
        "before_max_error_degrees": max(before_errors),
        "before_mean_error_degrees": float(np.mean(before_errors)),
        "before_passed": bool(max(before_errors) <= 2.0),
        "after_max_error_degrees": max(after_errors),
        "after_mean_error_degrees": float(np.mean(after_errors)),
        "after_passed": bool(max(after_errors) <= 2.0),
        "unified_m1_official_change_rms_mean": float(np.mean(finalizer_deltas)),
        "status": "M1_UNIFIED_FINALIZE_ENTRY_GATE_PASSED" if max(after_errors) <= 2.0 else "M1_UNIFIED_FINALIZE_ENTRY_GATE_FAILED",
    }
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--m1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.m0_root, args.m1_root, args.output), indent=2))


if __name__ == "__main__":
    main()
