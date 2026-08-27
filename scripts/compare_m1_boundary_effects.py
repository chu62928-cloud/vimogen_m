#!/usr/bin/env python3
"""Compare M0/M1 pose-vs-velocity boundary combinations offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_rep.baselines import build_b0  # noqa: E402
from motion_rep.consistent_finalizer import finalize_consistent_motion_tensor  # noqa: E402
from motion_rep.unified_finalizer import finalize_motion_tensor  # noqa: E402
from scripts.evaluate_m1_pilot import angle_stats  # noqa: E402


def _collect(root: Path, subdir: str, filename: str) -> torch.Tensor:
    paths = sorted(root.glob(f"{subdir}/batch_*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"no {filename} below {root}")
    return torch.cat([torch.load(path, map_location="cpu", weights_only=True) for path in paths]).float()


def _angles(base: torch.Tensor, candidate: torch.Tensor, delta: float) -> list[float]:
    return [float(angle_stats(base[i], candidate[i], delta, heading_mode="canonical_y")["median_absolute_target_error_degrees"]) for i in range(base.shape[0])]


def audit(m1_root: Path, output: Path) -> dict[str, object]:
    mean = torch.from_numpy(np.load(ROOT / "data/meta_info/mean.npy")).float()
    std = torch.from_numpy(np.load(ROOT / "data/meta_info/std.npy")).float()
    records: list[dict[str, object]] = []
    for seed in (0, 1, 2):
        for delta in (5.0, 10.0):
            run = m1_root / f"seed_{seed:03d}" / f"delta_{int(delta):02d}deg"
            m0 = _collect(run, "m0_artifacts", "m0_raw_norm_batch.pt") * std + mean
            m1 = _collect(run, "m1_artifacts", "m1_official_norm_batch.pt") * std + mean
            b0_velocity = torch.stack([finalize_motion_tensor(item).motion for item in m0])
            b0_legacy = torch.stack([build_b0(item).motion for item in m0])
            b0_direct = torch.stack([finalize_consistent_motion_tensor(item).motion for item in m0])
            m1_velocity = torch.stack([finalize_motion_tensor(item).motion for item in m1])
            m1_direct = torch.stack([finalize_consistent_motion_tensor(item).motion for item in m1])
            combos = {
                "velocity_velocity": _angles(b0_velocity, m1_velocity, delta),
                "legacy_direct": _angles(b0_legacy, m1_direct, delta),
                "direct_direct": _angles(b0_direct, m1_direct, delta),
                "velocity_direct": _angles(b0_velocity, m1_direct, delta),
                "direct_velocity": _angles(b0_direct, m1_velocity, delta),
            }
            for index in range(m1.shape[0]):
                records.append({"seed": seed, "delta": delta, "sample_index": index, **{name: values[index] for name, values in combos.items()}})
    summary: dict[str, object] = {"record_count": len(records), "records": records, "threshold_degrees": 2.0}
    for name in ("velocity_velocity", "legacy_direct", "direct_direct", "velocity_direct", "direct_velocity"):
        values = [float(item[name]) for item in records]
        summary[name] = {
            "fail_count": sum(value > 2.0 for value in values),
            "max_error_degrees": max(values),
            "mean_error_degrees": float(np.mean(values)),
            "median_error_degrees": float(np.median(values)),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.m1_root, args.output), indent=2))


if __name__ == "__main__":
    main()
