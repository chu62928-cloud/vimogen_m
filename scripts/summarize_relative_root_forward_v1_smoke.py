#!/usr/bin/env python3
"""Summarise the server-side relative-root-forward v1 smoke matrix."""

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

from evaluation.relative_root_forward_v1 import (
    compute_relative_root_forward_metrics,
    dose_monotonicity,
)


def _load_g0(root: Path, delta: int) -> torch.Tensor:
    sign = "+" if delta >= 0 else ""
    path = (
        root
        / "runs"
        / "smoke"
        / "seed_000"
        / f"delta_{sign}{delta}deg"
        / "attempt_01"
        / "guided_artifacts"
        / "batch_000"
        / "g0_norm_batch.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, weights_only=True).float()


def _load_m0(root: Path, delta: int) -> torch.Tensor:
    sign = "+" if delta >= 0 else ""
    path = (
        root
        / "runs"
        / "smoke"
        / "seed_000"
        / f"delta_{sign}{delta}deg"
        / "attempt_01"
        / "guided_artifacts"
        / "batch_000"
        / "m0_consistent_norm_batch.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, weights_only=True).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/phase7/relative_root_forward_v1"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    # The default root is <repo>/results/phase7/<protocol>; resolve data paths
    # relative to the repository instead of relying on the caller's cwd.
    repo = root.resolve().parents[2]
    mean = torch.from_numpy(np.load(repo / "data/meta_info/mean.npy")).float()
    std = torch.from_numpy(np.load(repo / "data/meta_info/std.npy")).float()

    m0 = {delta: _load_m0(root, delta) for delta in (0, 5, -5, 10, -10)}
    g0 = {delta: _load_g0(root, delta) for delta in (0, 5, -5, 10, -10)}
    mask = torch.ones(m0[0].shape[:2], dtype=torch.bool)
    result = {
        "protocol": "vimogen_relative_root_forward_v1_pose_authoritative",
        "runs": {},
        "m0_reproducibility": {},
    }
    for delta in (0, 5, -5, 10, -10):
        result["runs"][str(delta)] = compute_relative_root_forward_metrics(
            m0[delta] * std + mean,
            g0[delta] * std + mean,
            mask,
            float(delta),
        )
        result["runs"][str(delta)]["bitwise_zero"] = (
            bool(torch.equal(g0[delta], m0[delta])) if delta == 0 else None
        )
        result["m0_reproducibility"][str(delta)] = {
            "bitwise_vs_delta0": bool(torch.equal(m0[delta], m0[0])),
            "max_abs_norm_diff_vs_delta0": float((m0[delta] - m0[0]).abs().max()),
        }
    physical_m0 = m0[0] * std + mean
    result["dose_monotonicity"] = {
        "positive": dose_monotonicity(
            physical_m0,
            g0[5] * std + mean,
            g0[10] * std + mean,
            mask,
        ),
        "negative": dose_monotonicity(
            physical_m0,
            g0[-5] * std + mean,
            g0[-10] * std + mean,
            mask,
        ),
    }
    output = args.output or root / "summaries/smoke_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    for delta in (0, 5, -5, 10, -10):
        rows = result["runs"][str(delta)]["per_sample"]
        tail = result["runs"][str(delta)]["tail_safety"]["per_sample"]
        whole = result["runs"][str(delta)]["whole_body"]
        print(
            f"delta={delta:+d} "
            f"mean_abs={[round(row['mean_absolute_error_deg'], 4) for row in rows]} "
            f"vector_p95={[round(row['forward_vector_error_p95_deg'], 4) for row in rows]} "
            f"tail={[row['tail_pass'] for row in tail]} "
            f"q_rigid={whole['q_rigid']:.4f}"
        )
    print("m0_reproducibility=", json.dumps(result["m0_reproducibility"]))
    print("dose_monotonicity=", json.dumps(result["dose_monotonicity"]))


if __name__ == "__main__":
    main()
