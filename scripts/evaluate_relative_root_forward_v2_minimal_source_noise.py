#!/usr/bin/env python3
"""Offline external audit for one v2 source-noise run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.relative_root_forward_v1 import compute_relative_root_forward_metrics


def _find_one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern}, found {len(matches)}")
    return matches[0]


def evaluate(run_root: Path, target_delta_deg: float) -> dict:
    m0_path = _find_one(run_root, "m0_artifacts/batch_*/m0_official_norm_batch.pt")
    archive_path = _find_one(
        run_root, "trainer/test_visualization/*/batch_*/mbench_raw_norm_batch.pt"
    )
    m0 = torch.load(m0_path, map_location="cpu", weights_only=True)
    generated = torch.load(archive_path, map_location="cpu", weights_only=True)
    baseline_norm = m0.float()
    candidate_norm = generated["motion_norm"].float()
    valid_mask = generated["motion_mask"].bool()
    mean = generated["motion_mean"].float()
    std = generated["motion_std"].float()
    baseline = baseline_norm * std[:, None, :] + mean[:, None, :]
    candidate = candidate_norm * std[:, None, :] + mean[:, None, :]
    metrics = compute_relative_root_forward_metrics(
        baseline,
        candidate,
        valid_mask,
        target_delta_deg,
        protocol_name="vimogen_relative_root_forward_v2_minimal_source_noise",
    )
    result = {
        "protocol": "vimogen_relative_root_forward_v2_minimal_source_noise",
        "run_root": str(run_root),
        "baseline_path": str(m0_path),
        "candidate_path": str(archive_path),
        "target_delta_deg": float(target_delta_deg),
        "metrics": metrics,
    }
    output = run_root / "source_noise_external_evaluation.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["output"] = str(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--target-delta-deg", type=float, required=True)
    args = parser.parse_args()
    result = evaluate(args.run_root, args.target_delta_deg)
    row = result["metrics"]["per_sample"][0]
    print(json.dumps({
        "output": result["output"],
        "forward_p95_deg": row["forward_vector_error_p95_deg"],
        "pitch_p95_deg": row["p95_absolute_error_deg"],
        "dose_sign_correct": row["dose_sign_correct"],
        "tail": result["metrics"]["tail_safety"]["per_sample"][0],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
