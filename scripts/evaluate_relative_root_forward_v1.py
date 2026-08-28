"""Evaluate paired M0/G0 normalized batches for relative root-forward v1.

This is intentionally an offline evaluator: generation remains in
``train_eval_vimogen.py`` and the script never mutates earlier phase results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.relative_root_forward_v1 import compute_relative_root_forward_metrics
from sampling.relative_root_forward_guidance import PROTOCOL_NAME as V1_PROTOCOL_NAME
from sampling.relative_root_forward_guidance_v1_1 import PROTOCOL_NAME as V1_1_PROTOCOL_NAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True, help="M0 normalized [B,T,276] tensor")
    parser.add_argument("--candidate", type=Path, required=True, help="G0 normalized [B,T,276] tensor")
    parser.add_argument("--mean", type=Path, required=True)
    parser.add_argument("--std", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True, help="boolean [B,T] tensor")
    parser.add_argument("--delta-deg", type=float, required=True)
    parser.add_argument(
        "--protocol",
        choices=("v1", "v1_1"),
        default="v1",
        help="protocol label to write into the metrics artifact",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=True).float()
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True).float()
    mean = torch.from_numpy(np.load(args.mean)).float() if args.mean.suffix == ".npy" else torch.load(args.mean, map_location="cpu", weights_only=True).float()
    std = torch.from_numpy(np.load(args.std)).float() if args.std.suffix == ".npy" else torch.load(args.std, map_location="cpu", weights_only=True).float()
    mask = torch.load(args.mask, map_location="cpu", weights_only=True).bool()
    baseline_physical = baseline * std + mean
    candidate_physical = candidate * std + mean
    protocol_name = V1_1_PROTOCOL_NAME if args.protocol == "v1_1" else V1_PROTOCOL_NAME
    result = compute_relative_root_forward_metrics(
        baseline_physical,
        candidate_physical,
        mask,
        args.delta_deg,
        protocol_name=protocol_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
