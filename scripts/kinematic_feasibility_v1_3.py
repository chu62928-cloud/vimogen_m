#!/usr/bin/env python3
"""Offline v1.3 feasibility check on frozen M0 endpoints.

This deliberately does not call the diffusion model.  It answers whether the
root-plus-spine parameterisation can satisfy the angular and temporal budgets
on the two frozen smoke samples before sampler-transfer effects are involved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from evaluation.relative_root_forward_v1 import compute_relative_root_forward_metrics
from sampling.relative_root_forward_guidance_v1_3 import (
    PROTOCOL_NAME,
    ShadowPoseHierarchicalConfig,
    ShadowPoseHierarchicalRootForwardGuidance,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0", type=Path, required=True)
    parser.add_argument("--mean", type=Path, required=True)
    parser.add_argument("--std", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--doses", type=float, nargs="+", default=[-10.0, -5.0, 5.0, 10.0])
    args = parser.parse_args()

    m0 = torch.load(args.m0, map_location="cpu", weights_only=True).float()
    mean = torch.from_numpy(np.load(args.mean)).float()
    std = torch.from_numpy(np.load(args.std)).float()
    mask = torch.ones(m0.shape[:2], dtype=torch.bool)
    results = []
    for dose in args.doses:
        strategy = ShadowPoseHierarchicalRootForwardGuidance(
            baseline_motion_norm=m0,
            valid_mask=mask,
            mean=mean,
            std=std,
            target_delta_deg=float(dose),
            config=ShadowPoseHierarchicalConfig(enabled=True, sigma_min=0.0),
        )
        physical = strategy._physical(m0)
        proposal = {}
        stacked = torch.zeros((*mask.shape, 5), dtype=physical.dtype)
        solver_iterations = 0
        # The per-step trust region is intentionally smaller than the 10
        # degree dose.  Repeat the closed-loop solve on the updated physical
        # shadow until the frozen target is reached.
        for solver_iterations in range(1, 9):
            pitch, heading, beta, proposal = strategy._iterative_proposal(physical, mask)
            stacked = torch.cat((pitch.unsqueeze(-1), heading.unsqueeze(-1), beta), dim=-1)
            stacked = strategy._temporal_project(stacked, mask)
            candidate = strategy._build_candidate(
                physical,
                stacked[..., 0],
                stacked[..., 1],
                stacked[..., 2:],
            )
            physical = candidate
            fwd = compute_relative_root_forward_metrics(
                strategy.baseline_physical,
                physical,
                mask,
                float(dose),
                protocol_name=PROTOCOL_NAME,
                skeleton=strategy.skeleton,
            )
            p95 = max(row["p95_absolute_error_deg"] for row in fwd["per_sample"])
            if p95 <= 1e-3:
                break
        candidate_physical = physical
        metrics = compute_relative_root_forward_metrics(
            strategy.baseline_physical,
            candidate_physical,
            mask,
            float(dose),
            protocol_name=PROTOCOL_NAME,
            skeleton=strategy.skeleton,
        )
        results.append({
            "dose": float(dose),
            "proposal": proposal,
            "metrics": metrics,
            "spine_sum_max_deg": float(stacked[..., 2:].abs().sum(-1).max()),
            "temporal_step_max_deg": float(torch.linalg.vector_norm(stacked[:, 1:] - stacked[:, :-1], dim=-1).max()),
            "solver_iterations": int(solver_iterations),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"protocol": PROTOCOL_NAME, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
