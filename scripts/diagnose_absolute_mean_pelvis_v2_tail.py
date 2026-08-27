#!/usr/bin/env python3
"""Read-only tail-instability probe for one absolute-mean pelvis v2 run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_rep.consistency_v2 import _fuse_root_rotation, _moving_average
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe
from motion_rep.rotation_transform import axis_angle_to_mat3x3, mat3x3_to_axis_angle
from sampling.absolute_mean_pelvis_guidance_v2 import (
    AbsoluteMeanPelvisConfigV2,
    AbsoluteMeanPelvisGuidanceV2,
    pelvis_angle_curve,
)


def _tail(values: torch.Tensor, count: int) -> list[float]:
    return [float(value) for value in values[-count:].detach().cpu()]


def _max_step_degrees(rotation: torch.Tensor, start: int) -> float:
    if rotation.shape[0] < 2:
        return 0.0
    relative = rotation[1:] @ rotation[:-1].transpose(-1, -2)
    degrees = mat3x3_to_axis_angle(relative).norm(dim=-1) * (180.0 / math.pi)
    return float(degrees[max(start - 1, 0) :].max().detach().cpu())


def _integrate_root_velocity(direct: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    frames = [direct[:1]]
    for index in range(velocity.shape[0]):
        frames.append(velocity[index : index + 1] @ frames[-1])
    return torch.cat(frames, dim=0)


def _geodesic_degrees(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    relative = left @ right.transpose(-1, -2)
    return mat3x3_to_axis_angle(relative).norm(dim=-1) * (180.0 / math.pi)


def _probe_motion(name: str, normalized: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, tail_frames: int) -> dict:
    motion = (normalized.float() * std + mean).float()
    curve = pelvis_angle_curve(motion)
    direct = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation])
    velocity = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation_velocity])
    integrated = _integrate_root_velocity(direct, velocity)
    fused = _fuse_root_rotation(
        direct,
        motion[:, MOTION_LAYOUT.root_rotation_velocity],
        window=9,
        weight=1.0,
    )
    # Use explicit roots to distinguish direct, integrated and fused sources.
    from motion_rep.sagittal_pelvis_angle import pelvis_sagittal_tilt_degrees

    direct_curve = pelvis_sagittal_tilt_degrees(direct)
    integrated_curve = pelvis_sagittal_tilt_degrees(integrated[:-1])
    fused_curve = pelvis_sagittal_tilt_degrees(fused[:-1])
    start = max(1, motion.shape[0] - tail_frames)
    return {
        "name": name,
        "tail_frame_start": start,
        "packed_angle_tail_deg": _tail(curve, tail_frames),
        "direct_root_angle_tail_deg": _tail(direct_curve, tail_frames),
        "velocity_integrated_angle_tail_deg": _tail(integrated_curve, tail_frames),
        "fused_root_angle_tail_deg": _tail(fused_curve, tail_frames),
        "packed_tail_max_step_deg": float((curve[1:] - curve[:-1]).abs()[start - 1 :].max().detach().cpu()),
        "direct_tail_max_rotation_step_deg": _max_step_degrees(direct, start),
        "integrated_tail_max_rotation_step_deg": _max_step_degrees(integrated[:-1], start),
        "direct_vs_velocity_tail_max_deg": float(_geodesic_degrees(direct, integrated[:-1])[start:].max().detach().cpu()),
        "direct_vs_fused_tail_max_deg": float(_geodesic_degrees(direct, fused[:-1])[start:].max().detach().cpu()),
    }


def _guidance_gradient_probe(
    baseline_norm: torch.Tensor,
    candidate_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    target: float,
    tail_frames: int,
) -> dict:
    """Measure the frozen loss gradient by frame without taking an update."""

    valid_mask = torch.ones(candidate_norm.shape[:2], dtype=torch.bool)
    guidance = AbsoluteMeanPelvisGuidanceV2(
        baseline_motion_norm=baseline_norm,
        valid_mask=valid_mask,
        mean=mean,
        std=std,
        target_mean_deg=target,
        config=AbsoluteMeanPelvisConfigV2(),
    )
    candidate_norm = candidate_norm.detach().clone().requires_grad_(True)
    candidate = guidance._reconcile(candidate_norm, output_standardized=False)
    with torch.no_grad():
        baseline = guidance._reconcile(baseline_norm, output_standardized=False)
    mask = valid_mask & candidate.valid_mask & baseline.valid_mask
    count = mask.sum().clamp_min(1).to(candidate_norm.dtype)
    angle = pelvis_angle_curve(candidate.motion)
    baseline_angle = pelvis_angle_curve(baseline.motion)
    mean_angle = (angle * mask).sum(dim=-1) / count
    baseline_mean = (baseline_angle * mask).sum(dim=-1) / count
    centered = angle - mean_angle.unsqueeze(-1)
    baseline_centered = baseline_angle - baseline_mean.unsqueeze(-1)
    mean_loss = (mean_angle - target).square().mean()
    shape_loss = ((centered - baseline_centered).square() * mask).sum() / count
    candidate_standardized = (candidate.motion - mean) / std
    baseline_standardized = (baseline.motion - mean) / std
    motion_loss = ((candidate_standardized - baseline_standardized).square() * mask.unsqueeze(-1)).sum() / (count * candidate_norm.shape[-1])
    loss = mean_loss + 0.1 * shape_loss + 0.1 * motion_loss
    gradient = torch.autograd.grad(loss, candidate_norm)[0][0]
    per_frame = torch.sqrt(gradient.square().mean(dim=-1))
    root_direct = torch.sqrt(gradient[:, MOTION_LAYOUT.root_rotation].square().mean(dim=-1))
    root_velocity = torch.sqrt(gradient[:, MOTION_LAYOUT.root_rotation_velocity].square().mean(dim=-1))
    start = max(0, per_frame.numel() - tail_frames)
    return {
        "mean_angle_deg": float(mean_angle[0].detach().cpu()),
        "baseline_mean_angle_deg": float(baseline_mean[0].detach().cpu()),
        "loss": float(loss.detach().cpu()),
        "loss_terms": {
            "mean": float(mean_loss.detach().cpu()),
            "shape": float(shape_loss.detach().cpu()),
            "motion": float(motion_loss.detach().cpu()),
        },
        "tail_gradient_rms": _tail(per_frame, tail_frames),
        "tail_root_direct_gradient_rms": _tail(root_direct, tail_frames),
        "tail_root_velocity_gradient_rms": _tail(root_velocity, tail_frames),
        "tail_vs_prefix_root_direct_ratio": float(
            root_direct[start:].mean().detach().cpu()
            / root_direct[:start].mean().clamp_min(1e-12).detach().cpu()
        ),
        "tail_vs_prefix_root_velocity_ratio": float(
            root_velocity[start:].mean().detach().cpu()
            / root_velocity[:start].mean().clamp_min(1e-12).detach().cpu()
        ),
    }


def _fuse_root_rotation_hold_last(
    direct: torch.Tensor, velocity: torch.Tensor, window: int, weight: float
) -> torch.Tensor:
    """Diagnostic-only variant: hidden T+1 direct pose holds the last pose."""

    velocity_matrix = decode_rot6d_safe(velocity)
    direct_stream = torch.cat((direct, direct[-1:]), dim=0)
    frames = [direct[:1]]
    for index in range(velocity_matrix.shape[0]):
        frames.append(velocity_matrix[index : index + 1] @ frames[-1])
    velocity_stream = torch.cat(frames, dim=0)
    correction = direct_stream @ velocity_stream.transpose(-1, -2)
    correction_axis = mat3x3_to_axis_angle(correction)
    smoothed = _moving_average(correction_axis, window)
    return axis_angle_to_mat3x3(float(weight) * smoothed) @ velocity_stream


def _root_fusion_boundary_probe(motion_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, tail_frames: int) -> dict:
    """Compare current extrapolated T+1 fusion with a hold-last diagnostic variant."""

    physical = (motion_norm[0].float() * std + mean).float()
    direct_source = physical[:, MOTION_LAYOUT.root_rotation].detach().clone().requires_grad_(True)
    velocity_source = physical[:, MOTION_LAYOUT.root_rotation_velocity].detach().clone().requires_grad_(True)

    def gradient_for(fusion) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        direct = decode_rot6d_safe(direct_source)
        fused = fusion(direct, velocity_source, 9, 1.0)
        from motion_rep.sagittal_pelvis_angle import pelvis_sagittal_tilt_degrees

        objective = pelvis_sagittal_tilt_degrees(fused[:-1]).mean()
        direct_grad, velocity_grad = torch.autograd.grad(
            objective, (direct_source, velocity_source), retain_graph=True
        )
        return objective, direct_grad, velocity_grad

    current_objective, current_direct, current_velocity = gradient_for(_fuse_root_rotation)
    hold_objective, hold_direct, hold_velocity = gradient_for(_fuse_root_rotation_hold_last)
    start = max(0, physical.shape[0] - tail_frames)

    def record(objective: torch.Tensor, direct_grad: torch.Tensor, velocity_grad: torch.Tensor) -> dict:
        direct_rms = torch.sqrt(direct_grad.square().mean(dim=-1))
        velocity_rms = torch.sqrt(velocity_grad.square().mean(dim=-1))
        return {
            "mean_angle_deg": float(objective.detach().cpu()),
            "tail_direct_gradient_rms": _tail(direct_rms, tail_frames),
            "tail_velocity_gradient_rms": _tail(velocity_rms, tail_frames),
            "last_vs_penultimate_direct_gradient_ratio": float(
                (direct_rms[-1] / direct_rms[-2].clamp_min(1e-12)).detach().cpu()
            ),
            "tail_vs_prefix_direct_gradient_ratio": float(
                (direct_rms[start:].mean() / direct_rms[:start].mean().clamp_min(1e-12)).detach().cpu()
            ),
        }

    window_one_objective, window_one_direct, window_one_velocity = gradient_for(
        lambda direct, velocity, _window, weight: _fuse_root_rotation(
            direct, velocity, window=1, weight=weight
        )
    )
    return {
        "current_window9_extrapolated": record(current_objective, current_direct, current_velocity),
        "hold_last_window9_diagnostic": record(hold_objective, hold_direct, hold_velocity),
        "current_extrapolated_window1_diagnostic": record(
            window_one_objective, window_one_direct, window_one_velocity
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tail-frames", type=int, default=8)
    parser.add_argument("--target-mean-deg", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.tail_frames < 2:
        raise ValueError("--tail-frames must be >= 2")

    mean = torch.from_numpy(np.load(ROOT / "data/meta_info/mean.npy")).float()
    std = torch.from_numpy(np.load(ROOT / "data/meta_info/std.npy")).float()
    artifacts = args.run_root / "guided_artifacts" / "batch_000"
    candidates = {
        "m0_raw": args.run_root / "m0_artifacts" / "batch_000" / "m0_raw_norm_batch.pt",
        "m0_official": args.run_root / "m0_artifacts" / "batch_000" / "m0_official_norm_batch.pt",
        "guided_raw": artifacts / "guided_raw_norm_batch.pt",
        "guided_official": artifacts / "guided_official_norm_batch.pt",
        "g0": artifacts / "g0_norm_batch.pt",
        "g1": artifacts / "g1_norm_batch.pt",
    }
    report = {"run_root": str(args.run_root), "tail_frames": args.tail_frames, "motions": {}}
    for name, path in candidates.items():
        if not path.exists():
            continue
        tensor = torch.load(path, map_location="cpu", weights_only=True).float()
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError(f"{path} must be [1,T,276], got {tuple(tensor.shape)}")
        report["motions"][name] = _probe_motion(name, tensor[0], mean, std, args.tail_frames)
    baseline = torch.load(candidates["m0_raw"], map_location="cpu", weights_only=True).float()
    report["root_fusion_boundary_probe"] = _root_fusion_boundary_probe(
        baseline, mean, std, args.tail_frames
    )
    for name in ("m0_raw", "guided_raw"):
        path = candidates[name]
        if path.exists():
            candidate = torch.load(path, map_location="cpu", weights_only=True).float()
            report.setdefault("frozen_loss_gradient", {})[name] = _guidance_gradient_probe(
                baseline, candidate, mean, std, args.target_mean_deg, args.tail_frames
            )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
