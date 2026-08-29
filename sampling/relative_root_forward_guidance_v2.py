"""Minimal source-noise root-forward control (v2).

The frozen ViMoGen sampler is treated as ``G(z)``.  This module optimizes
only a source-noise offset and never edits a pose channel during denoising.
The root objective is used to find feasible candidates; among observed
feasible candidates, source-noise RMS is the sole tie-breaker.  Trunk, leg,
foot and naturalness quantities are intentionally not part of the gradient.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional

import torch

from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe
from motion_rep.pose_authority import (
    _root_forward,
    authority_project,
    forward_vector_loss,
    prepare_targets,
)
from sampling.differentiable_flow_sampler import (
    DifferentiableSamplerConfig,
    differentiable_generate,
)
from sampling.flow_sampler import FlowSampleResult


PROTOCOL_NAME = "vimogen_relative_root_forward_v2_minimal_source_noise"


@dataclass(frozen=True)
class MinimalSourceNoiseConfig:
    iterations: int = 12
    step_rms: float = 0.03
    max_delta_rms: float = 1.0
    line_search_steps: int = 4
    feasible_pitch_mae_deg: float = 1.0
    feasible_forward_p95_deg: float = 2.0
    forward_loss_temperature: float = 5.0
    use_gradient_checkpointing: bool = True

    def validate(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if not 0.0 < self.step_rms <= self.max_delta_rms:
            raise ValueError("step_rms must lie in (0,max_delta_rms]")
        if self.max_delta_rms <= 0.0:
            raise ValueError("max_delta_rms must be positive")
        if self.line_search_steps < 0:
            raise ValueError("line_search_steps must be non-negative")
        if self.feasible_pitch_mae_deg <= 0.0 or self.feasible_forward_p95_deg <= 0.0:
            raise ValueError("feasibility tolerances must be positive")
        if self.forward_loss_temperature <= 0.0:
            raise ValueError("forward_loss_temperature must be positive")


@dataclass
class SourceNoiseOptimizationResult:
    protocol: str
    motion_norm: torch.Tensor
    baseline_norm: torch.Tensor
    optimized_norm: torch.Tensor
    source_delta: torch.Tensor
    summary: dict


def _physical_motion(
    normalized_motion: torch.Tensor,
    motion_mean: torch.Tensor,
    motion_std: torch.Tensor,
) -> torch.Tensor:
    mean = motion_mean.float()
    std = motion_std.float()
    if mean.ndim == 1:
        mean = mean[None, None, :]
        std = std[None, None, :]
    elif mean.ndim == 2:
        mean = mean[:, None, :]
        std = std[:, None, :]
    if mean.ndim != 3 or mean.shape[-1] != MOTION_LAYOUT.total_dim:
        raise ValueError("motion_mean/motion_std must broadcast to [B,T,276]")
    return normalized_motion.float() * std + mean


def _target_and_metrics(
    *,
    baseline_norm: torch.Tensor,
    candidate_norm: torch.Tensor,
    valid_mask: torch.Tensor,
    motion_mean: torch.Tensor,
    motion_std: torch.Tensor,
    target_delta_deg: float,
) -> tuple[torch.Tensor, dict]:
    baseline = _physical_motion(baseline_norm.detach(), motion_mean, motion_std)
    candidate = _physical_motion(candidate_norm, motion_mean, motion_std)
    targets = prepare_targets(baseline, valid_mask.bool(), target_delta_deg)
    b_root = decode_rot6d_safe(baseline[..., MOTION_LAYOUT.root_rotation])
    c_root = decode_rot6d_safe(candidate[..., MOTION_LAYOUT.root_rotation])
    _, _, _, phi0 = _root_forward(b_root)
    forward, _, _, phig = _root_forward(c_root)
    pitch_error = (phi0 - phig - float(target_delta_deg)).abs()
    cross = torch.linalg.vector_norm(
        torch.cross(forward, targets.target_forward.detach(), dim=-1), dim=-1
    )
    dot = (forward * targets.target_forward.detach()).sum(-1).clamp(-1.0, 1.0)
    vector_error = torch.atan2(cross, dot) * 180.0 / math.pi
    mask = valid_mask.bool()
    pitch_values = pitch_error[mask]
    forward_values = vector_error[mask]
    metrics = {
        "pitch_mae_deg": float(pitch_values.mean().item()),
        "pitch_p95_deg": float(torch.quantile(pitch_values, 0.95).item()),
        "forward_p95_deg": float(torch.quantile(forward_values, 0.95).item()),
        "forward_mean_deg": float(forward_values.mean().item()),
        "dose_sign_correct": bool(
            torch.sign((phi0 - phig)[mask].mean())
            == torch.sign(torch.as_tensor(target_delta_deg, device=phi0.device))
        ),
    }
    return targets.target_forward.detach(), metrics


def _root_loss(
    *,
    candidate_norm: torch.Tensor,
    baseline_norm: torch.Tensor,
    valid_mask: torch.Tensor,
    motion_mean: torch.Tensor,
    motion_std: torch.Tensor,
    target_delta_deg: float,
    forward_loss_temperature: float,
) -> torch.Tensor:
    baseline = _physical_motion(baseline_norm.detach(), motion_mean, motion_std)
    candidate = _physical_motion(candidate_norm, motion_mean, motion_std)
    targets = prepare_targets(baseline, valid_mask.bool(), target_delta_deg)
    root = decode_rot6d_safe(candidate[..., MOTION_LAYOUT.root_rotation])
    forward = _root_forward(root)[0]
    target = targets.target_forward.detach()
    cross = torch.linalg.vector_norm(torch.cross(forward, target, dim=-1), dim=-1)
    dot = (forward * target).sum(-1).clamp(-1.0, 1.0)
    angle_deg = torch.atan2(cross, dot) * 180.0 / math.pi
    mask = valid_mask.bool().to(dtype=angle_deg.dtype)
    # A single soft-maximum root-forward error emphasizes the worst frames
    # without introducing a second task or an arbitrary cross-unit weight.
    temperature = float(forward_loss_temperature)
    masked = (temperature * angle_deg).masked_fill(~valid_mask.bool(), float("-inf"))
    return (torch.logsumexp(masked, dim=-1) / temperature).mean()


def _within_trust_region(delta: torch.Tensor, max_rms: float) -> torch.Tensor:
    rms = torch.sqrt(delta.float().square().mean())
    scale = torch.minimum(
        torch.ones((), dtype=delta.dtype, device=delta.device),
        torch.as_tensor(max_rms, dtype=delta.dtype, device=delta.device)
        / rms.clamp_min(1e-12),
    )
    return delta * scale


def _is_feasible(metrics: dict, config: MinimalSourceNoiseConfig) -> bool:
    return (
        metrics["pitch_mae_deg"] <= config.feasible_pitch_mae_deg
        and metrics["forward_p95_deg"] <= config.feasible_forward_p95_deg
        and metrics["dose_sign_correct"]
    )


def _source_rms(delta: torch.Tensor) -> float:
    return float(torch.sqrt(delta.float().square().mean()).item())


def run_minimal_source_noise_optimization(
    *,
    model,
    scheduler,
    official_result: FlowSampleResult,
    prompt_emb: torch.Tensor,
    prompt_emb_null: torch.Tensor,
    valid_mask: torch.Tensor,
    ref_motion: torch.Tensor,
    ref_motion_mask: torch.Tensor,
    condition_on_text: bool,
    attend_to_text_mask: Optional[torch.Tensor],
    motion_mean: torch.Tensor,
    motion_std: torch.Tensor,
    dtype: torch.dtype,
    target_delta_deg: float,
    config: MinimalSourceNoiseConfig,
    sampler_config: DifferentiableSamplerConfig,
) -> SourceNoiseOptimizationResult:
    """Find a feasible source-noise edit and perform one final authority pass."""

    config.validate()
    if sampler_config.num_inference_steps != 50:
        raise ValueError("v2 source-noise control requires 50 sampling steps")
    if official_result.initial_noise.shape[0] != 1:
        raise ValueError("v2 source-noise control currently requires batch=1")
    if not -10.0 <= float(target_delta_deg) <= 10.0:
        raise ValueError("target_delta_deg must lie in [-10,10]")

    was_training = model.training
    parameters = list(model.parameters())
    original_requires_grad = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    model.eval()
    z0 = official_result.initial_noise.detach()
    delta = torch.zeros_like(z0, dtype=torch.float32, requires_grad=True)
    baseline_norm = official_result.official_pre_cast.detach().float()
    baseline_metrics = _target_and_metrics(
        baseline_norm=baseline_norm,
        candidate_norm=baseline_norm,
        valid_mask=valid_mask,
        motion_mean=motion_mean,
        motion_std=motion_std,
        target_delta_deg=target_delta_deg,
    )[1]
    history = []
    best_delta = None
    best_norm = None
    best_metrics = None
    best_rms = math.inf

    for iteration in range(config.iterations):
        candidate_noise = z0 + delta.to(dtype=z0.dtype)
        candidate = differentiable_generate(
            model=model,
            scheduler=scheduler,
            prompt_emb=prompt_emb,
            prompt_emb_null=prompt_emb_null,
            initial_noise=candidate_noise,
            valid_mask=valid_mask,
            ref_motion=ref_motion,
            ref_motion_mask=ref_motion_mask,
            condition_on_text=condition_on_text,
            attend_to_text_mask=attend_to_text_mask,
            dtype=dtype,
            config=sampler_config,
        )
        loss = _root_loss(
            candidate_norm=candidate.official_pre_cast,
            baseline_norm=baseline_norm,
            valid_mask=valid_mask,
            motion_mean=motion_mean,
            motion_std=motion_std,
            target_delta_deg=target_delta_deg,
            forward_loss_temperature=config.forward_loss_temperature,
        )
        gradient = torch.autograd.grad(loss, delta, retain_graph=False)[0]
        with torch.no_grad():
            metrics = _target_and_metrics(
                baseline_norm=baseline_norm,
                candidate_norm=candidate.official_pre_cast.detach(),
                valid_mask=valid_mask,
                motion_mean=motion_mean,
                motion_std=motion_std,
                target_delta_deg=target_delta_deg,
            )[1]
            delta_rms = _source_rms(delta)
            feasible = _is_feasible(metrics, config)
            if feasible and delta_rms < best_rms:
                best_delta = delta.detach().clone()
                best_norm = candidate.official_pre_cast.detach().float().clone()
                best_metrics = metrics
                best_rms = delta_rms
            grad_norm = torch.linalg.vector_norm(gradient.float()).clamp_min(1e-12)
            # ``step_rms`` is defined over all source-noise elements, whereas
            # vector_norm is an L2 norm.  Convert the requested RMS to the
            # corresponding L2 radius before normalising the gradient.
            step_l2 = float(config.step_rms) * math.sqrt(gradient.numel())
            step = gradient * (step_l2 / grad_norm)
            delta_next = _within_trust_region(delta - step, config.max_delta_rms)
            history.append(
                {
                    "iteration": iteration,
                    "loss_forward_softmax_deg": float(loss.detach().item()),
                    "source_delta_rms": delta_rms,
                    "gradient_l2": float(grad_norm.item()),
                    "feasible": feasible,
                    "metrics": metrics,
                }
            )
            delta = delta_next.detach().requires_grad_(True)
        del gradient, loss, candidate

    # Second-level source-distance search along the best feasible direction.
    if best_delta is not None and config.line_search_steps:
        low = 0.0
        high = 1.0
        for _ in range(config.line_search_steps):
            alpha = 0.5 * (low + high)
            trial_delta = best_delta * alpha
            trial = differentiable_generate(
                model=model,
                scheduler=scheduler,
                prompt_emb=prompt_emb,
                prompt_emb_null=prompt_emb_null,
                initial_noise=z0 + trial_delta.to(dtype=z0.dtype),
                valid_mask=valid_mask,
                ref_motion=ref_motion,
                ref_motion_mask=ref_motion_mask,
                condition_on_text=condition_on_text,
                attend_to_text_mask=attend_to_text_mask,
                dtype=dtype,
                config=sampler_config,
            )
            trial_metrics = _target_and_metrics(
                baseline_norm=baseline_norm,
                candidate_norm=trial.official_pre_cast.detach(),
                valid_mask=valid_mask,
                motion_mean=motion_mean,
                motion_std=motion_std,
                target_delta_deg=target_delta_deg,
            )[1]
            if _is_feasible(trial_metrics, config):
                high = alpha
                best_delta = trial_delta.detach().clone()
                best_norm = trial.official_pre_cast.detach().float().clone()
                best_metrics = trial_metrics
                best_rms = _source_rms(best_delta)
            else:
                low = alpha
            del trial

    if best_delta is None:
        chosen_delta = delta.detach()
        chosen_norm = baseline_norm
        chosen_metrics = history[-1]["metrics"] if history else baseline_metrics
        status = "INFEASIBLE_WITHIN_BUDGET"
        feasible = False
    else:
        chosen_delta = best_delta
        chosen_norm = best_norm
        chosen_metrics = best_metrics
        status = "FEASIBLE"
        feasible = True

    # The only direct-pose authority operation is at the final output boundary.
    projection = authority_project(
        _physical_motion(chosen_norm, motion_mean, motion_std),
        valid_mask=valid_mask.bool(),
        input_standardized=False,
        output_standardized=True,
        mean=motion_mean,
        std=motion_std,
        output_dtype=torch.float32,
    )
    summary = {
        "protocol": PROTOCOL_NAME,
        "status": status,
        "feasible": feasible,
        "target_delta_deg": float(target_delta_deg),
        "config": asdict(config),
        "sampler": asdict(sampler_config),
        "baseline_metrics": baseline_metrics,
        "chosen_metrics": chosen_metrics,
        "source_delta_rms": _source_rms(chosen_delta),
        "source_delta_max_abs": float(chosen_delta.float().abs().max().item()),
        "iterations_completed": len(history),
        "history": history,
        "authority_projection_audits": list(projection.audits),
    }
    for parameter, requires_grad in zip(parameters, original_requires_grad):
        parameter.requires_grad_(requires_grad)
    model.train(was_training)
    if z0.device.type == "cuda":
        torch.cuda.empty_cache()
    return SourceNoiseOptimizationResult(
        protocol=PROTOCOL_NAME,
        motion_norm=projection.motion,
        baseline_norm=baseline_norm,
        optimized_norm=chosen_norm,
        source_delta=chosen_delta,
        summary=summary,
    )


__all__ = [
    "PROTOCOL_NAME",
    "MinimalSourceNoiseConfig",
    "SourceNoiseOptimizationResult",
    "run_minimal_source_noise_optimization",
]
