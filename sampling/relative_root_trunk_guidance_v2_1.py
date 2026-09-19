"""Independent v2.1 source-noise search for root--trunk relative angle.

The historical v2 module is intentionally not extended here. This module
reuses its small numerical helpers and implements a separate search boundary
so the v2 protocol remains frozen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Optional

import torch

from evaluation.relative_root_trunk_v2_1 import relative_angle_loss, relative_angle_metrics
from motion_rep.pose_authority import authority_project
from sampling.differentiable_flow_sampler import DifferentiableSamplerConfig, differentiable_generate
from sampling.flow_sampler import FlowSampleResult
from sampling.relative_root_forward_guidance_v2 import (
    MinimalSourceNoiseConfig,
    SourceNoiseOptimizationResult,
    _normalized_negative_gradient,
    _physical_motion,
    _restore_model_state_on_exit,
    _source_rms,
    _within_trust_region,
)


PROTOCOL_NAME = "vimogen_relative_root_trunk_v2_1_minimal_source_noise"


@dataclass(frozen=True)
class RelativeRootTrunkConfig(MinimalSourceNoiseConfig):
    feasible_relative_mae_deg: float = 1.0
    feasible_relative_p95_deg: float = 2.0

    def validate(self) -> None:
        super().validate()
        if self.feasible_relative_mae_deg <= 0.0 or self.feasible_relative_p95_deg <= 0.0:
            raise ValueError("relative-angle feasibility tolerances must be positive")


def _metrics(
    *, baseline_norm: torch.Tensor, candidate_norm: torch.Tensor,
    valid_mask: torch.Tensor, motion_mean: torch.Tensor, motion_std: torch.Tensor,
    target_delta_deg: float, **_: Any,
) -> dict[str, Any]:
    baseline = _physical_motion(baseline_norm.detach(), motion_mean, motion_std)
    candidate = _physical_motion(candidate_norm, motion_mean, motion_std)
    return relative_angle_metrics(baseline, candidate, valid_mask.bool(), target_delta_deg)


def _loss(
    *, baseline_norm: torch.Tensor, candidate_norm: torch.Tensor,
    valid_mask: torch.Tensor, motion_mean: torch.Tensor, motion_std: torch.Tensor,
    target_delta_deg: float, forward_loss_temperature: float, **_: Any,
) -> torch.Tensor:
    baseline = _physical_motion(baseline_norm.detach(), motion_mean, motion_std)
    candidate = _physical_motion(candidate_norm, motion_mean, motion_std)
    return relative_angle_loss(
        baseline, candidate, valid_mask.bool(), target_delta_deg,
        temperature=forward_loss_temperature,
    )


def _feasible(metrics: dict[str, Any], config: RelativeRootTrunkConfig) -> bool:
    return (
        metrics.get("relative_angle_mae_deg") is not None
        and metrics["relative_angle_mae_deg"] <= config.feasible_relative_mae_deg
        and metrics["relative_angle_p95_deg"] <= config.feasible_relative_p95_deg
        and bool(metrics.get("dose_sign_correct", False))
    )


def _infeasible_score(metrics: dict[str, Any]) -> tuple[float, float]:
    return (
        float(metrics.get("relative_angle_p95_deg", float("inf"))),
        float(metrics.get("relative_angle_mae_deg", float("inf"))),
    )


@_restore_model_state_on_exit
def run_minimal_source_noise_relative_root_trunk_optimization(
    *, model, scheduler, official_result: FlowSampleResult,
    prompt_emb: torch.Tensor, prompt_emb_null: torch.Tensor,
    valid_mask: torch.Tensor, ref_motion: torch.Tensor,
    ref_motion_mask: torch.Tensor, condition_on_text: bool,
    attend_to_text_mask: Optional[torch.Tensor], motion_mean: torch.Tensor,
    motion_std: torch.Tensor, dtype: torch.dtype, target_delta_deg: float,
    config: RelativeRootTrunkConfig,
    sampler_config: DifferentiableSamplerConfig,
) -> SourceNoiseOptimizationResult:
    """Search source noise using only the relative-angle objective."""

    config.validate()
    if sampler_config.num_inference_steps != 50:
        raise ValueError("v2.1 source-noise control requires 50 sampling steps")
    if official_result.initial_noise.shape[0] != 1:
        raise ValueError("v2.1 source-noise control currently requires batch=1")
    if valid_mask.shape != official_result.initial_noise.shape[:2] or not valid_mask.bool().any():
        raise ValueError("valid_mask must match source noise and contain a valid frame")
    if not -10.0 <= float(target_delta_deg) <= 10.0:
        raise ValueError("target_delta_deg must lie in [-10,10]")

    parameters = list(model.parameters())
    for parameter in parameters:
        parameter.requires_grad_(False)
    model.eval()
    valid_mask = valid_mask.bool()
    z0 = official_result.initial_noise.detach()
    baseline_norm = official_result.official_pre_cast.detach().float()
    baseline_metrics = _metrics(
        baseline_norm=baseline_norm, candidate_norm=baseline_norm,
        valid_mask=valid_mask, motion_mean=motion_mean, motion_std=motion_std,
        target_delta_deg=target_delta_deg,
    )
    delta = torch.zeros_like(z0, dtype=torch.float32, requires_grad=True)
    moment = torch.zeros_like(delta)
    second_moment = torch.zeros_like(delta)
    best_delta = None
    best_norm = None
    best_metrics = None
    best_rms = math.inf
    best_infeasible = None
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    adam_step = 0
    stalled_steps = 0

    def timed_out() -> bool:
        return bool(config.max_runtime_seconds > 0.0 and time.monotonic() - started >= config.max_runtime_seconds)

    def remember(candidate_delta: torch.Tensor, candidate_norm: torch.Tensor, metrics: dict[str, Any]) -> None:
        nonlocal best_delta, best_norm, best_metrics, best_rms, best_infeasible
        actual_rms = _source_rms(candidate_delta, valid_mask)
        if _feasible(metrics, config) and actual_rms < best_rms:
            best_delta = candidate_delta.detach().float().clone()
            best_norm = candidate_norm.detach().float().clone()
            best_metrics = dict(metrics)
            best_rms = actual_rms
            return
        score = _infeasible_score(metrics)
        if best_infeasible is None or score < best_infeasible["score"]:
            best_infeasible = {"score": score, "metrics": dict(metrics), "source_delta_rms": actual_rms}

    def generate(trial_delta: torch.Tensor) -> FlowSampleResult:
        return differentiable_generate(
            model=model, scheduler=scheduler, prompt_emb=prompt_emb,
            prompt_emb_null=prompt_emb_null, initial_noise=z0 + trial_delta.to(dtype=z0.dtype),
            valid_mask=valid_mask, ref_motion=ref_motion, ref_motion_mask=ref_motion_mask,
            condition_on_text=condition_on_text, attend_to_text_mask=attend_to_text_mask,
            dtype=dtype, config=sampler_config,
        )

    for iteration in range(config.iterations):
        if timed_out():
            break
        base_delta = delta.detach().clone()
        candidate = generate(delta)
        loss = _loss(
            baseline_norm=baseline_norm, candidate_norm=candidate.official_pre_cast,
            valid_mask=valid_mask, motion_mean=motion_mean, motion_std=motion_std,
            target_delta_deg=target_delta_deg, forward_loss_temperature=config.forward_loss_temperature,
        )
        gradient = torch.autograd.grad(loss, delta, retain_graph=False)[0]
        with torch.no_grad():
            actual_delta = (z0 + delta.to(dtype=z0.dtype)).float() - z0.float()
            metrics = _metrics(
                baseline_norm=baseline_norm, candidate_norm=candidate.official_pre_cast.detach(),
                valid_mask=valid_mask, motion_mean=motion_mean, motion_std=motion_std,
                target_delta_deg=target_delta_deg,
            )
            remember(actual_delta, candidate.official_pre_cast, metrics)
            gradient_finite = bool(torch.isfinite(gradient).all().item())
            gradient_rms = _source_rms(gradient, valid_mask)
            entry = {
                "iteration": iteration,
                "loss_relative_angle_softmax_deg": float(loss.detach().item()),
                "source_delta_rms": _source_rms(actual_delta, valid_mask),
                "gradient_rms": gradient_rms,
                "gradient_finite": gradient_finite,
                "feasible": _feasible(metrics, config),
                "accepted": True,
                "metrics": metrics,
            }
            history.append(entry)
            if not gradient_finite or not math.isfinite(gradient_rms) or gradient_rms <= 1e-12:
                entry["accepted"] = False
                entry["stop_reason"] = "nonfinite_or_zero_gradient"
                del gradient, loss, candidate
                break
            adam_step += 1
            moment.mul_(0.9).add_(gradient, alpha=0.1)
            second_moment.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
            direction = -((moment / (1.0 - 0.9 ** adam_step)) / ((second_moment / (1.0 - 0.999 ** adam_step)).sqrt() + 1e-8))
            direction_rms = _source_rms(direction, valid_mask)
            if direction_rms <= 1e-12 or not math.isfinite(direction_rms):
                entry["accepted"] = False
                entry["stop_reason"] = "zero_or_nonfinite_direction"
                del gradient, loss, candidate
                break
            direction = direction * (float(config.step_rms) / direction_rms)
            accepted = False
            base_loss = float(loss.detach().item())
            for backtrack in range(5):
                trial_delta = _within_trust_region(base_delta + direction * (0.5 ** backtrack), config.max_delta_rms, valid_mask)
                trial = generate(trial_delta)
                trial_loss = float(_loss(
                    baseline_norm=baseline_norm, candidate_norm=trial.official_pre_cast,
                    valid_mask=valid_mask, motion_mean=motion_mean, motion_std=motion_std,
                    target_delta_deg=target_delta_deg, forward_loss_temperature=config.forward_loss_temperature,
                ).item())
                trial_metrics = _metrics(
                    baseline_norm=baseline_norm, candidate_norm=trial.official_pre_cast,
                    valid_mask=valid_mask, motion_mean=motion_mean, motion_std=motion_std,
                    target_delta_deg=target_delta_deg,
                )
                remember((z0 + trial_delta.to(dtype=z0.dtype)).float() - z0.float(), trial.official_pre_cast, trial_metrics)
                if math.isfinite(trial_loss) and trial_loss < base_loss:
                    delta = trial_delta.detach().float().requires_grad_(True)
                    entry["backtrack"] = backtrack
                    entry["accepted_loss"] = trial_loss
                    accepted = True
                    del trial
                    break
                del trial
            if not accepted:
                fallback = _normalized_negative_gradient(gradient, config.step_rms, valid_mask)
                if fallback is not None:
                    for backtrack in range(5):
                        trial_delta = _within_trust_region(base_delta + fallback * (0.5 ** backtrack), config.max_delta_rms, valid_mask)
                        trial = generate(trial_delta)
                        trial_loss = float(_loss(
                            baseline_norm=baseline_norm, candidate_norm=trial.official_pre_cast,
                            valid_mask=valid_mask, motion_mean=motion_mean, motion_std=motion_std,
                            target_delta_deg=target_delta_deg, forward_loss_temperature=config.forward_loss_temperature,
                        ).item())
                        trial_metrics = _metrics(
                            baseline_norm=baseline_norm, candidate_norm=trial.official_pre_cast,
                            valid_mask=valid_mask, motion_mean=motion_mean, motion_std=motion_std,
                            target_delta_deg=target_delta_deg,
                        )
                        remember((z0 + trial_delta.to(dtype=z0.dtype)).float() - z0.float(), trial.official_pre_cast, trial_metrics)
                        if math.isfinite(trial_loss) and trial_loss < base_loss:
                            delta = trial_delta.detach().float().requires_grad_(True)
                            entry["backtrack"] = backtrack
                            entry["accepted_loss"] = trial_loss
                            entry["direction_fallback"] = "normalized_negative_gradient"
                            accepted = True
                            del trial
                            break
                        del trial
            if not accepted:
                stalled_steps += 1
                delta = base_delta.detach().float().requires_grad_(True)
                entry["accepted"] = False
                entry["stop_reason"] = "repeated_no_verified_descent" if stalled_steps >= 2 else "no_verified_descent"
                if stalled_steps >= 2:
                    del gradient, loss, candidate
                    break
            else:
                stalled_steps = 0
        del gradient, loss, candidate

    if best_delta is not None and config.line_search_steps and not timed_out():
        anchor = best_delta.detach().float().clone()
        for index in range(config.line_search_steps + 1):
            trial_delta = anchor * (0.5 ** index)
            trial = generate(trial_delta)
            trial_metrics = _metrics(
                baseline_norm=baseline_norm, candidate_norm=trial.official_pre_cast,
                valid_mask=valid_mask, motion_mean=motion_mean, motion_std=motion_std,
                target_delta_deg=target_delta_deg,
            )
            actual_delta = (z0 + trial_delta.to(dtype=z0.dtype)).float() - z0.float()
            if _feasible(trial_metrics, config) and _source_rms(actual_delta, valid_mask) < best_rms:
                best_delta, best_norm, best_metrics = actual_delta.detach().clone(), trial.official_pre_cast.detach().float().clone(), dict(trial_metrics)
                best_rms = _source_rms(actual_delta, valid_mask)
            del trial

    if best_delta is None:
        chosen_delta, chosen_norm, chosen_metrics = torch.zeros_like(z0, dtype=torch.float32), baseline_norm, dict(baseline_metrics)
        status, feasible, selected_source = "INFEASIBLE_WITHIN_BUDGET", False, "m0_fallback"
    else:
        chosen_delta, chosen_norm, chosen_metrics = best_delta, best_norm, best_metrics
        status, feasible, selected_source = "FEASIBLE", True, "best_verified_feasible"
    projection = authority_project(
        _physical_motion(chosen_norm, motion_mean, motion_std), valid_mask=valid_mask,
        input_standardized=False, output_standardized=True, mean=motion_mean, std=motion_std,
        output_dtype=torch.float32,
    )
    summary = {
        "protocol": PROTOCOL_NAME, "status": status, "feasible": feasible,
        "selected_source": selected_source, "target_delta_deg": float(target_delta_deg),
        "config": asdict(config), "sampler": asdict(sampler_config),
        "baseline_metrics": baseline_metrics, "chosen_metrics": chosen_metrics,
        "best_infeasible": best_infeasible, "source_delta_rms": _source_rms(chosen_delta, valid_mask),
        "source_delta_max_abs": float(chosen_delta.float().abs().max().item()),
        "iterations_completed": len(history), "history": history,
        "authority_projection_audits": list(projection.audits),
    }
    return SourceNoiseOptimizationResult(
        protocol=PROTOCOL_NAME, motion_norm=projection.motion,
        baseline_norm=baseline_norm, optimized_norm=chosen_norm,
        source_delta=chosen_delta, summary=summary,
    )


__all__ = [
    "PROTOCOL_NAME", "RelativeRootTrunkConfig", "SourceNoiseOptimizationResult",
    "run_minimal_source_noise_relative_root_trunk_optimization",
]
