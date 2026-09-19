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
import time
from functools import wraps
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
    iterations: int = 120
    step_rms: float = 0.01
    max_delta_rms: float = 1.0
    line_search_steps: int = 8
    feasible_pitch_mae_deg: float = 1.0
    feasible_forward_p95_deg: float = 2.0
    forward_loss_temperature: float = 5.0
    use_gradient_checkpointing: bool = True
    max_runtime_seconds: float = 0.0

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
        if self.max_runtime_seconds < 0.0:
            raise ValueError("max_runtime_seconds must be non-negative")


@dataclass
class SourceNoiseOptimizationResult:
    protocol: str
    motion_norm: torch.Tensor
    baseline_norm: torch.Tensor
    optimized_norm: torch.Tensor
    source_delta: torch.Tensor
    summary: dict


def select_source_noise_output(
    baseline_result,
    source_noise_result: Optional[SourceNoiseOptimizationResult],
    *,
    return_delta: bool = False,
):
    """Select one source-noise result and keep its provenance paired.

    ``None`` is the explicit infeasible fallback.  Keeping this boundary in a
    small pure function makes the train/eval routing testable without loading
    the model and prevents a later M0 default assignment from overwriting a
    selected source-noise candidate.
    """

    if source_noise_result is None:
        selected = baseline_result
        if isinstance(baseline_result, torch.Tensor):
            delta = torch.zeros_like(baseline_result, dtype=torch.float32)
        else:
            delta = None
    else:
        selected = source_noise_result.motion_norm
        delta = getattr(source_noise_result, "source_delta", None)
        if delta is None:
            delta = torch.zeros_like(selected, dtype=torch.float32)
    if return_delta:
        return selected, delta
    return selected


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


def _valid_source_mask(delta: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones(delta.shape[:-1], dtype=torch.bool, device=delta.device)
    if valid_mask.shape != delta.shape[:2]:
        raise ValueError("valid_mask must match source noise [B,T]")
    return valid_mask.bool()


def _within_trust_region(
    delta: torch.Tensor,
    max_rms: float,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mask = _valid_source_mask(delta, valid_mask)
    masked_delta = delta * mask[..., None].to(delta.dtype)
    element_count = mask.sum().clamp_min(1) * delta.shape[-1]
    rms = torch.sqrt(masked_delta.float().square().sum() / element_count)
    scale = torch.minimum(
        torch.ones((), dtype=delta.dtype, device=delta.device),
        torch.as_tensor(max_rms, dtype=delta.dtype, device=delta.device)
        / rms.clamp_min(1e-12),
    )
    return masked_delta * scale


def _is_feasible(metrics: dict, config: MinimalSourceNoiseConfig) -> bool:
    return (
        metrics["pitch_mae_deg"] <= config.feasible_pitch_mae_deg
        and metrics["forward_p95_deg"] <= config.feasible_forward_p95_deg
        and metrics["dose_sign_correct"]
    )


def _source_rms(delta: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> float:
    mask = _valid_source_mask(delta, valid_mask)
    masked_delta = delta.float() * mask[..., None]
    element_count = mask.sum().clamp_min(1) * delta.shape[-1]
    return float(torch.sqrt(masked_delta.square().sum() / element_count).item())


def _normalized_negative_gradient(
    gradient: torch.Tensor,
    step_rms: float,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor | None:
    """Return a masked negative-gradient proposal with the requested RMS."""

    rms = _source_rms(gradient, valid_mask)
    if not math.isfinite(rms) or rms <= 1e-12:
        return None
    direction = -gradient * (float(step_rms) / rms)
    mask = _valid_source_mask(direction, valid_mask)
    return direction * mask[..., None].to(direction.dtype)


def _restore_model_state_on_exit(function):
    """Restore training and ``requires_grad`` flags even on optimizer errors."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        model = kwargs.get("model")
        if model is None:
            raise TypeError("source-noise optimizer requires a model keyword")
        parameters = list(model.parameters())
        was_training = model.training
        original_requires_grad = [parameter.requires_grad for parameter in parameters]
        try:
            return function(*args, **kwargs)
        finally:
            for parameter, requires_grad in zip(parameters, original_requires_grad):
                parameter.requires_grad_(requires_grad)
            model.train(was_training)

    return wrapped


@_restore_model_state_on_exit
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
    if valid_mask.shape != official_result.initial_noise.shape[:2] or not valid_mask.bool().any():
        raise ValueError("valid_mask must match source noise and contain a valid frame")

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
    valid_mask = valid_mask.bool()
    history = []
    best_delta = None
    best_norm = None
    best_metrics = None
    best_rms = math.inf
    best_infeasible = None
    started = time.monotonic()
    moment = torch.zeros_like(delta)
    second_moment = torch.zeros_like(delta)
    adam_step = 0
    stalled_steps = 0

    def _timed_out() -> bool:
        return bool(
            config.max_runtime_seconds > 0.0
            and time.monotonic() - started >= config.max_runtime_seconds
        )

    def _remember_candidate(candidate_delta, candidate_norm, metrics):
        nonlocal best_delta, best_norm, best_metrics, best_rms, best_infeasible
        actual_rms = _source_rms(candidate_delta, valid_mask)
        if _is_feasible(metrics, config) and actual_rms < best_rms:
            best_delta = candidate_delta.detach().float().clone()
            best_norm = candidate_norm.detach().float().clone()
            best_metrics = dict(metrics)
            best_rms = actual_rms
        elif best_infeasible is None or (
            metrics["forward_p95_deg"], metrics["pitch_mae_deg"]
        ) < best_infeasible["score"]:
            best_infeasible = {
                "score": (metrics["forward_p95_deg"], metrics["pitch_mae_deg"]),
                "metrics": dict(metrics),
                "source_delta_rms": actual_rms,
            }

    for iteration in range(config.iterations):
        if _timed_out():
            break
        base_delta = delta.detach().clone()
        # Keep the live delta in the differentiable path.  ``base_delta`` is
        # detached only for proposal bookkeeping and replay metadata.
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
            actual_delta = candidate_noise.float() - z0.float()
            metrics = _target_and_metrics(
                baseline_norm=baseline_norm,
                candidate_norm=candidate.official_pre_cast.detach(),
                valid_mask=valid_mask,
                motion_mean=motion_mean,
                motion_std=motion_std,
                target_delta_deg=target_delta_deg,
            )[1]
            delta_rms = _source_rms(actual_delta, valid_mask)
            feasible = _is_feasible(metrics, config)
            _remember_candidate(actual_delta, candidate.official_pre_cast, metrics)
            grad_finite = bool(torch.isfinite(gradient).all().item())
            grad_norm = torch.linalg.vector_norm(gradient.float())
            entry = {
                "iteration": iteration,
                "loss_forward_softmax_deg": float(loss.detach().item()),
                "source_delta_rms": delta_rms,
                "gradient_l2": float(grad_norm.item()),
                "gradient_finite": grad_finite,
                "feasible": feasible,
                "accepted": True,
                "metrics": metrics,
            }
            history.append(entry)
            if abs(float(target_delta_deg)) < 1e-8:
                delta = torch.zeros_like(delta, dtype=torch.float32, requires_grad=True)
                del gradient, loss, candidate
                break
            if (
                not grad_finite
                or not math.isfinite(float(grad_norm.item()))
                or grad_norm <= 1e-12
            ):
                entry["accepted"] = False
                entry["stop_reason"] = "nonfinite_or_zero_gradient"
                del gradient, loss, candidate
                break

            adam_step += 1
            moment.mul_(0.9).add_(gradient, alpha=0.1)
            second_moment.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
            bias1 = 1.0 - 0.9 ** adam_step
            bias2 = 1.0 - 0.999 ** adam_step
            direction = -(
                (moment / bias1)
                / ((second_moment / bias2).sqrt() + 1e-8)
            )
            direction = direction * _valid_source_mask(direction, valid_mask)[..., None]
            direction_rms = _source_rms(direction, valid_mask)
            if direction_rms > 1e-12 and math.isfinite(direction_rms):
                direction = direction * (float(config.step_rms) / direction_rms)
            else:
                direction = torch.zeros_like(gradient)
                entry["adam_direction"] = "zero_or_nonfinite"
            accepted = False
            base_loss = float(loss.detach().item())
            trial_loss = None
            accepted_trial = None
            accepted_trial_metrics = None
            accepted_trial_delta = None
            for backtrack in range(5):
                scale = 0.5 ** backtrack
                trial_delta = _within_trust_region(
                    base_delta + direction * scale,
                    config.max_delta_rms,
                    valid_mask,
                )
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
                with torch.no_grad():
                    trial_loss_tensor = _root_loss(
                        candidate_norm=trial.official_pre_cast,
                        baseline_norm=baseline_norm,
                        valid_mask=valid_mask,
                        motion_mean=motion_mean,
                        motion_std=motion_std,
                        target_delta_deg=target_delta_deg,
                        forward_loss_temperature=config.forward_loss_temperature,
                    )
                    trial_loss = float(trial_loss_tensor.item())
                    trial_actual_delta = (
                        (z0 + trial_delta.to(dtype=z0.dtype)).float() - z0.float()
                    )
                    trial_metrics = _target_and_metrics(
                        baseline_norm=baseline_norm,
                        candidate_norm=trial.official_pre_cast,
                        valid_mask=valid_mask,
                        motion_mean=motion_mean,
                        motion_std=motion_std,
                        target_delta_deg=target_delta_deg,
                    )[1]
                if math.isfinite(trial_loss) and trial_loss < base_loss:
                    delta = trial_delta.detach().float().requires_grad_(True)
                    accepted = True
                    accepted_trial = trial
                    accepted_trial_metrics = trial_metrics
                    accepted_trial_delta = trial_actual_delta
                    entry["backtrack"] = backtrack
                    entry["accepted_loss"] = trial_loss
                    break
                del trial
            if accepted and accepted_trial is not None:
                # The accepted trial is a real candidate, not merely a
                # proposal. Register it before the next loop or a timeout so
                # the last verified update can never disappear from the
                # archive.
                _remember_candidate(
                    accepted_trial_delta,
                    accepted_trial.official_pre_cast,
                    accepted_trial_metrics,
                )
                del accepted_trial
            if not accepted:
                # Adam can point into a locally flat or badly conditioned
                # direction.  Retry the same verified backtracking rule with
                # the raw negative gradient, normalized in the actual
                # masked RMS metric, before declaring a stall.
                fallback_direction = _normalized_negative_gradient(
                    gradient, config.step_rms, valid_mask
                )
                if fallback_direction is not None:
                    for backtrack in range(5):
                        scale = 0.5 ** backtrack
                        trial_delta = _within_trust_region(
                            base_delta + fallback_direction * scale,
                            config.max_delta_rms,
                            valid_mask,
                        )
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
                        with torch.no_grad():
                            fallback_loss = float(
                                _root_loss(
                                    candidate_norm=trial.official_pre_cast,
                                    baseline_norm=baseline_norm,
                                    valid_mask=valid_mask,
                                    motion_mean=motion_mean,
                                    motion_std=motion_std,
                                    target_delta_deg=target_delta_deg,
                                    forward_loss_temperature=config.forward_loss_temperature,
                                ).item()
                            )
                            fallback_delta = (
                                (z0 + trial_delta.to(dtype=z0.dtype)).float()
                                - z0.float()
                            )
                            fallback_metrics = _target_and_metrics(
                                baseline_norm=baseline_norm,
                                candidate_norm=trial.official_pre_cast,
                                valid_mask=valid_mask,
                                motion_mean=motion_mean,
                                motion_std=motion_std,
                                target_delta_deg=target_delta_deg,
                            )[1]
                        if math.isfinite(fallback_loss) and fallback_loss < base_loss:
                            delta = trial_delta.detach().float().requires_grad_(True)
                            _remember_candidate(
                                fallback_delta,
                                trial.official_pre_cast,
                                fallback_metrics,
                            )
                            entry["backtrack"] = backtrack
                            entry["accepted_loss"] = fallback_loss
                            entry["direction_fallback"] = "normalized_negative_gradient"
                            accepted = True
                            del trial
                            break
                        del trial
            if not accepted:
                stalled_steps += 1
                delta = base_delta.detach().float().requires_grad_(True)
                entry["accepted"] = False
                entry["backtrack"] = 4
                entry["stop_reason"] = "no_verified_descent"
                if stalled_steps >= 2:
                    entry["stop_reason"] = "repeated_no_verified_descent"
                    del gradient, loss, candidate
                    break
            else:
                stalled_steps = 0
        del gradient, loss, candidate

    # Evaluate the final iterate explicitly so the last update can be selected.
    if not _timed_out() and abs(float(target_delta_deg)) >= 1e-8:
        final_noise = z0 + delta.detach().to(dtype=z0.dtype)
        with torch.no_grad():
            final_candidate = differentiable_generate(
                model=model,
                scheduler=scheduler,
                prompt_emb=prompt_emb,
                prompt_emb_null=prompt_emb_null,
                initial_noise=final_noise,
                valid_mask=valid_mask,
                ref_motion=ref_motion,
                ref_motion_mask=ref_motion_mask,
                condition_on_text=condition_on_text,
                attend_to_text_mask=attend_to_text_mask,
                dtype=dtype,
                config=sampler_config,
            )
        final_metrics = _target_and_metrics(
            baseline_norm=baseline_norm,
            candidate_norm=final_candidate.official_pre_cast,
            valid_mask=valid_mask,
            motion_mean=motion_mean,
            motion_std=motion_std,
            target_delta_deg=target_delta_deg,
        )[1]
        _remember_candidate(
            final_noise.float() - z0.float(), final_candidate.official_pre_cast, final_metrics
        )
        del final_candidate

    # Second-level source-distance search tests independent scales of a frozen
    # feasible direction; the anchor is never modified during the scan.
    if (
        best_delta is not None
        and config.line_search_steps
        and abs(float(target_delta_deg)) >= 1e-8
        and not _timed_out()
    ):
        anchor_delta = best_delta.detach().float().clone()
        alphas = [1.0] + [0.5 ** index for index in range(1, config.line_search_steps + 1)]
        for alpha in alphas:
            if _timed_out():
                break
            trial_delta = anchor_delta * alpha
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
            trial_actual_delta = (z0 + trial_delta.to(dtype=z0.dtype)).float() - z0.float()
            trial_metrics = _target_and_metrics(
                baseline_norm=baseline_norm,
                candidate_norm=trial.official_pre_cast,
                valid_mask=valid_mask,
                motion_mean=motion_mean,
                motion_std=motion_std,
                target_delta_deg=target_delta_deg,
            )[1]
            if _is_feasible(trial_metrics, config):
                actual_rms = _source_rms(trial_actual_delta, valid_mask)
                if actual_rms < best_rms:
                    best_delta = trial_actual_delta.detach().float().clone()
                    best_norm = trial.official_pre_cast.detach().float().clone()
                    best_metrics = dict(trial_metrics)
                    best_rms = actual_rms
            del trial

    if best_delta is None:
        # A failed search is an explicit M0 fallback.  Do not expose the
        # unevaluated post-update delta from the last iteration.
        chosen_delta = torch.zeros_like(z0, dtype=torch.float32)
        chosen_norm = baseline_norm
        chosen_metrics = dict(baseline_metrics)
        status = "INFEASIBLE_WITHIN_BUDGET"
        feasible = False
        selected_source = "m0_fallback"
    else:
        chosen_delta = best_delta
        chosen_norm = best_norm
        chosen_metrics = best_metrics
        status = "FEASIBLE"
        feasible = True
        selected_source = "best_verified_feasible"

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
        "selected_source": selected_source,
        "target_delta_deg": float(target_delta_deg),
        "config": asdict(config),
        "sampler": asdict(sampler_config),
        "baseline_metrics": baseline_metrics,
        "chosen_metrics": chosen_metrics,
        "best_infeasible": best_infeasible,
        "source_delta_rms": _source_rms(chosen_delta, valid_mask),
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
    "select_source_noise_output",
    "_normalized_negative_gradient",
    "run_minimal_source_noise_optimization",
]
