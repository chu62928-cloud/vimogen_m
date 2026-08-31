"""Differentiable replica of ViMoGen's frozen flow-matching sampler.

This module exists for the source-noise feasibility gate.  It intentionally
mirrors the unguided :class:`sampling.flow_sampler.FlowSampler` operation
order while preserving the graph from the final motion back to ``z0``.
The historical sampler remains unchanged.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
import math
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe
from motion_rep.pose_authority import _root_forward, forward_vector_loss, prepare_targets
from sampling.flow_sampler import FlowSampleResult
from utils import smooth_motion_rep


PROTOCOL_NAME = "vimogen_relative_root_forward_v2_differentiable_50step_gate"


@dataclass(frozen=True)
class DifferentiableSamplerConfig:
    num_inference_steps: int = 50
    denoising_strength: float = 0.7
    cfg_scale: float = 5.0
    smooth_kernel_size: int = 5
    smooth_sigma: float = 1.0
    use_gradient_checkpointing: bool = True


@dataclass(frozen=True)
class SourceNoiseGateConfig:
    target_delta_deg: float = 10.0
    max_reserved_mib: float = 28672.0


def _model_velocity(
    *,
    model,
    xt: torch.Tensor,
    timestep: torch.Tensor,
    context_input: torch.Tensor,
    valid_mask_input: torch.Tensor,
    ref_motion_input: torch.Tensor,
    ref_motion_mask_input: torch.Tensor,
    attend_to_text_mask_input: Optional[torch.Tensor],
    use_gradient_checkpointing: bool,
) -> torch.Tensor:
    latent_model_input = torch.cat([xt] * 2, dim=0)
    timestep_input = timestep.unsqueeze(0)

    def call_model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return model(
            x=x,
            timestep=t,
            context=context_input,
            x_mask=valid_mask_input,
            ref_motion=ref_motion_input,
            ref_motion_mask=ref_motion_mask_input,
            use_gradient_checkpointing=False,
            attend_to_text_mask=attend_to_text_mask_input,
        )

    if use_gradient_checkpointing:
        return checkpoint(
            call_model,
            latent_model_input,
            timestep_input,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    return call_model(latent_model_input, timestep_input)


def differentiable_generate(
    *,
    model,
    scheduler,
    prompt_emb: torch.Tensor,
    prompt_emb_null: torch.Tensor,
    initial_noise: torch.Tensor,
    valid_mask: torch.Tensor,
    ref_motion: torch.Tensor,
    ref_motion_mask: torch.Tensor,
    condition_on_text: bool,
    attend_to_text_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
    config: DifferentiableSamplerConfig,
) -> FlowSampleResult:
    """Run the official Euler/smoothing chain without detaching ``z0``."""

    if initial_noise.ndim != 3 or initial_noise.shape[-1] != MOTION_LAYOUT.total_dim:
        raise ValueError("initial_noise must have shape [B,T,276]")
    if valid_mask.shape != initial_noise.shape[:2]:
        raise ValueError("valid_mask must match initial_noise [B,T]")
    if ref_motion.ndim != 3 or ref_motion.shape[:2] != initial_noise.shape[:2]:
        raise ValueError("ref_motion must match initial_noise [B,T]")
    if ref_motion_mask.shape != initial_noise.shape[:2]:
        raise ValueError("ref_motion_mask must match initial_noise [B,T]")

    device = initial_noise.device
    xt = initial_noise
    scheduler.set_timesteps(
        config.num_inference_steps,
        training=False,
        denoising_strength=config.denoising_strength,
    )
    timesteps = scheduler.timesteps.to(device)

    if prompt_emb_null.size(1) < prompt_emb.size(1):
        padding = torch.zeros(
            prompt_emb.size(0),
            prompt_emb.size(1) - prompt_emb_null.size(1),
            prompt_emb.size(2),
            device=prompt_emb.device,
            dtype=prompt_emb.dtype,
        )
        prompt_emb_null = torch.cat([prompt_emb_null, padding], dim=1)

    valid_mask_input = torch.cat([valid_mask] * 2, dim=0)
    ref_motion_input = torch.cat([ref_motion, torch.zeros_like(ref_motion)], dim=0)
    ref_motion_mask_input = torch.cat([ref_motion_mask] * 2, dim=0)
    context_input = torch.cat([prompt_emb, prompt_emb_null], dim=0)
    attend_to_text_mask_input = None
    if attend_to_text_mask is not None:
        if attend_to_text_mask.shape != (initial_noise.shape[0],):
            raise ValueError("attend_to_text_mask must have shape [B]")
        attend_to_text_mask_input = torch.cat([attend_to_text_mask] * 2, dim=0)

    autocast_enabled = device.type == "cuda" and dtype in (
        torch.float16,
        torch.bfloat16,
    )
    autocast_ctx = (
        torch.amp.autocast(dtype=dtype, device_type=device.type)
        if autocast_enabled
        else contextlib.nullcontext()
    )
    for timestep in timesteps:
        with autocast_ctx:
            velocity = _model_velocity(
                model=model,
                xt=xt,
                timestep=timestep,
                context_input=context_input,
                valid_mask_input=valid_mask_input,
                ref_motion_input=ref_motion_input,
                ref_motion_mask_input=ref_motion_mask_input,
                attend_to_text_mask_input=attend_to_text_mask_input,
                use_gradient_checkpointing=config.use_gradient_checkpointing,
            )
            velocity_cond, velocity_uncond = velocity.chunk(2)
            if condition_on_text:
                velocity = velocity_uncond + config.cfg_scale * (
                    velocity_cond - velocity_uncond
                )
            else:
                velocity = velocity_cond
            xt = scheduler.step(velocity, timestep, xt)

    raw = xt.clone()
    smoothed = []
    for index in range(raw.shape[0]):
        smoothed.append(
            smooth_motion_rep(
                raw[index],
                kernel_size=config.smooth_kernel_size,
                sigma=config.smooth_sigma,
            )
        )
    official = torch.stack(smoothed, dim=0)
    return FlowSampleResult(
        initial_noise=initial_noise,
        raw=raw,
        official_pre_cast=official,
        official=official.to(dtype=dtype),
        sigmas=scheduler.sigmas.detach().clone(),
        timesteps=scheduler.timesteps.detach().clone(),
    )


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


def _root_objective(
    *,
    candidate_norm: torch.Tensor,
    baseline_norm: torch.Tensor,
    valid_mask: torch.Tensor,
    motion_mean: torch.Tensor,
    motion_std: torch.Tensor,
    target_delta_deg: float,
) -> torch.Tensor:
    baseline = _physical_motion(baseline_norm.detach(), motion_mean, motion_std)
    candidate = _physical_motion(candidate_norm, motion_mean, motion_std)
    targets = prepare_targets(baseline, valid_mask.bool(), target_delta_deg)
    candidate_root = decode_rot6d_safe(candidate[..., MOTION_LAYOUT.root_rotation])
    candidate_forward = _root_forward(candidate_root)[0]
    return forward_vector_loss(
        candidate_forward,
        targets.target_forward.detach(),
        valid_mask.bool(),
    )


def _subspace_response_vector(
    normalized_motion: torch.Tensor,
    motion_mean: torch.Tensor,
    motion_std: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Flatten root, trunk, foot-joint and translation responses per frame."""

    motion = _physical_motion(normalized_motion, motion_mean, motion_std)
    root = decode_rot6d_safe(motion[..., MOTION_LAYOUT.root_rotation])
    _, heading, _, pitch = _root_forward(root)
    joints = motion[..., MOTION_LAYOUT.joints].reshape(*motion.shape[:2], 22, 3)
    trunk = joints[..., 9, :] - joints[..., 3, :]
    translation = motion[..., MOTION_LAYOUT.root_translation]
    feet = joints[..., (10, 11), :].reshape(*motion.shape[:2], 6)
    values = torch.cat((pitch.unsqueeze(-1), heading, trunk, feet, translation), dim=-1)
    return values[valid_mask]


def run_source_noise_subspace_probe(
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
    sampler_config: DifferentiableSamplerConfig,
    historical_delta: Optional[torch.Tensor] = None,
    direction_seed: int = 314159,
    rms_values: tuple[float, ...] = (0.005, 0.01),
    target_delta_deg: float = 10.0,
) -> dict:
    """Probe 16 fixed source-noise directions with real central differences."""

    if official_result.initial_noise.shape[0] != 1:
        raise ValueError("source-noise subspace probe requires batch=1")
    was_training = model.training
    parameters = list(model.parameters())
    original_requires_grad = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    model.eval()
    z0 = official_result.initial_noise.detach()
    gradient_noise = z0.detach().clone().requires_grad_(True)
    gradient_result = differentiable_generate(
        model=model, scheduler=scheduler, prompt_emb=prompt_emb,
        prompt_emb_null=prompt_emb_null, initial_noise=gradient_noise,
        valid_mask=valid_mask, ref_motion=ref_motion,
        ref_motion_mask=ref_motion_mask, condition_on_text=condition_on_text,
        attend_to_text_mask=attend_to_text_mask, dtype=dtype,
        config=sampler_config,
    )
    objective = _root_objective(
        candidate_norm=gradient_result.official_pre_cast,
        baseline_norm=official_result.official_pre_cast,
        valid_mask=valid_mask,
        motion_mean=motion_mean,
        motion_std=motion_std,
        target_delta_deg=10.0,
    )
    gradient = torch.autograd.grad(objective, gradient_noise)[0].detach().float()
    del gradient_result, gradient_noise

    generator = torch.Generator(device=z0.device).manual_seed(direction_seed)
    random = torch.randn((14, *z0.shape), generator=generator, device=z0.device, dtype=torch.float32)
    directions = [gradient]
    if historical_delta is not None:
        if historical_delta.shape != z0.shape:
            raise ValueError("historical_delta must match initial noise")
        directions.append(historical_delta.detach().float())
    else:
        directions.append(torch.zeros_like(gradient))
    directions.extend(random[index] for index in range(14))

    normalized_directions = []
    direction_records = []
    mask = valid_mask.bool()[..., None]
    for index, direction in enumerate(directions):
        direction = direction * mask
        rms = torch.sqrt(direction.square().sum() / mask.sum().clamp_min(1) / direction.shape[-1])
        if not torch.isfinite(rms) or rms <= 1e-12:
            direction_records.append({"index": index, "status": "DEGENERATE", "rms": float(rms) if torch.isfinite(rms) else None, "sha256": None})
            normalized_directions.append(torch.zeros_like(direction))
            continue
        normalized = direction / rms
        normalized_directions.append(normalized)
        direction_records.append({"index": index, "status": "VALID", "rms": float(rms), "normalized_rms": 1.0, "sha256": __import__("hashlib").sha256(direction.detach().cpu().contiguous().numpy().tobytes()).hexdigest()})

    baseline_features = _subspace_response_vector(official_result.official_pre_cast, motion_mean, motion_std, valid_mask)
    no_grad_config = DifferentiableSamplerConfig(
        num_inference_steps=sampler_config.num_inference_steps,
        denoising_strength=sampler_config.denoising_strength,
        cfg_scale=sampler_config.cfg_scale,
        smooth_kernel_size=sampler_config.smooth_kernel_size,
        smooth_sigma=sampler_config.smooth_sigma,
        use_gradient_checkpointing=False,
    )
    response_matrices = []
    actual_perturbations = []
    for rms_value in rms_values:
        rows = []
        perturbation_rows = []
        for direction in normalized_directions:
            plus_noise = z0 + (float(rms_value) * direction).to(dtype=z0.dtype)
            minus_noise = z0 - (float(rms_value) * direction).to(dtype=z0.dtype)
            with torch.no_grad():
                plus = differentiable_generate(
                    model=model, scheduler=scheduler, prompt_emb=prompt_emb,
                    prompt_emb_null=prompt_emb_null, initial_noise=plus_noise,
                    valid_mask=valid_mask, ref_motion=ref_motion,
                    ref_motion_mask=ref_motion_mask, condition_on_text=condition_on_text,
                    attend_to_text_mask=attend_to_text_mask, dtype=dtype,
                    config=no_grad_config,
                )
                minus = differentiable_generate(
                    model=model, scheduler=scheduler, prompt_emb=prompt_emb,
                    prompt_emb_null=prompt_emb_null, initial_noise=minus_noise,
                    valid_mask=valid_mask, ref_motion=ref_motion,
                    ref_motion_mask=ref_motion_mask, condition_on_text=condition_on_text,
                    attend_to_text_mask=attend_to_text_mask, dtype=dtype,
                    config=no_grad_config,
                )
            plus_features = _subspace_response_vector(plus.official_pre_cast, motion_mean, motion_std, valid_mask)
            minus_features = _subspace_response_vector(minus.official_pre_cast, motion_mean, motion_std, valid_mask)
            rows.append(((plus_features - minus_features) / (2.0 * float(rms_value))).detach().cpu())
            perturbation_rows.append({
                "plus_rms": float(torch.sqrt((plus_noise.float() - z0.float()).square().mean()).item()),
                "minus_rms": float(torch.sqrt((minus_noise.float() - z0.float()).square().mean()).item()),
                "plus_minus_identical": bool(torch.equal(plus_noise, minus_noise)),
            })
            del plus, minus
        response_matrices.append(torch.stack(rows))
        actual_perturbations.append(perturbation_rows)
    response = torch.stack(response_matrices)
    # Solve the small linearized coefficient problem only after all response
    # directions have been measured.  This solver never changes the v2
    # optimizer output; it is an independent diagnostic oracle.
    from scipy.optimize import minimize

    response_np = response.detach().cpu().numpy().mean(axis=0)
    baseline_np = baseline_features.detach().cpu().numpy()
    target_np = baseline_np.copy()
    target_np[:, 0] -= float(target_delta_deg)
    frame_count = baseline_np.shape[0]

    def hold_cost(coefficients_np: np.ndarray) -> float:
        pred = baseline_np + np.einsum("d,dnf->nf", coefficients_np, response_np)
        trunk = (pred[:, 4:7] - baseline_np[:, 4:7]) / 0.01
        feet = (pred[:, 7:13] - baseline_np[:, 7:13]) / 0.00025
        translation = (pred[:, 13:16] - baseline_np[:, 13:16]) / 0.001
        return float(np.mean(np.concatenate((trunk, feet, translation), axis=1) ** 2) + 1e-6 * np.dot(coefficients_np, coefficients_np))

    def root_constraints(coefficients_np: np.ndarray) -> np.ndarray:
        pred = baseline_np + np.einsum("d,dnf->nf", coefficients_np, response_np)
        pitch_error = pred[:, 0] - target_np[:, 0]
        heading_delta = np.linalg.norm(pred[:, 1:4] - baseline_np[:, 1:4], axis=1)
        return np.asarray([
            1.0 - np.max(np.abs(pitch_error)),
            2.0 * np.pi / 180.0 - np.max(heading_delta),
            0.02 - np.linalg.norm(coefficients_np),
        ])

    linear_solution = minimize(
        hold_cost,
        np.zeros(len(normalized_directions), dtype=np.float64),
        method="SLSQP",
        bounds=[(-0.02, 0.02)] * len(normalized_directions),
        constraints={"type": "ineq", "fun": root_constraints},
        options={"maxiter": 100, "ftol": 1e-12, "disp": False},
    )
    coefficient = torch.tensor(linear_solution.x, device=z0.device, dtype=torch.float32)
    combined = torch.zeros_like(z0, dtype=torch.float32)
    for index, direction in enumerate(normalized_directions):
        combined = combined + coefficient[index] * direction
    combined_rms = torch.sqrt(combined.square().mean()).item()
    actual_validation = []
    for alpha in (1.0, 0.5, 0.25, 0.125):
        trial_noise = z0 + (float(alpha) * combined).to(dtype=z0.dtype)
        with torch.no_grad():
            trial = differentiable_generate(
                model=model, scheduler=scheduler, prompt_emb=prompt_emb,
                prompt_emb_null=prompt_emb_null, initial_noise=trial_noise,
                valid_mask=valid_mask, ref_motion=ref_motion,
                ref_motion_mask=ref_motion_mask, condition_on_text=condition_on_text,
                attend_to_text_mask=attend_to_text_mask, dtype=dtype,
                config=no_grad_config,
            )
        trial_features = _subspace_response_vector(trial.official_pre_cast, motion_mean, motion_std, valid_mask)
        target_features = _subspace_response_vector(official_result.official_pre_cast, motion_mean, motion_std, valid_mask)
        pitch_error = (trial_features[:, 0] - (target_features[:, 0] - float(target_delta_deg))).abs()
        actual_validation.append({
            "alpha": float(alpha),
            "source_delta_rms": float(torch.sqrt((trial_noise.float() - z0.float()).square().mean()).item()),
            "pitch_mae_deg": float(pitch_error.mean().item()),
            "pitch_p95_deg": float(torch.quantile(pitch_error, 0.95).item()),
            "feature_max_abs_change": float((trial_features - target_features).abs().max().item()),
            "root_target_predicted_pass": bool(np.max(np.abs((baseline_np + alpha * np.einsum("d,dnf->nf", linear_solution.x, response_np))[:, 0] - target_np[:, 0])) <= 1.0),
        })
        del trial
    for parameter, requires_grad in zip(parameters, original_requires_grad):
        parameter.requires_grad_(requires_grad)
    model.train(was_training)
    return {
        "protocol": "diagnostic_oracle_source_noise_subspace",
        "direction_seed": direction_seed,
        "direction_count": len(normalized_directions),
        "direction_records": direction_records,
        "rms_values": [float(value) for value in rms_values],
        "feature_names": ["root_pitch_deg", "heading_x", "heading_y", "heading_z", "trunk_x", "trunk_y", "trunk_z", "left_foot_xyz", "right_foot_xyz", "root_translation_xyz"],
        "feature_count": int(baseline_features.shape[-1]),
        "feature_frame_count": int(baseline_features.shape[0]),
        "actual_perturbations": actual_perturbations,
        "response_matrices": response,
        "baseline_features": baseline_features.detach().cpu(),
        "linear_solver": {
            "method": "scipy.optimize.SLSQP",
            "success": bool(linear_solution.success),
            "message": str(linear_solution.message),
            "iterations": int(getattr(linear_solution, "nit", 0)),
            "objective": float(linear_solution.fun),
            "coefficients": [float(value) for value in linear_solution.x],
            "constraint_margins": [float(value) for value in root_constraints(linear_solution.x)],
            "combined_source_delta_rms": float(combined_rms),
        },
        "actual_validation": actual_validation,
    }


def _comparison(left: torch.Tensor, right: torch.Tensor) -> dict:
    if left.shape != right.shape:
        return {
            "bitwise_equal": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "max_abs_error": math.inf,
        }
    difference = (left.float() - right.float()).abs()
    return {
        "bitwise_equal": bool(torch.equal(left, right)),
        "shape": list(left.shape),
        "max_abs_error": float(difference.max().item()),
        "different_elements": int(torch.count_nonzero(left != right).item()),
    }


def run_source_noise_reproduction_gate(
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
    sampler_config: DifferentiableSamplerConfig,
    gate_config: SourceNoiseGateConfig,
) -> dict:
    """Compare ``G(z0)`` to M0 and backpropagate one root-only objective."""

    if sampler_config.num_inference_steps != 50:
        raise ValueError("the reproduction stop gate requires exactly 50 steps")
    if official_result.initial_noise.shape[0] != 1:
        raise ValueError("the reproduction stop gate requires batch=1")

    was_training = model.training
    parameters = list(model.parameters())
    original_requires_grad = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    model.eval()

    device = official_result.initial_noise.device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    z0 = official_result.initial_noise.detach().clone().requires_grad_(True)
    differentiable_result = differentiable_generate(
        model=model,
        scheduler=scheduler,
        prompt_emb=prompt_emb,
        prompt_emb_null=prompt_emb_null,
        initial_noise=z0,
        valid_mask=valid_mask,
        ref_motion=ref_motion,
        ref_motion_mask=ref_motion_mask,
        condition_on_text=condition_on_text,
        attend_to_text_mask=attend_to_text_mask,
        dtype=dtype,
        config=sampler_config,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - started
    forward_peak_allocated = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    )
    forward_peak_reserved = (
        torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0
    )

    comparisons = {
        "raw": _comparison(differentiable_result.raw, official_result.raw),
        "official_pre_cast": _comparison(
            differentiable_result.official_pre_cast,
            official_result.official_pre_cast,
        ),
        "official": _comparison(differentiable_result.official, official_result.official),
    }
    objective = _root_objective(
        candidate_norm=differentiable_result.official_pre_cast,
        baseline_norm=official_result.official_pre_cast,
        valid_mask=valid_mask,
        motion_mean=motion_mean,
        motion_std=motion_std,
        target_delta_deg=gate_config.target_delta_deg,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_started = time.perf_counter()
    gradient = torch.autograd.grad(objective, z0, retain_graph=False)[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - backward_started
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0
    )

    gradient_finite = bool(torch.isfinite(gradient).all().item())
    gradient_nonzero = int(torch.count_nonzero(gradient).item())
    bitwise_reproduction = all(
        comparison["bitwise_equal"] for comparison in comparisons.values()
    )
    memory_pass = peak_reserved < gate_config.max_reserved_mib
    passed = bitwise_reproduction and gradient_finite and gradient_nonzero > 0 and memory_pass
    record = {
        "protocol": PROTOCOL_NAME,
        "status": "PASSED" if passed else "STOPPED",
        "passed": passed,
        "sampler": asdict(sampler_config),
        "gate": asdict(gate_config),
        "batch_size": int(z0.shape[0]),
        "sequence_length": int(z0.shape[1]),
        "channels": int(z0.shape[2]),
        "dtype": str(dtype).removeprefix("torch."),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "comparisons": comparisons,
        "bitwise_reproduction": bitwise_reproduction,
        "objective_deg_squared": float(objective.detach().item()),
        "gradient": {
            "finite": gradient_finite,
            "nonzero_elements": gradient_nonzero,
            "total_elements": int(gradient.numel()),
            "l2_norm": float(torch.linalg.vector_norm(gradient.float()).item()),
            "max_abs": float(gradient.float().abs().max().item()),
        },
        "timing_seconds": {
            "forward": forward_seconds,
            "backward": backward_seconds,
            "total": forward_seconds + backward_seconds,
        },
        "memory_mib": {
            "forward_peak_allocated": forward_peak_allocated,
            "forward_peak_reserved": forward_peak_reserved,
            "forward_backward_peak_allocated": peak_allocated,
            "forward_backward_peak_reserved": peak_reserved,
            "reserved_limit": gate_config.max_reserved_mib,
            "pass": memory_pass,
        },
    }

    del gradient, objective, differentiable_result, z0
    for parameter, requires_grad in zip(parameters, original_requires_grad):
        parameter.requires_grad_(requires_grad)
    model.train(was_training)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


__all__ = [
    "PROTOCOL_NAME",
    "DifferentiableSamplerConfig",
    "SourceNoiseGateConfig",
    "differentiable_generate",
    "run_source_noise_reproduction_gate",
]
