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
