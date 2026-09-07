"""Differentiable helpers shared by sampling-time guidance methods."""

from __future__ import annotations

import torch

from geometry.authoritative_motion import authoritative_motion
from geometry.pelvis_angle import angle_error_deg


def align_stat(value: torch.Tensor, motion: torch.Tensor, name: str) -> torch.Tensor:
    value = value.to(device=motion.device, dtype=torch.float32)
    if value.shape[-1] != motion.shape[-1]:
        raise ValueError(f"{name} must have {motion.shape[-1]} channels")
    if value.ndim == 1:
        value = value[None, None, :]
    elif value.ndim == 2:
        value = value[:, None, :]
    try:
        return torch.broadcast_to(value, motion.shape)
    except RuntimeError as error:
        raise ValueError(f"{name} is not broadcastable to motion") from error


def normalized_from_physical(
    motion: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return (motion.float() - align_stat(mean, motion, "mean")) / align_stat(
        std, motion, "std"
    )


def predicted_clean(
    x_sigma: torch.Tensor, velocity: torch.Tensor, sigma: torch.Tensor | float
) -> torch.Tensor:
    value = torch.as_tensor(sigma, dtype=torch.float32, device=x_sigma.device)
    while value.ndim < x_sigma.ndim:
        value = value.unsqueeze(-1)
    return x_sigma.float() - value * velocity.float()


def velocity_from_clean(
    x_sigma: torch.Tensor,
    clean: torch.Tensor,
    sigma: torch.Tensor | float,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    value = torch.as_tensor(sigma, dtype=torch.float32, device=x_sigma.device)
    if torch.any(value.abs() <= eps):
        raise ValueError("cannot recompose velocity at terminal sigma")
    while value.ndim < x_sigma.ndim:
        value = value.unsqueeze(-1)
    return (x_sigma.float() - clean.float()) / value


def authoritative_normalized(
    clean_norm: torch.Tensor,
    valid_mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected = authoritative_motion(
        clean_norm,
        valid_mask=valid_mask,
        mean=mean,
        std=std,
        input_standardized=True,
        output_standardized=False,
    )
    physical = projected.motion
    return physical, normalized_from_physical(physical, mean, std)


def masked_angle_loss(
    physical_motion: torch.Tensor,
    target_curve_deg: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = angle_error_deg(
        physical_motion, target_curve_deg.to(physical_motion.device)
    )
    selected = residual[valid_mask]
    if not selected.numel():
        raise ValueError("angle loss has no valid frames")
    return selected.square().mean(), residual


def masked_rms(value: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    selected = value[valid_mask]
    if not selected.numel():
        return torch.zeros((), dtype=value.dtype, device=value.device)
    return torch.sqrt(selected.square().mean())


def clip_rms(
    value: torch.Tensor,
    valid_mask: torch.Tensor,
    limit: float,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    rms = masked_rms(value, valid_mask)
    scale = torch.clamp(
        torch.as_tensor(limit, dtype=value.dtype, device=value.device)
        / rms.clamp_min(eps),
        max=1.0,
    )
    return value * scale, rms
