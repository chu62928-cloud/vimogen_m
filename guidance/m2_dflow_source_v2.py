"""M2-v2: protected, per-sample source-noise optimisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from geometry.authoritative_motion import authoritative_motion
from geometry.pelvis_angle import angle_error_deg
from guidance.base import (
    ConstraintPack,
    GuidedSample,
    GuidanceDiagnostics,
    GuidanceRequest,
    RunTimer,
)
from guidance.m2_dflow_source import _rollout
from motion_rep.phase1 import MOTION_LAYOUT


METHOD_NAME = "M2_DFLOW_SOURCE_OPTIMIZATION_V2"
PROTOCOL_NAME = "vimogen_m2_dflow_source_optimization_c0_v2"


@dataclass(frozen=True)
class M2V2Config:
    learning_rate: float = 0.005
    iterations: int = 8
    source_regularization: float = 0.001
    content_weight: float = 0.01
    root_weight: float = 0.05
    gradient_clip_norm: float = 10.0
    source_trust_radius: float = 25.0
    step_trust_radius: float = 5.0
    early_stop_mae_deg: float = 1.0
    early_stop_patience: int = 2

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "M2V2Config":
        cfg = cls(**dict(values or {}))
        if cfg.learning_rate <= 0 or cfg.iterations < 1:
            raise ValueError("M2-v2 requires positive learning rate and iterations")
        if min(cfg.source_regularization, cfg.content_weight, cfg.root_weight) < 0:
            raise ValueError("M2-v2 objective weights must be non-negative")
        if min(
            cfg.gradient_clip_norm,
            cfg.source_trust_radius,
            cfg.step_trust_radius,
            cfg.early_stop_mae_deg,
        ) <= 0:
            raise ValueError("M2-v2 clip, trust, and early-stop limits must be positive")
        if cfg.early_stop_patience < 1:
            raise ValueError("M2-v2 early_stop_patience must be positive")
        return cfg


def _masked_mean_per_sample(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.to(value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(value)
    return (value * expanded).flatten(1).sum(-1) / expanded.flatten(1).sum(-1).clamp_min(1.0)


def _clip_rows_(gradient: torch.Tensor, limit: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(gradient.flatten(1), dim=-1)
    scale = (float(limit) / norms.clamp_min(1.0e-12)).clamp(max=1.0)
    gradient.mul_(scale.reshape(-1, *([1] * (gradient.ndim - 1))))
    return norms


def _project_rows_(value: torch.Tensor, centre: torch.Tensor, radius: float) -> torch.Tensor:
    delta = value - centre
    norms = torch.linalg.vector_norm(delta.flatten(1), dim=-1)
    scale = (float(radius) / norms.clamp_min(1.0e-12)).clamp(max=1.0)
    value.copy_(centre + delta * scale.reshape(-1, *([1] * (value.ndim - 1))))
    return norms


class M2DFlowSourceOptimizationV2:
    """Optimise a batch jointly while selecting and protecting each row independently."""

    name = METHOD_NAME

    def run(
        self,
        vimogen: Any,
        request: GuidanceRequest,
        cfg: Mapping[str, Any] | None = None,
    ) -> GuidedSample:
        if request.constraint_pack is not ConstraintPack.C0:
            raise NotImplementedError("M2-v2 implements C0 only")
        config = M2V2Config.from_mapping(cfg)
        diagnostics = GuidanceDiagnostics()
        batch = request.baseline_motion.shape[0]
        if float(request.target_dose_deg) == 0.0:
            diagnostics.extra.update(
                {
                    "protocol": PROTOCOL_NAME,
                    "optimization_scope": "independent_per_sample",
                    "zero_dose_bypass": True,
                    "per_sample_best_iteration": [-1] * batch,
                    "source_trust_region_hit_count": [0] * batch,
                    "step_trust_region_hit_count": [0] * batch,
                    "iteration_history": [[] for _ in range(batch)],
                }
            )
            return GuidedSample(
                request.baseline_motion.detach().clone(),
                diagnostics,
                "COMPLETED_ZERO_DOSE_BYPASS",
            )

        initial = request.base_noise.detach().float()
        source = initial.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([source], lr=config.learning_rate)
        valid = request.shared_evidence.valid_mask.to(initial.device)
        target = request.shared_evidence.target_angle_curve_deg.to(initial.device)
        baseline = authoritative_motion(
            request.baseline_motion.float(), valid_mask=valid
        ).motion.detach()
        best_motion = baseline.clone()
        best_source = initial.clone()
        best_objective = torch.full(
            (batch,), float("inf"), dtype=initial.dtype, device=initial.device
        )
        best_iteration = torch.full((batch,), -1, dtype=torch.long, device=initial.device)
        passing_without_improvement = torch.zeros(
            batch, dtype=torch.long, device=initial.device
        )
        source_hits = torch.zeros(batch, dtype=torch.long, device=initial.device)
        step_hits = torch.zeros(batch, dtype=torch.long, device=initial.device)
        histories: list[list[dict[str, float | int | bool]]] = [
            [] for _ in range(batch)
        ]
        status = "COMPLETED"

        with RunTimer() as timer:
            try:
                for iteration in range(config.iterations):
                    optimizer.zero_grad(set_to_none=True)
                    motion = _rollout(vimogen, source, request)
                    physical = authoritative_motion(motion, valid_mask=valid).motion
                    residual = angle_error_deg(physical, target)
                    angle_loss = _masked_mean_per_sample(residual.square(), valid)
                    source_loss = (source - initial).square().flatten(1).mean(-1)
                    content_loss = _masked_mean_per_sample(
                        (physical - baseline).square(), valid
                    )
                    root_loss = _masked_mean_per_sample(
                        (
                            physical[..., MOTION_LAYOUT.root_translation]
                            - baseline[..., MOTION_LAYOUT.root_translation]
                        ).square(),
                        valid,
                    )
                    objective = (
                        angle_loss
                        + config.source_regularization * source_loss
                        + config.content_weight * content_loss
                        + config.root_weight * root_loss
                    )
                    if not torch.isfinite(objective).all():
                        raise FloatingPointError("M2-v2 objective became non-finite")
                    objective.sum().backward()
                    diagnostics.backward_count += 1
                    if source.grad is None:
                        raise RuntimeError("M2-v2 source gradient is missing")
                    gradient_norm = _clip_rows_(source.grad, config.gradient_clip_norm)
                    mae = _masked_mean_per_sample(residual.abs(), valid)
                    improved = objective.detach() < best_objective
                    if improved.any():
                        best_objective = torch.where(
                            improved, objective.detach(), best_objective
                        )
                        best_motion[improved] = physical.detach()[improved]
                        best_source[improved] = source.detach()[improved]
                        best_iteration[improved] = iteration
                        passing_without_improvement[improved] = 0
                    passed = mae <= config.early_stop_mae_deg
                    passing_without_improvement[passed & ~improved] += 1
                    for index in range(batch):
                        histories[index].append(
                            {
                                "iteration": iteration,
                                "objective": float(objective[index].detach().cpu()),
                                "angle_mae_deg": float(mae[index].detach().cpu()),
                                "source_regularization": float(source_loss[index].detach().cpu()),
                                "content_proxy": float(content_loss[index].detach().cpu()),
                                "root_proxy": float(root_loss[index].detach().cpu()),
                                "gradient_norm": float(gradient_norm[index].detach().cpu()),
                                "passed_angle_gate": bool(passed[index].item()),
                                "best_so_far": bool(improved[index].item()),
                            }
                        )
                    diagnostics.full_rollout_count += 1
                    diagnostics.model_nfe += int(getattr(vimogen, "nfe_per_rollout", 1))
                    if bool(
                        torch.all(
                            passed
                            & (passing_without_improvement >= config.early_stop_patience)
                        )
                    ):
                        break
                    previous = source.detach().clone()
                    optimizer.step()
                    diagnostics.solver_iterations += 1
                    with torch.no_grad():
                        step_norms = _project_rows_(
                            source, previous, config.step_trust_radius
                        )
                        step_hits += step_norms.gt(config.step_trust_radius).long()
                        source_norms = _project_rows_(
                            source, initial, config.source_trust_radius
                        )
                        source_hits += source_norms.gt(config.source_trust_radius).long()

                final_residual = angle_error_deg(best_motion, target).abs()
                per_sample_mae = _masked_mean_per_sample(final_residual, valid)
                per_sample_p95 = torch.stack(
                    [
                        torch.quantile(final_residual[index][valid[index]], 0.95)
                        for index in range(batch)
                    ]
                )
                update = best_source - initial
                update_norms = torch.linalg.vector_norm(update.flatten(1), dim=-1)
                diagnostics.angle_residual_mae_deg = float(per_sample_mae.mean().cpu())
                diagnostics.angle_residual_p95_deg = float(per_sample_p95.max().cpu())
                diagnostics.state_update_norm = float(update_norms.max().cpu())
                diagnostics.max_update_norm = float(update.abs().max().cpu())
                diagnostics.smplx_forward_count = diagnostics.full_rollout_count
                diagnostics.rejected_steps = int((source_hits + step_hits).sum().cpu())
                diagnostics.extra.update(
                    {
                        "protocol": PROTOCOL_NAME,
                        "optimized_variable": "initial_source_noise_only",
                        "optimization_scope": "independent_per_sample",
                        "best_state_objective": "angle+source+content+root",
                        "target_redefined_during_optimization": False,
                        "zero_dose_bypass": False,
                        "per_sample_best_iteration": best_iteration.detach().cpu().tolist(),
                        "per_sample_best_objective": best_objective.detach().cpu().tolist(),
                        "per_sample_source_delta_norm": update_norms.detach().cpu().tolist(),
                        "source_trust_region_hit_count": source_hits.detach().cpu().tolist(),
                        "step_trust_region_hit_count": step_hits.detach().cpu().tolist(),
                        "iteration_history": histories,
                    }
                )
            except Exception as error:
                best_motion = request.baseline_motion.detach().clone()
                diagnostics.fallback_used = True
                diagnostics.failure_reason = repr(error)
                diagnostics.nonfinite_count = int((~torch.isfinite(best_motion)).sum().cpu())
                status = "FAILED_FALLBACK_M0"
        diagnostics.wall_time_sec = timer.wall_time_sec
        diagnostics.peak_gpu_mem_gb = timer.peak_gpu_mem_gb
        return GuidedSample(best_motion, diagnostics, status)


__all__ = [
    "METHOD_NAME",
    "PROTOCOL_NAME",
    "M2V2Config",
    "M2DFlowSourceOptimizationV2",
]
