"""M2: D-Flow-style optimisation of the initial source noise."""

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
from guidance.sampling_common import masked_angle_loss


METHOD_NAME = "M2_DFLOW_SOURCE_OPTIMIZATION"
PROTOCOL_NAME = "vimogen_m2_dflow_source_optimization_c0_v1"


@dataclass(frozen=True)
class M2Config:
    learning_rate: float = 0.01
    iterations: int = 12
    source_regularization: float = 0.01
    gradient_clip_norm: float = 10.0
    early_stop_mae_deg: float = 1.0
    early_stop_patience: int = 2

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "M2Config":
        cfg = cls(**dict(values or {}))
        if cfg.learning_rate <= 0 or cfg.iterations < 1:
            raise ValueError("M2 requires positive learning rate and iterations")
        if cfg.source_regularization < 0 or cfg.gradient_clip_norm <= 0:
            raise ValueError("M2 regularization must be nonnegative and clip positive")
        if cfg.early_stop_mae_deg <= 0 or cfg.early_stop_patience < 1:
            raise ValueError("M2 early-stop settings must be positive")
        return cfg


def _rollout(vimogen: Any, source: torch.Tensor, request: GuidanceRequest) -> torch.Tensor:
    if not hasattr(vimogen, "rollout"):
        raise TypeError("M2 runtime must expose rollout(source_noise, request=...)")
    result = vimogen.rollout(source, request=request, differentiable=True)
    motion = result.motion if hasattr(result, "motion") else result
    if not isinstance(motion, torch.Tensor) or motion.shape != request.baseline_motion.shape:
        raise ValueError("M2 rollout must return motion with shape [B,T,276]")
    return motion


class M2DFlowSourceOptimization:
    """Optimise only ``z0``; the paired M0 target remains immutable."""

    name = METHOD_NAME

    def run(
        self,
        vimogen: Any,
        request: GuidanceRequest,
        cfg: Mapping[str, Any] | None = None,
    ) -> GuidedSample:
        if request.constraint_pack is not ConstraintPack.C0:
            raise NotImplementedError("M2 v1 implements C0 only")
        config = M2Config.from_mapping(cfg)
        initial = request.base_noise.detach().float()
        source = initial.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([source], lr=config.learning_rate)
        diagnostics = GuidanceDiagnostics()
        best_motion = request.baseline_motion.detach().clone()
        best_source = initial.clone()
        best_key = (float("inf"), float("inf"))
        passing_without_improvement = 0
        history: list[dict[str, float | int | bool]] = []
        valid = request.shared_evidence.valid_mask.to(initial.device)
        target = request.shared_evidence.target_angle_curve_deg.to(initial.device)
        with RunTimer() as timer:
            try:
                for iteration in range(config.iterations):
                    optimizer.zero_grad(set_to_none=True)
                    motion = _rollout(vimogen, source, request)
                    physical = authoritative_motion(motion, valid_mask=valid).motion
                    angle_loss, residual = masked_angle_loss(physical, target, valid)
                    source_loss = (source - initial).square().mean()
                    objective = angle_loss + config.source_regularization * source_loss
                    if not torch.isfinite(objective):
                        raise FloatingPointError("M2 objective became non-finite")
                    objective.backward()
                    diagnostics.backward_count += 1
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        [source], config.gradient_clip_norm
                    )
                    mae = float(residual[valid].abs().mean().detach().cpu())
                    key = (mae, float(source_loss.detach().cpu()))
                    improved = key < best_key
                    if improved:
                        best_key = key
                        best_motion = physical.detach().clone()
                        best_source = source.detach().clone()
                        passing_without_improvement = 0
                    elif mae <= config.early_stop_mae_deg:
                        passing_without_improvement += 1
                    history.append(
                        {
                            "iteration": iteration,
                            "objective": float(objective.detach().cpu()),
                            "angle_mae_deg": mae,
                            "source_regularization": float(source_loss.detach().cpu()),
                            "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
                            "passed_angle_gate": mae <= config.early_stop_mae_deg,
                        }
                    )
                    diagnostics.full_rollout_count += 1
                    diagnostics.model_nfe += int(getattr(vimogen, "nfe_per_rollout", 1))
                    if mae <= config.early_stop_mae_deg and (
                        passing_without_improvement >= config.early_stop_patience
                    ):
                        break
                    optimizer.step()
                    diagnostics.solver_iterations += 1
                final_residual = angle_error_deg(best_motion, target)[valid]
                update = best_source - initial
                diagnostics.angle_residual_mae_deg = float(final_residual.abs().mean().cpu())
                diagnostics.angle_residual_p95_deg = float(
                    torch.quantile(final_residual.abs(), 0.95).cpu()
                )
                diagnostics.state_update_norm = float(torch.linalg.vector_norm(update).cpu())
                diagnostics.max_update_norm = float(update.abs().max().cpu())
                diagnostics.smplx_forward_count = diagnostics.full_rollout_count
                diagnostics.extra.update(
                    {
                        "protocol": PROTOCOL_NAME,
                        "optimized_variable": "initial_source_noise_only",
                        "target_redefined_during_optimization": False,
                        "iteration_history": history,
                    }
                )
                status = "COMPLETED"
            except Exception as error:
                best_motion = request.baseline_motion.detach().clone()
                diagnostics.fallback_used = True
                diagnostics.failure_reason = repr(error)
                diagnostics.nonfinite_count = int((~torch.isfinite(best_motion)).sum().cpu())
                status = "FAILED_FALLBACK_M0"
        diagnostics.wall_time_sec = timer.wall_time_sec
        diagnostics.peak_gpu_mem_gb = timer.peak_gpu_mem_gb
        return GuidedSample(best_motion, diagnostics, status)
