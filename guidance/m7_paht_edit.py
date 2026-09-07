"""M7 PAHT geometric-edit baseline under the corrected theoretical scope."""

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
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe, encode_rot6d
from motion_rep.sagittal_pelvis_angle import apply_person_right_axis_rotation


METHOD_NAME = "M7_PAHT_GEOMETRIC_EDIT"
PROTOCOL_NAME = "vimogen_m7_paht_geometric_edit_c0_v1"


@dataclass(frozen=True)
class M7Config:
    eps: float = 1.0e-6

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "M7Config":
        values = dict(values or {})
        unknown = set(values) - {"eps"}
        if unknown:
            raise ValueError(f"unknown M7 settings: {sorted(unknown)}")
        cfg = cls(**values)
        if cfg.eps <= 0:
            raise ValueError("eps must be positive")
        return cfg


class M7PAHTGeometricEdit:
    name = METHOD_NAME

    def run(
        self,
        vimogen: Any,
        request: GuidanceRequest,
        cfg: Mapping[str, Any] | None = None,
    ) -> GuidedSample:
        del vimogen
        config = M7Config.from_mapping(cfg)
        if request.constraint_pack is not ConstraintPack.C0:
            raise NotImplementedError("M7 v1 currently implements only frozen C0")
        valid = request.shared_evidence.valid_mask.to(request.baseline_motion.device)
        diagnostics = GuidanceDiagnostics(full_rollout_count=0)
        with RunTimer() as timer:
            try:
                baseline = authoritative_motion(
                    request.baseline_motion,
                    valid_mask=valid,
                ).motion
                candidate = baseline.clone()
                root = decode_rot6d_safe(
                    baseline[..., MOTION_LAYOUT.root_rotation]
                )
                edited_root = apply_person_right_axis_rotation(
                    root,
                    float(request.target_dose_deg),
                    eps=config.eps,
                )
                candidate[..., MOTION_LAYOUT.root_rotation] = encode_rot6d(
                    edited_root
                )
                output = authoritative_motion(candidate, valid_mask=valid).motion
                residual = angle_error_deg(
                    output,
                    request.shared_evidence.target_angle_curve_deg.to(output.device),
                )[valid]
                update = (output - baseline)[valid]
                diagnostics.angle_residual_mae_deg = float(
                    residual.abs().mean().detach().cpu()
                )
                diagnostics.angle_residual_p95_deg = float(
                    torch.quantile(residual.abs(), 0.95).detach().cpu()
                )
                diagnostics.state_update_norm = float(
                    torch.linalg.vector_norm(update).detach().cpu()
                )
                diagnostics.max_update_norm = float(
                    update.abs().max().detach().cpu()
                )
                diagnostics.smplx_forward_count = int(output.shape[0])
                diagnostics.solver_iterations = 1
                diagnostics.extra.update(
                    {
                        "protocol": PROTOCOL_NAME,
                        "theoretical_role": "post_generation_geometric_edit_baseline",
                        "claims_generator_response": False,
                    }
                )
                status = "COMPLETED"
            except Exception as error:
                output = request.baseline_motion.detach().clone()
                diagnostics.fallback_used = True
                diagnostics.failure_reason = repr(error)
                diagnostics.nonfinite_count = int(
                    (~torch.isfinite(output)).sum().detach().cpu()
                )
                status = "FAILED_FALLBACK_M0"
        diagnostics.wall_time_sec = timer.wall_time_sec
        diagnostics.peak_gpu_mem_gb = timer.peak_gpu_mem_gb
        return GuidedSample(output, diagnostics, status)
