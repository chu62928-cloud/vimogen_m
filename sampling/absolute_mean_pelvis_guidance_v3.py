"""Tail-safe full-FK absolute mean pelvis guidance (protocol v3).

The loss and sagittal angle semantics are inherited from v2.  The only
changed boundary is the v3 tail-safe reconciliation, which prevents the
synthetic T+1 endpoint from feeding back into the last physical output pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from motion_rep.consistency_v3 import (
    differentiable_forward_kinematics,
    load_smplx_neutral_22_skeleton,
    reconcile_motion_tensor_v3,
)
from motion_rep.finalize import finalize_motion
from sampling.absolute_mean_pelvis_guidance_v2 import (
    AbsoluteMeanFinalOutputsV2,
    AbsoluteMeanPelvisConfigV2,
    AbsoluteMeanPelvisGuidanceV2,
    _align_statistic,
    _masked_mean,
    _prefix_pose_mask,
    pelvis_angle_curve,
)
from motion_rep.sagittal_pelvis_angle import (
    apply_person_right_axis_rotation,
    pelvis_sagittal_tilt_degrees,
)
from motion_rep.unified_finalizer import recover_motion_stream


PROTOCOL_NAME = "vimogen_absolute_mean_pelvis_v3_tail_safe"
AbsoluteMeanPelvisConfigV3 = AbsoluteMeanPelvisConfigV2


@dataclass(frozen=True)
class AbsoluteMeanFinalOutputsV3:
    g0: torch.Tensor
    g1: torch.Tensor
    g0_valid_mask: torch.Tensor
    g1_valid_mask: torch.Tensor
    terminal_records: tuple[dict[str, Any], ...]
    protocol: str = PROTOCOL_NAME


class AbsoluteMeanPelvisGuidanceV3(AbsoluteMeanPelvisGuidanceV2):
    """v2 loss semantics with the v3 tail-safe full-FK boundary."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_diagnostics["protocol"] = PROTOCOL_NAME

    def slice(self, index: int) -> "AbsoluteMeanPelvisGuidanceV3":
        if self.baseline_motion_norm.shape[0] == 1:
            return self
        mean, std = self.mean, self.std
        if mean.ndim == 2 and mean.shape[0] == self.baseline_motion_norm.shape[0]:
            mean, std = mean[index:index + 1], std[index:index + 1]
        return AbsoluteMeanPelvisGuidanceV3(
            baseline_motion_norm=self.baseline_motion_norm[index:index + 1],
            valid_mask=self.valid_mask[index:index + 1],
            mean=mean,
            std=std,
            target_mean_deg=self.target_mean_deg,
            config=self.config,
        )

    def protocol_record(self) -> dict[str, Any]:
        record = super().protocol_record()
        record.update(
            {
                "protocol": PROTOCOL_NAME,
                "post_update_boundary": "tail_safe_fused_root_then_fk22_then_repack_all_276",
                "tail_boundary": "output_only_window_then_hidden_pose_hold_last",
            }
        )
        return record

    def _reconcile(self, motion_norm: torch.Tensor, *, output_standardized: bool):
        skeleton = (
            None
            if self.config.skeleton_path is None
            else load_smplx_neutral_22_skeleton(self.config.skeleton_path)
        )
        return reconcile_motion_tensor_v3(
            motion_norm,
            fusion_window=self.config.fusion_window,
            anchor_weight=self.config.anchor_weight,
            root_rotation_anchor_weight=self.config.anchor_weight,
            valid_mask=self.valid_mask.to(motion_norm.device),
            mean=self.mean,
            std=self.std,
            input_standardized=True,
            output_standardized=output_standardized,
            output_dtype=torch.float32,
            skeleton=skeleton,
        )

    def correct_velocity(self, *args, **kwargs):
        corrected, diagnostics = super().correct_velocity(*args, **kwargs)
        diagnostics["protocol"] = PROTOCOL_NAME
        diagnostics["post_update_boundary"] = "tail_safe_fused_root_then_fk22_then_repack_all_276"
        self.last_diagnostics = diagnostics
        return corrected, diagnostics

    def _terminal_single(
        self,
        g0_norm: torch.Tensor,
        mask: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        physical_result = self._reconcile(g0_norm.unsqueeze(0), output_standardized=False)
        physical = physical_result.motion[0]
        out_mask = physical_result.valid_mask[0]
        angle = pelvis_angle_curve(physical)
        mean_before = float(_masked_mean(angle.unsqueeze(0), out_mask.unsqueeze(0))[0].detach().cpu())
        residual = self.target_mean_deg - mean_before
        record: dict[str, Any] = {
            "target_mean_deg": self.target_mean_deg,
            "mean_before_deg": mean_before,
            "residual_before_deg": residual,
            "eligible": abs(residual) <= self.config.terminal_max_deg + self.config.eps,
            "triggered": False,
            "failed_residual_over_limit": False,
            "applied_deg": 0.0,
            "mean_after_deg": mean_before,
        }
        if not self.config.terminal_enabled:
            record["eligible"] = False
            record["disabled"] = True
            return g0_norm, out_mask, record
        if abs(residual) > self.config.terminal_max_deg + self.config.eps:
            record["failed_residual_over_limit"] = True
            return g0_norm, out_mask, record

        stream = recover_motion_stream(physical)
        candidates = torch.linspace(
            -self.config.terminal_max_deg,
            self.config.terminal_max_deg,
            401,
            dtype=physical.dtype,
            device=physical.device,
        )
        candidate_count = candidates.numel()
        root_stream = stream.root_rotation
        root_candidates = root_stream.unsqueeze(0).expand(candidate_count, -1, -1, -1)
        candidate_degrees = candidates[:, None].expand(-1, root_stream.shape[0])
        roots = apply_person_right_axis_rotation(
            root_candidates, candidate_degrees, eps=self.config.eps
        )
        skeleton = (
            load_smplx_neutral_22_skeleton(self.config.skeleton_path)
            if self.config.skeleton_path is not None
            else load_smplx_neutral_22_skeleton()
        )
        body_stream = stream.body_rotations
        valid_length = int(out_mask.sum().item())
        if valid_length:
            body_stream = body_stream.clone()
            body_stream[valid_length:] = body_stream[valid_length - 1 : valid_length]
        body_candidates = body_stream.unsqueeze(0).expand(
            candidate_count, -1, -1, -1, -1
        )
        translation_candidates = stream.root_translation.unsqueeze(0).expand(
            candidate_count, -1, -1
        )
        candidate_fk = differentiable_forward_kinematics(
            body_candidates,
            roots,
            translation_candidates,
            skeleton=skeleton,
        )
        candidate_finalized = finalize_motion(
            body_candidates,
            candidate_fk.joints,
            roots,
            translation_candidates,
            valid_mask=_prefix_pose_mask(out_mask).unsqueeze(0).expand(
                candidate_count, -1
            ),
        )
        probe_angle = pelvis_sagittal_tilt_degrees(
            roots[..., :-1, :, :], eps=self.config.eps
        )
        probe_mean = _masked_mean(probe_angle, candidate_finalized.valid_mask)
        best = torch.argmin((probe_mean - self.target_mean_deg).abs())
        applied = candidates[best]
        finalized = type(candidate_finalized)(
            motion=candidate_finalized.motion[best],
            valid_mask=candidate_finalized.valid_mask[best],
        )
        mean_aligned = _align_statistic(mean, finalized.motion, "mean")
        std_aligned = _align_statistic(std, finalized.motion, "std")
        g1 = ((finalized.motion - mean_aligned) / std_aligned).masked_fill(
            ~finalized.valid_mask.unsqueeze(-1), 0
        )
        mean_after = float(probe_mean[best].detach().cpu())
        record.update(
            {
                "triggered": abs(float(applied.detach().cpu())) > self.config.eps,
                "applied_deg": float(applied.detach().cpu()),
                "mean_after_deg": mean_after,
            }
        )
        return g1, finalized.valid_mask, record

    @torch.no_grad()
    def finalize_outputs(self, official_norm: torch.Tensor) -> AbsoluteMeanFinalOutputsV3:
        g0_result = self._reconcile(official_norm.float(), output_standardized=True)
        g0 = g0_result.motion
        g1_rows, masks, records = [], [], []
        for index in range(g0.shape[0]):
            mean, std = self.mean, self.std
            if mean.ndim == 2 and mean.shape[0] == g0.shape[0]:
                mean, std = mean[index:index + 1], std[index:index + 1]
            g1, g1_mask, record = self._terminal_single(
                g0[index], self.valid_mask[index].to(g0.device), mean, std
            )
            g1_rows.append(g1)
            masks.append(g1_mask)
            records.append(record)
        return AbsoluteMeanFinalOutputsV3(
            g0=g0,
            g1=torch.stack(g1_rows),
            g0_valid_mask=g0_result.valid_mask,
            g1_valid_mask=torch.stack(masks),
            terminal_records=tuple(records),
        )


__all__ = [
    "AbsoluteMeanFinalOutputsV3",
    "AbsoluteMeanPelvisConfigV3",
    "AbsoluteMeanPelvisGuidanceV3",
    "PROTOCOL_NAME",
    "pelvis_angle_curve",
]
