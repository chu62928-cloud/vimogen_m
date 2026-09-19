"""Small, explicit policies for offline terminal endpoint projection.

The sampling-time projector keeps its historical defaults.  This module is
the seam used by the v0.4.1 offline ablation so a terminal dead-zone cannot be
confused with the numerical convergence tolerance of the optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from evaluation.pelvis_contact_compensation_v3 import pelvis_pitch_delta_deg, target_root_rotation


@dataclass(frozen=True)
class TerminalProjectionPolicy:
    """Terminal controls that are safe to vary without changing sampling."""

    dead_zone_deg: float = 0.0
    pelvis_enabled: bool = True
    contact_enabled: bool = True

    def validate(self) -> None:
        if self.dead_zone_deg < 0.0 or self.dead_zone_deg >= 180.0:
            raise ValueError("terminal dead-zone must lie in [0, 180)")


def build_terminal_root_target(
    m0_root: torch.Tensor,
    source_root: torch.Tensor,
    target_dose_deg: float,
    *,
    valid_mask: torch.Tensor,
    policy: TerminalProjectionPolicy,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target roots, signed pre-cast error, and active-frame mask.

    ``error`` is ``target - actual`` in the frozen sagittal convention.  The
    target preserves exactly the requested dead-zone residual, while invalid
    frames and a disabled pelvis terminal pass retain the source root.
    """

    policy.validate()
    if m0_root.shape != source_root.shape or m0_root.shape[-2:] != (3, 3):
        raise ValueError("M0 and source roots must have equal [...,3,3] shapes")
    valid = torch.as_tensor(valid_mask, device=source_root.device, dtype=torch.bool)
    valid = torch.broadcast_to(valid, source_root.shape[:-2])
    actual = pelvis_pitch_delta_deg(m0_root, source_root)
    error = torch.remainder(float(target_dose_deg) - actual + 180.0, 360.0) - 180.0
    active = valid & policy.pelvis_enabled & (error.abs() > float(policy.dead_zone_deg))
    correction = torch.sign(error) * (error.abs() - float(policy.dead_zone_deg)).clamp_min(0.0)
    desired = actual + correction
    target = source_root.clone()
    if bool(valid.any()) and policy.pelvis_enabled:
        target[valid] = target_root_rotation(m0_root[valid], desired[valid])
    return target, error, active

