"""Frozen-M0 sole-marker gates for the staged local-pelvis experiments."""

from __future__ import annotations

from typing import Any

import torch

from geometry.contacts import freeze_marker_contact_evidence
from geometry.ground import estimate_ground_height


VERSION = "local_pelvis_frozen_m0_contact_v1"
MARKERS = (("left", "heel"), ("left", "toe"), ("right", "heel"), ("right", "toe"))


def _p95(value: torch.Tensor) -> float | None:
    if not value.numel():
        return None
    return float(torch.quantile(value.float(), 0.95))


def _unit_angle_deg(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left / torch.linalg.vector_norm(left, dim=-1, keepdim=True).clamp_min(1e-8)
    right = right / torch.linalg.vector_norm(right, dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.rad2deg(torch.acos((left * right).sum(-1).clamp(-1.0, 1.0)))


def frozen_contact_reference(markers: torch.Tensor, valid: torch.Tensor) -> dict[str, Any]:
    """Freeze ground and heel/toe masks from one authoritative M0, never a candidate."""
    if markers.ndim != 3 or markers.shape[0] != 4 or markers.shape[-1] != 3:
        raise ValueError("markers must have shape [4,T,3]")
    if valid.dtype is not torch.bool or valid.shape != (markers.shape[1],):
        raise ValueError("valid must have shape [T]")
    if not torch.isfinite(markers).all() or not valid.any():
        raise ValueError("M0 markers must be finite with valid frames")
    ground = estimate_ground_height(markers.permute(1, 0, 2)[None], valid[None], up_axis=2)
    mapping = {
        side: {
            marker: markers[index][None]
            for index, (name, marker) in enumerate(MARKERS) if name == side
        }
        for side in ("left", "right")
    }
    evidence = freeze_marker_contact_evidence(mapping, valid[None], ground, up_axis=2)
    contact = torch.stack([evidence.contact[side][name][0] for side, name in MARKERS])
    flat = torch.stack([evidence.flat_contact[side][name][0] for side, name in MARKERS])
    return {
        "version": VERSION,
        "ground_height_m": float(ground[0]),
        "contact_mask": contact,
        "flat_contact_mask": flat,
        "valid_mask": valid.clone(),
        "m0_markers": markers.clone(),
    }


def evaluate_frozen_contact(
    candidate_markers: torch.Tensor, reference: dict[str, Any], *, fps: float = 20.0,
) -> dict[str, Any]:
    base = reference["m0_markers"].float()
    candidate = candidate_markers.float()
    valid = reference["valid_mask"].bool()
    contact = reference["contact_mask"].bool()
    if candidate.shape != base.shape or base.shape[0] != 4 or base.shape[-1] != 3:
        raise ValueError("M0 and candidate markers must share [4,T,3]")
    if not torch.isfinite(candidate).all() or fps <= 0:
        raise ValueError("candidate markers and fps must be finite and valid")
    selected = contact & valid[None]
    displacement_mm = 1000.0 * torch.linalg.vector_norm(candidate - base, dim=-1)
    contact_displacement_p95 = _p95(displacement_mm[selected])
    pair = selected[:, 1:] & selected[:, :-1]
    base_slide = 1000.0 * float(fps) * torch.linalg.vector_norm(
        base[:, 1:, :2] - base[:, :-1, :2], dim=-1
    )
    cand_slide = 1000.0 * float(fps) * torch.linalg.vector_norm(
        candidate[:, 1:, :2] - candidate[:, :-1, :2], dim=-1
    )
    base_slide_p95 = _p95(base_slide[pair])
    cand_slide_p95 = _p95(cand_slide[pair])
    slide_increment_p95 = (
        None if base_slide_p95 is None or cand_slide_p95 is None
        else cand_slide_p95 - base_slide_p95
    )
    slide_limit = None if base_slide_p95 is None else max(0.05 * base_slide_p95, 5.0)

    ground = float(reference["ground_height_m"])
    base_penetration = 1000.0 * (ground - base[..., 2]).clamp_min(0.0)
    cand_penetration = 1000.0 * (ground - candidate[..., 2]).clamp_min(0.0)
    baseline_penetration_p95 = _p95(base_penetration[:, valid])
    candidate_penetration_p95 = _p95(cand_penetration[:, valid])
    penetration_increment_p95 = (
        candidate_penetration_p95 - baseline_penetration_p95
        if candidate_penetration_p95 is not None and baseline_penetration_p95 is not None
        else None
    )
    support = torch.stack((selected[0] | selected[1], selected[2] | selected[3]))
    directions = []
    for side_index in range(2):
        heel, toe = 2 * side_index, 2 * side_index + 1
        directions.append(_unit_angle_deg(
            base[toe] - base[heel], candidate[toe] - candidate[heel]
        ))
    foot_direction = torch.stack(directions)
    foot_direction_p95 = _p95(foot_direction[support])
    # Flat-foot and toe-only windows are reported separately, but support is
    # never inferred from the candidate's height or speed.
    flat = reference["flat_contact_mask"].bool()
    flat_frames = int(((flat[0] & flat[1]) | (flat[2] & flat[3])).sum())
    toe_only_frames = int(((selected[1] & ~selected[0]) | (selected[3] & ~selected[2])).sum())
    observed = {
        "contact_displacement": int(selected.sum()),
        "continuous_contact_pairs": int(pair.sum()),
        "support_foot_direction_frames": int(support.sum()),
    }
    gates = {
        "contact_displacement": (
            None if contact_displacement_p95 is None else contact_displacement_p95 <= 10.0
        ),
        "sliding_increment": (
            None if slide_increment_p95 is None or slide_limit is None
            else slide_increment_p95 <= slide_limit
        ),
        "penetration": bool(
            candidate_penetration_p95 is not None and penetration_increment_p95 is not None
            and candidate_penetration_p95 <= 5.0 and penetration_increment_p95 <= 1.0
        ),
        "support_foot_direction": (
            None if foot_direction_p95 is None else foot_direction_p95 <= 3.0
        ),
    }
    return {
        "evaluator_version": VERSION,
        "status": "EVALUATED" if all(value is not None for value in gates.values()) else "NOT_EVALUABLE",
        "full_contact_pass": bool(all(value is True for value in gates.values())),
        "gates": gates,
        "observations": observed,
        "flat_contact_frames": flat_frames,
        "toe_only_contact_frames": toe_only_frames,
        "ground_height_m": ground,
        "contact_displacement_p95_mm": contact_displacement_p95,
        "baseline_slide_p95_mm_per_second": base_slide_p95,
        "candidate_slide_p95_mm_per_second": cand_slide_p95,
        "slide_increment_p95_mm_per_second": slide_increment_p95,
        "slide_increment_limit_mm_per_second": slide_limit,
        "baseline_penetration_p95_mm": baseline_penetration_p95,
        "candidate_penetration_p95_mm": candidate_penetration_p95,
        "penetration_increment_p95_mm": penetration_increment_p95,
        "penetration_max_mm": float(cand_penetration[:, valid].max()),
        "penetration_above_5mm_frame_rate": float(
            (cand_penetration[:, valid] > 5.0).any(dim=0).float().mean()
        ),
        "support_foot_direction_p95_deg": foot_direction_p95,
        "per_frame": {
            "contact_displacement_mm": displacement_mm.tolist(),
            "candidate_penetration_mm": cand_penetration.tolist(),
            "candidate_slide_mm_per_second": cand_slide.tolist(),
            "frozen_contact_mask": selected.tolist(),
        },
    }


__all__ = ["VERSION", "frozen_contact_reference", "evaluate_frozen_contact"]
