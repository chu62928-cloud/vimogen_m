"""M0-only foot-contact evidence used by all M1--M7 methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


CONTACT_VERSION = "m1_m7_frozen_heel_toe_contact_v1"
MARKER_CONTACT_VERSION = "m1_m7_frozen_marker_contact_v2"


@dataclass(frozen=True)
class FrozenContactEvidence:
    contact: Mapping[str, torch.Tensor]
    flat_contact: Mapping[str, torch.Tensor]
    continuous_pair: Mapping[str, torch.Tensor]
    confidence: Mapping[str, torch.Tensor]
    ground_height_m: torch.Tensor
    version: str = CONTACT_VERSION


@dataclass(frozen=True)
class FrozenMarkerContactEvidence:
    contact: Mapping[str, Mapping[str, torch.Tensor]]
    flat_contact: Mapping[str, Mapping[str, torch.Tensor]]
    continuous_pair: Mapping[str, Mapping[str, torch.Tensor]]
    confidence: Mapping[str, Mapping[str, torch.Tensor]]
    ground_height_m: torch.Tensor
    version: str = MARKER_CONTACT_VERSION


def _speed(position: torch.Tensor) -> torch.Tensor:
    speed = torch.zeros(position.shape[:2], dtype=position.dtype, device=position.device)
    speed[:, 1:] = torch.linalg.vector_norm(position[:, 1:] - position[:, :-1], dim=-1)
    return speed


def freeze_contact_evidence(
    foot_markers: Mapping[str, Mapping[str, torch.Tensor]],
    valid_mask: torch.Tensor,
    ground_height_m: torch.Tensor,
    *,
    up_axis: int = 2,
    height_threshold_m: float = 0.025,
    speed_threshold_m_per_frame: float = 0.030,
    flat_height_difference_m: float = 0.020,
) -> FrozenContactEvidence:
    """Freeze hard masks and a continuous confidence from paired M0 only."""

    if valid_mask.dtype is not torch.bool or valid_mask.ndim != 2:
        raise ValueError("valid_mask must be bool[B,T]")
    if ground_height_m.shape != (valid_mask.shape[0],):
        raise ValueError("ground_height_m must have shape [B]")
    if set(foot_markers) != {"left", "right"}:
        raise ValueError("foot_markers must contain left and right")
    contact: dict[str, torch.Tensor] = {}
    flat: dict[str, torch.Tensor] = {}
    pairs: dict[str, torch.Tensor] = {}
    confidence: dict[str, torch.Tensor] = {}
    floor = ground_height_m[:, None]
    for side in ("left", "right"):
        markers = foot_markers[side]
        if set(markers) != {"heel", "toe"}:
            raise ValueError(f"{side} markers must contain heel and toe")
        heel, toe = markers["heel"], markers["toe"]
        expected = (*valid_mask.shape, 3)
        if tuple(heel.shape) != expected or tuple(toe.shape) != expected:
            raise ValueError(f"{side} heel/toe must have shape {expected}")
        heel_height = heel[..., up_axis] - floor
        toe_height = toe[..., up_axis] - floor
        heel_speed, toe_speed = _speed(heel), _speed(toe)
        low = torch.minimum(heel_height, toe_height)
        slow = torch.maximum(heel_speed, toe_speed)
        mask = (
            valid_mask
            & (low <= float(height_threshold_m))
            & (slow <= float(speed_threshold_m_per_frame))
        )
        # The first-frame speed has no predecessor and is never evidence.
        mask[:, 0] = False
        flat_mask = mask & (
            (heel_height - toe_height).abs() <= float(flat_height_difference_m)
        )
        pair = mask[:, 1:] & mask[:, :-1]
        height_score = (1.0 - low.clamp_min(0.0) / float(height_threshold_m)).clamp(0.0, 1.0)
        speed_score = (1.0 - slow / float(speed_threshold_m_per_frame)).clamp(0.0, 1.0)
        score = torch.minimum(height_score, speed_score) * valid_mask.to(heel.dtype)
        score[:, 0] = 0.0
        contact[side] = mask
        flat[side] = flat_mask
        pairs[side] = pair
        confidence[side] = score
    return FrozenContactEvidence(contact, flat, pairs, confidence, ground_height_m)


def freeze_marker_contact_evidence(
    foot_markers: Mapping[str, Mapping[str, torch.Tensor]],
    valid_mask: torch.Tensor,
    ground_height_m: torch.Tensor,
    *,
    up_axis: int = 2,
    height_threshold_m: float = 0.025,
    speed_threshold_m_per_frame: float = 0.030,
    flat_height_difference_m: float = 0.020,
) -> FrozenMarkerContactEvidence:
    """Freeze heel and toe contact independently from paired M0.

    The v1 side-level API above remains unchanged for historical protocols.
    This v2 evidence prevents a low heel from promoting an airborne toe (or
    vice versa) into the candidate evaluation mask.
    """

    if valid_mask.dtype is not torch.bool or valid_mask.ndim != 2:
        raise ValueError("valid_mask must be bool[B,T]")
    if ground_height_m.shape != (valid_mask.shape[0],):
        raise ValueError("ground_height_m must have shape [B]")
    if set(foot_markers) != {"left", "right"}:
        raise ValueError("foot_markers must contain left and right")
    contact: dict[str, dict[str, torch.Tensor]] = {}
    flat: dict[str, dict[str, torch.Tensor]] = {}
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    confidence: dict[str, dict[str, torch.Tensor]] = {}
    floor = ground_height_m[:, None]
    for side in ("left", "right"):
        markers = foot_markers[side]
        if set(markers) != {"heel", "toe"}:
            raise ValueError(f"{side} markers must contain heel and toe")
        expected = (*valid_mask.shape, 3)
        if any(tuple(markers[name].shape) != expected for name in ("heel", "toe")):
            raise ValueError(f"{side} heel/toe must have shape {expected}")
        height_difference = (
            markers["heel"][..., up_axis] - markers["toe"][..., up_axis]
        ).abs()
        contact[side] = {}
        flat[side] = {}
        pairs[side] = {}
        confidence[side] = {}
        for marker in ("heel", "toe"):
            position = markers[marker]
            height = position[..., up_axis] - floor
            speed = _speed(position)
            mask = (
                valid_mask
                & (height <= float(height_threshold_m))
                & (speed <= float(speed_threshold_m_per_frame))
            )
            mask = mask.clone()
            mask[:, 0] = False
            score = torch.minimum(
                (1.0 - height.clamp_min(0.0) / float(height_threshold_m)).clamp(0.0, 1.0),
                (1.0 - speed / float(speed_threshold_m_per_frame)).clamp(0.0, 1.0),
            ) * valid_mask.to(position.dtype)
            score = score.clone()
            score[:, 0] = 0.0
            contact[side][marker] = mask
            flat[side][marker] = mask & (
                height_difference <= float(flat_height_difference_m)
            )
            pairs[side][marker] = mask[:, 1:] & mask[:, :-1]
            confidence[side][marker] = score
    return FrozenMarkerContactEvidence(
        contact, flat, pairs, confidence, ground_height_m
    )
