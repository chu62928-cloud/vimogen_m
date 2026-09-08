"""Frozen paired-M0 physical reference for the M1--M7 comparison."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from geometry.contacts import MARKER_CONTACT_VERSION, freeze_marker_contact_evidence
from geometry.ground import GROUND_VERSION, estimate_ground_height
from guidance.base import tensor_sha256
from motion_rep.consistency_v2 import Skeleton22, default_smplx_neutral_22_skeleton
from motion_rep.phase1 import MOTION_LAYOUT, SMPLX_22_JOINT_INDEX
from motion_rep.pose_authority import PROTOCOL_NAME as AUTHORITY_VERSION, authority_project


REFERENCE_CACHE_VERSION = "m1_m7_physical_reference_v2"
FK_VERSION = "smplx_neutral_22_fk_v1"
MARKER_NAMES = ("left_heel", "left_toe", "right_heel", "right_toe")
MARKER_JOINTS = {
    "left_heel": "left_ankle",
    "left_toe": "left_foot",
    "right_heel": "right_ankle",
    "right_toe": "right_foot",
}


def marker_positions_from_motion(
    motion_physical: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    skeleton: Skeleton22 | Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor, torch.Tensor]:
    """Rebuild one motion at the authority boundary and extract joint markers.

    The 22-joint representation has no surface vertices.  The frozen marker
    definition therefore uses the ankle joints as heel proxies and the foot
    joints as toe proxies.  Every candidate and paired M0 uses exactly these
    same joint identifiers and the same FK skeleton.
    """

    projection = authority_project(
        motion_physical,
        valid_mask=valid_mask,
        skeleton=skeleton,
        output_standardized=False,
        output_dtype=torch.float32,
    )
    canonical = projection.physical_motion
    unbatched = canonical.ndim == 2
    if unbatched:
        canonical = canonical.unsqueeze(0)
    joints = canonical[..., MOTION_LAYOUT.joints].reshape(
        canonical.shape[0], canonical.shape[1], 22, 3
    )
    markers = {
        "left": {
            "heel": joints[..., SMPLX_22_JOINT_INDEX[MARKER_JOINTS["left_heel"]], :],
            "toe": joints[..., SMPLX_22_JOINT_INDEX[MARKER_JOINTS["left_toe"]], :],
        },
        "right": {
            "heel": joints[..., SMPLX_22_JOINT_INDEX[MARKER_JOINTS["right_heel"]], :],
            "toe": joints[..., SMPLX_22_JOINT_INDEX[MARKER_JOINTS["right_toe"]], :],
        },
    }
    root_world = joints[..., SMPLX_22_JOINT_INDEX["pelvis"], :]
    if unbatched:
        markers = {
            side: {name: value[0] for name, value in side_markers.items()}
            for side, side_markers in markers.items()
        }
        root_world = root_world[0]
    return markers, root_world, projection.motion


def _stack_markers(
    markers: Mapping[str, Mapping[str, torch.Tensor]],
) -> torch.Tensor:
    return torch.stack(
        (
            markers["left"]["heel"],
            markers["left"]["toe"],
            markers["right"]["heel"],
            markers["right"]["toe"],
        ),
        dim=1,
    )


def _stack_marker_masks(
    masks: Mapping[str, Mapping[str, torch.Tensor]],
) -> torch.Tensor:
    return torch.stack(
        (
            masks["left"]["heel"],
            masks["left"]["toe"],
            masks["right"]["heel"],
            masks["right"]["toe"],
        ),
        dim=1,
    )


def materialize_reference(
    m0_motion_physical: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    sample_ids: Sequence[str],
    seeds: Sequence[int],
    skeleton: Skeleton22 | Mapping[str, Any] | None = None,
    code_commit: str = "not_recorded",
) -> dict[str, Any]:
    """Create a self-contained shared reference cache from paired M0 only."""

    if m0_motion_physical.ndim != 3:
        raise ValueError("m0_motion_physical must have shape [B,T,276]")
    batch = m0_motion_physical.shape[0]
    if valid_mask.dtype is not torch.bool or tuple(valid_mask.shape) != tuple(m0_motion_physical.shape[:2]):
        raise ValueError("valid_mask must be bool[B,T]")
    if len(sample_ids) != batch or len(seeds) != batch:
        raise ValueError("sample_ids and seeds must match the M0 batch")
    sample_ids = [str(value) for value in sample_ids]
    seeds = [int(value) for value in seeds]
    if len(set(zip(seeds, sample_ids))) != batch:
        raise ValueError("each seed/sample pair must be unique")

    tree = default_smplx_neutral_22_skeleton() if skeleton is None else skeleton
    markers, root_world, canonical = marker_positions_from_motion(
        m0_motion_physical, valid_mask, skeleton=tree
    )
    marker_world = _stack_markers(markers)
    # Ground estimator expects [B,T,M,3], while the cache is stored [B,M,T,3].
    ground_height = estimate_ground_height(
        marker_world.permute(0, 2, 1, 3), valid_mask, up_axis=2
    )
    contact = freeze_marker_contact_evidence(
        markers, valid_mask, ground_height, up_axis=2
    )
    normal = torch.zeros((batch, 3), dtype=ground_height.dtype, device=ground_height.device)
    normal[:, 2] = 1.0
    ground_plane = torch.cat((normal, -ground_height[:, None]), dim=-1)
    paired_ids = [
        f"seed{seed}_sample{sample_id}_{tensor_sha256(canonical[index:index + 1])[:16]}"
        for index, (seed, sample_id) in enumerate(zip(seeds, sample_ids))
    ]
    skeleton_source = getattr(tree, "source", "injected")
    return {
        "cache_version": REFERENCE_CACHE_VERSION,
        "authority_version": AUTHORITY_VERSION,
        "fk_version": FK_VERSION,
        "skeleton_source": str(skeleton_source),
        "contact_version": MARKER_CONTACT_VERSION,
        "ground_version": GROUND_VERSION,
        "code_commit": str(code_commit),
        "unit": "m",
        "world_up_axis": 2,
        "marker_names": list(MARKER_NAMES),
        "marker_joints": dict(MARKER_JOINTS),
        "sample_ids": sample_ids,
        "seeds": seeds,
        "paired_m0_ids": paired_ids,
        "valid_frame_mask": valid_mask.detach().cpu(),
        "contact_mask": _stack_marker_masks(contact.contact).detach().cpu(),
        "flat_contact_mask": _stack_marker_masks(contact.flat_contact).detach().cpu(),
        "continuous_contact_pair": _stack_marker_masks(contact.continuous_pair).detach().cpu(),
        "contact_confidence": _stack_marker_masks(contact.confidence).detach().cpu(),
        "ground_height_m": ground_height.detach().cpu(),
        "ground_plane": ground_plane.detach().cpu(),
        "m0_marker_world": marker_world.detach().cpu(),
        "m0_root_world": root_world.detach().cpu(),
        "m0_motion_physical": canonical.detach().cpu(),
        "contact_definition": {
            "height_threshold_m": 0.025,
            "speed_threshold_m_per_frame": 0.030,
            "flat_height_difference_m": 0.020,
            "first_frame_excluded": True,
            "mask_granularity": "PER_MARKER",
            "candidate_reclassification_forbidden": True,
        },
    }


def reference_markers(
    reference: Mapping[str, Any], index: int
) -> dict[str, dict[str, torch.Tensor]]:
    stacked = torch.as_tensor(reference["m0_marker_world"])[index : index + 1]
    return {
        "left": {"heel": stacked[:, 0], "toe": stacked[:, 1]},
        "right": {"heel": stacked[:, 2], "toe": stacked[:, 3]},
    }


def reference_contact_masks(
    reference: Mapping[str, Any], index: int
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[str, dict[str, torch.Tensor]],
]:
    contacts = torch.as_tensor(reference["contact_mask"]).bool()[index : index + 1]
    pairs = torch.as_tensor(reference["continuous_contact_pair"]).bool()[index : index + 1]
    return (
        {
            "left": {"heel": contacts[:, 0], "toe": contacts[:, 1]},
            "right": {"heel": contacts[:, 2], "toe": contacts[:, 3]},
        },
        {
            "left": {"heel": pairs[:, 0], "toe": pairs[:, 1]},
            "right": {"heel": pairs[:, 2], "toe": pairs[:, 3]},
        },
    )


__all__ = [
    "FK_VERSION",
    "MARKER_JOINTS",
    "MARKER_NAMES",
    "REFERENCE_CACHE_VERSION",
    "marker_positions_from_motion",
    "materialize_reference",
    "reference_contact_masks",
    "reference_markers",
]
