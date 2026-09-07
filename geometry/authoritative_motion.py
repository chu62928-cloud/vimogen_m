"""One authority boundary for packed ViMoGen motions."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from motion_rep.pose_authority import AuthorityProjection, authority_project


def authoritative_motion(
    motion: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    input_standardized: bool = False,
    output_standardized: bool = False,
    skeleton: Mapping[str, Any] | Any | None = None,
) -> AuthorityProjection:
    """Rebuild every derived channel from direct pose/root/translation."""

    return authority_project(
        motion,
        valid_mask=valid_mask,
        mean=mean,
        std=std,
        input_standardized=input_standardized,
        output_standardized=output_standardized,
        output_dtype=torch.float32,
        skeleton=skeleton,
    )
