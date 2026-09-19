#!/usr/bin/env python3
"""Build a diagnostic candidate that changes only the prescribed root rotation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pelvis_contact_compensation_v3 import target_root_rotation
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe, encode_rot6d
from motion_rep.pose_authority import authority_project


def build_root_only_motion(m0: torch.Tensor, target_delta_deg: float) -> torch.Tensor:
    """Apply the frozen M0-frame dose and rebuild all derived channels."""

    if m0.ndim == 2:
        m0 = m0.unsqueeze(0)
    if m0.ndim != 3 or m0.shape[-1] != MOTION_LAYOUT.total_dim or m0.shape[0] != 1:
        raise ValueError("m0 must have shape [T,276] or [1,T,276]")
    direct = m0.float().clone()
    root = decode_rot6d_safe(direct[..., MOTION_LAYOUT.root_rotation])
    direct[..., MOTION_LAYOUT.root_rotation] = encode_rot6d(
        target_root_rotation(root, float(target_delta_deg))
    )
    valid = torch.ones(direct.shape[:2], dtype=torch.bool, device=direct.device)
    return authority_project(direct, valid_mask=valid, output_dtype=torch.float32).physical_motion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--target-delta-deg", type=float, default=10.0)
    args = parser.parse_args()
    m0 = torch.load(args.m0_path, map_location="cpu", weights_only=True)
    candidate = build_root_only_motion(m0, args.target_delta_deg)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(candidate[0], args.output_path)
    print({"output": str(args.output_path), "shape": list(candidate[0].shape), "finite": bool(torch.isfinite(candidate).all())})


if __name__ == "__main__":
    main()
