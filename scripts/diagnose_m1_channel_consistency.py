#!/usr/bin/env python3
"""Offline audit of redundant M1 motion channels.

The packed representation stores both pose samples and forward differences.
This script checks whether the stored velocity channels reproduce the pose
channels that are present in the same tensor.  It never runs the model and
does not rewrite any M1 output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe  # noqa: E402


def _collect(root: Path, filename: str) -> torch.Tensor:
    paths = sorted(root.glob(f"m1_artifacts/batch_*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"no {filename} below {root}")
    return torch.cat([torch.load(path, map_location="cpu", weights_only=True) for path in paths]).float()


def _angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    rel = a @ b.transpose(-1, -2)
    cosine = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def _sample_record(motion: torch.Tensor) -> dict[str, float]:
    # All calculations are in physical units.  Rot6D channels are decoded
    # before comparing rotations; raw 6D vectors are not rotations themselves.
    pos = motion[:, MOTION_LAYOUT.joints].reshape(-1, 22, 3)
    pos_v = motion[:, MOTION_LAYOUT.joints_velocity].reshape(-1, 22, 3)
    pos_recovered = pos[:1] + torch.cumsum(pos_v, dim=0)
    pos_err = (pos[1:] - pos_recovered[:-1]).norm(dim=-1)

    trans = motion[:, MOTION_LAYOUT.root_translation]
    trans_v = motion[:, MOTION_LAYOUT.root_translation_velocity]
    trans_recovered = trans[:1] + torch.cumsum(trans_v, dim=0)
    trans_err = (trans[1:] - trans_recovered[:-1]).norm(dim=-1)

    root = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation])
    root_v = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation_velocity])
    root_recovered = [root[:1]]
    for index in range(motion.shape[0]):
        root_recovered.append(root_v[index:index + 1] @ root_recovered[-1])
    root_recovered = torch.cat(root_recovered, dim=0)
    root_err = _angle_deg(root[1:], root_recovered[1:-1])

    def med_max_mean(value: torch.Tensor) -> tuple[float, float, float]:
        value = value.reshape(-1)
        return float(value.median()), float(value.max()), float(value.mean())

    joint_median, joint_max, joint_mean = med_max_mean(pos_err)
    trans_median, trans_max, trans_mean = med_max_mean(trans_err)
    root_median, root_max, root_mean = med_max_mean(root_err)
    return {
        "joint_position_error_median_m": joint_median,
        "joint_position_error_max_m": joint_max,
        "joint_position_error_mean_m": joint_mean,
        "root_translation_error_median_m": trans_median,
        "root_translation_error_max_m": trans_max,
        "root_translation_error_mean_m": trans_mean,
        "root_rotation_error_median_deg": root_median,
        "root_rotation_error_max_deg": root_max,
        "root_rotation_error_mean_deg": root_mean,
    }


def audit(m1_root: Path, output: Path) -> dict[str, object]:
    mean = torch.from_numpy(np.load(ROOT / "data/meta_info/mean.npy")).float()
    std = torch.from_numpy(np.load(ROOT / "data/meta_info/std.npy")).float()
    records: list[dict[str, object]] = []
    root = Path(m1_root)
    for seed_dir in sorted(root.glob("seed_*")):
        for delta_dir in sorted(seed_dir.glob("delta_*")):
            raw = _collect(delta_dir, "m1_raw_norm_batch.pt") * std + mean
            official = _collect(delta_dir, "m1_official_norm_batch.pt") * std + mean
            if raw.shape != official.shape:
                raise ValueError(f"raw/official shape mismatch: {raw.shape} vs {official.shape}")
            for index in range(official.shape[0]):
                item = _sample_record(official[index])
                item.update({
                    "seed": int(seed_dir.name.split("_")[-1]),
                    "delta": float(delta_dir.name.split("_")[-1].replace("deg", "")),
                    "sample_index": index,
                })
                records.append(item)
    def values(key: str) -> list[float]:
        return [float(item[key]) for item in records]

    summary: dict[str, object] = {"record_count": len(records), "records": records}
    for key in (
        "joint_position_error_median_m",
        "joint_position_error_max_m",
        "root_translation_error_median_m",
        "root_translation_error_max_m",
        "root_rotation_error_median_deg",
        "root_rotation_error_max_deg",
    ):
        vals = values(key)
        summary[f"{key}_across_records_median"] = float(np.median(vals))
        summary[f"{key}_across_records_max"] = float(np.max(vals))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.m1_root, args.output), indent=2))


if __name__ == "__main__":
    main()
