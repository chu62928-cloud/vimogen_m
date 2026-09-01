#!/usr/bin/env python3
"""Freeze the v3.0 geometry, M0 contact evidence and case manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from smplx import SMPLX

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pelvis_contact_compensation_v3 import (
    GEOMETRY_PROTOCOL,
    PROTOCOL_NAME,
    contact_evidence,
    foot_patches,
    patch_centres,
    patch_hash,
    select_stable_window,
)
from evaluation.relative_root_trunk_v2_1 import direct_smpl_parameters
from motion_rep.smplx_utils import default_smpl_model_path
from motion_rep.pose_authority import authority_project


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with child.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    else:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")


def _vertices(motion: torch.Tensor, model: SMPLX, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        params = direct_smpl_parameters(motion.unsqueeze(0).to(device))
        params = {key: value[0] for key, value in params.items()}
        return model(**params, return_verts=True).vertices.detach().float().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0-norm", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True, help="matching v1.3 archive containing motion_mask")
    parser.add_argument("--mean", type=Path, required=True)
    parser.add_argument("--std", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-ids", nargs=2, default=["94", "34122"])
    parser.add_argument("--sample-indices", nargs=2, type=int, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"frozen protocol already exists: {args.output}")
    m0_norm = torch.load(args.m0_norm, map_location="cpu", weights_only=True).float()
    archive = torch.load(args.archive, map_location="cpu", weights_only=True)
    valid_mask = archive["motion_mask"].bool()
    mean = torch.from_numpy(np.load(args.mean)).float()
    std = torch.from_numpy(np.load(args.std)).float()
    if m0_norm.ndim != 3 or m0_norm.shape[-1] != 276:
        raise ValueError("m0-norm must be [B,T,276]")
    if valid_mask.shape != m0_norm.shape[:2] or mean.shape[-1] != 276 or std.shape[-1] != 276:
        raise ValueError("M0, mask, mean and std dimensions do not match")
    m0_physical = m0_norm * std.view(1, 1, -1) + mean.view(1, 1, -1)
    m0_physical = authority_project(m0_physical, valid_mask=valid_mask, output_dtype=torch.float32).physical_motion
    archive_ids = archive.get("sample_ids")
    if args.sample_indices is None:
        if not isinstance(archive_ids, (list, tuple)):
            raise ValueError("archive must contain sample_ids when --sample-indices is omitted")
        lookup = {str(sample_id): index for index, sample_id in enumerate(archive_ids)}
        try:
            indices = [lookup[str(sample_id)] for sample_id in args.sample_ids]
        except KeyError as exc:
            raise ValueError(f"requested sample id is absent from archive: {exc.args[0]}") from exc
    else:
        indices = [int(x) for x in args.sample_indices]
    if len(set(indices)) != 2 or any(index < 0 or index >= m0_physical.shape[0] for index in indices):
        raise ValueError("sample-indices must identify two distinct M0 rows")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    model_path = args.model_path or default_smpl_model_path("smplx", ROOT)
    device = torch.device(args.device)
    model = SMPLX(model_path=str(model_path), gender="neutral", num_betas=10, batch_size=int(m0_physical.shape[1]), use_pca=False).to(device)
    patches = foot_patches(model)
    (output / "foot_patches.json").write_text(json.dumps(patches, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected_m0 = m0_physical[indices].cpu()
    selected_mask = valid_mask[indices].cpu()
    torch.save(selected_m0, output / "m0_physical.pt")
    torch.save(selected_mask, output / "valid_mask.pt")
    cases: list[dict[str, Any]] = []
    for row_index, (sample_id, source_index) in enumerate(zip(args.sample_ids, indices)):
        vertices = _vertices(m0_physical[source_index], model, device)
        sides: dict[str, Any] = {}
        for side in ("left", "right"):
            heel, toe = patch_centres(vertices, patches[side])
            evidence = contact_evidence(heel, toe, valid_mask=valid_mask[source_index].cpu())
            confidence = torch.tensor(evidence["confidence"], dtype=torch.float32)
            window = select_stable_window(
                confidence,
                valid_mask=valid_mask[source_index].cpu(),
                stable_mask=torch.as_tensor(evidence["valid_masks"]["flat_contact"], dtype=torch.bool),
                pad=4,
            )
            sides[side] = {"evidence": evidence, "stable_window": window}
        cases.append({"sample_id": str(sample_id), "source_index": int(source_index), "row_index": row_index, "seed": 0, "target_delta_deg": 10.0, "sides": sides})
    protocol = {
        "protocol": PROTOCOL_NAME,
        "geometry_protocol": GEOMETRY_PROTOCOL,
        "dose_sign": "v1.3: positive dose is M0_pitch minus candidate_pitch; +10 lowers M0 forward axis by 10 degrees",
        "target_root": "Rot(M0_right, -target_delta_deg) @ M0_root_rotation",
        "contact": {
            "contact_height_m": 0.025,
            "contact_speed_m_per_frame": 0.030,
            "flat_gap_m": 0.020,
            "stable_confidence": 0.80,
            "minimum_evidence_frames_or_pairs": 3,
            "first_frame_speed_is_valid": False,
        },
        "trust_region": {"max_rotation_deg": 30.0, "max_translation_m": 0.05, "diagnostic_expanded_bounds": {"max_rotation_deg": 45.0, "max_translation_m": 0.10, "counts_as_success": False}},
        "cases": cases,
        "foot_patches": {"source": "neutral_SMPL-X_template_bottom_quartile_and_skinning_weights", "sha256": patch_hash(patches), "file": "foot_patches.json"},
        "inputs": {
            "m0_norm": {"path": str(args.m0_norm), "sha256": _sha256(args.m0_norm)},
            "archive": {"path": str(args.archive), "sha256": _sha256(args.archive)},
            "mean": {"path": str(args.mean), "sha256": _sha256(args.mean)},
            "std": {"path": str(args.std), "sha256": _sha256(args.std)},
            "smplx_model": {"path": str(model_path), "sha256": _sha256(Path(model_path)) if Path(model_path).exists() else None},
        },
        "artifacts": {"m0_physical": "m0_physical.pt", "valid_mask": "valid_mask.pt"},
    }
    _write_json(output / "protocol.json", protocol)
    print(json.dumps({"protocol": PROTOCOL_NAME, "output": str(output), "patch_sha256": patch_hash(patches), "cases": [row["sample_id"] for row in cases]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
