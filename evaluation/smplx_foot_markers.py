"""Frozen SMPL-X sole-patch markers for candidate physical evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from evaluation.relative_root_trunk_v2_1 import direct_smpl_parameters
from motion_rep.phase1 import validate_motion_tensor


MARKER_VERSION = "smplx_neutral_sole_patch_centres_v1"
MARKER_NAMES = ("left_heel", "left_toe", "right_heel", "right_toe")


def sha256_path(path: Path) -> str:
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


def foot_patches(model: Any) -> dict[str, dict[str, list[int]]]:
    template = model.v_template.detach().cpu()
    joints = model.J_regressor.detach().cpu() @ template
    dominant_joint = model.lbs_weights.detach().cpu().argmax(-1)
    patches: dict[str, dict[str, list[int]]] = {}
    for side, ankle_index, foot_index in (("left", 7, 10), ("right", 8, 11)):
        foot_vertices = torch.nonzero(
            (dominant_joint == ankle_index) | (dominant_joint == foot_index),
            as_tuple=False,
        ).flatten()
        candidate = template[foot_vertices]
        sole = foot_vertices[candidate[:, 1] <= torch.quantile(candidate[:, 1], 0.25)]
        forward = joints[foot_index] - joints[ankle_index]
        forward[1] = 0.0
        forward = forward / torch.linalg.vector_norm(forward).clamp_min(1.0e-8)
        longitudinal = ((template[sole] - joints[ankle_index]) * forward).sum(-1)
        heel = sole[longitudinal <= torch.quantile(longitudinal, 0.25)]
        toe = sole[longitudinal >= torch.quantile(longitudinal, 0.75)]
        if heel.numel() < 5 or toe.numel() < 5:
            raise RuntimeError(f"insufficient {side} heel/toe vertices")
        patches[side] = {
            "heel": [int(value) for value in heel.tolist()],
            "toe": [int(value) for value in toe.tolist()],
            "sole": [int(value) for value in sole.tolist()],
        }
    return patches


def patch_sha256(patches: Mapping[str, Mapping[str, list[int]]]) -> str:
    payload = json.dumps(
        patches, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SMPLXFootMarkerExtractor:
    def __init__(
        self,
        model_path: Path,
        *,
        frame_count: int,
        device: str = "cuda:0",
        patches: Mapping[str, Mapping[str, list[int]]] | None = None,
    ) -> None:
        if frame_count < 2:
            raise ValueError("frame_count must be at least two")
        from smplx import SMPLX

        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        self.device = torch.device(device)
        self.model = SMPLX(
            model_path=str(self.model_path),
            gender="neutral",
            num_betas=10,
            batch_size=int(frame_count),
            use_pca=False,
        ).to(self.device).eval()
        self.frame_count = int(frame_count)
        self.patches = (
            foot_patches(self.model)
            if patches is None
            else {
                side: {name: [int(value) for value in values] for name, values in patch.items()}
                for side, patch in patches.items()
            }
        )
        if set(self.patches) != {"left", "right"}:
            raise ValueError("patches must contain left and right")
        for side in ("left", "right"):
            if set(self.patches[side]) != {"heel", "toe", "sole"}:
                raise ValueError(f"{side} patch must contain heel, toe and sole")

    @property
    def patch_hash(self) -> str:
        return patch_sha256(self.patches)

    @torch.inference_mode()
    def __call__(self, motion_physical: torch.Tensor) -> torch.Tensor:
        validate_motion_tensor(motion_physical)
        if motion_physical.ndim == 2:
            motion_physical = motion_physical.unsqueeze(0)
        if motion_physical.ndim != 3 or motion_physical.shape[0] != 1:
            raise ValueError("extractor accepts one motion with shape [T,276] or [1,T,276]")
        if motion_physical.shape[1] != self.frame_count:
            raise ValueError(
                f"motion frame count {motion_physical.shape[1]} != {self.frame_count}"
            )
        parameters = direct_smpl_parameters(motion_physical.to(self.device))
        parameters = {name: value[0] for name, value in parameters.items()}
        vertices = self.model(**parameters, return_verts=True).vertices.detach().float()
        markers = []
        for side in ("left", "right"):
            for name in ("heel", "toe"):
                indices = torch.as_tensor(
                    self.patches[side][name], dtype=torch.long, device=self.device
                )
                markers.append(vertices[:, indices].mean(dim=1))
        return torch.stack(markers, dim=0).cpu()

    def definition(self) -> dict[str, Any]:
        return {
            "version": MARKER_VERSION,
            "kind": "smplx_mesh_patch_centres",
            "marker_names": list(MARKER_NAMES),
            "patch_source": "neutral template foot skinning weights and sole quartiles",
            "patch_sha256": self.patch_hash,
            "patches": self.patches,
            "smplx_model_path": str(self.model_path),
            "smplx_model_sha256": sha256_path(self.model_path),
        }


__all__ = [
    "MARKER_NAMES",
    "MARKER_VERSION",
    "SMPLXFootMarkerExtractor",
    "foot_patches",
    "patch_sha256",
    "sha256_path",
]
