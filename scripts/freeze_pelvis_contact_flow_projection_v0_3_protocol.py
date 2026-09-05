#!/usr/bin/env python3
"""Freeze a current-environment paired M0 and its contact evidence.

This entry point deliberately does not modify the historical v3.0.1/v0.2
protocols.  The supplied ``official_pre_cast`` replay is authoritative for the
new protocol only after the same authority projection used by ViMoGen has been
applied.  An existing output directory is never reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from smplx import SMPLX

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pelvis_contact_compensation_v3 import (  # noqa: E402
    GEOMETRY_PROTOCOL,
    contact_evidence,
    foot_patches,
    patch_hash,
    patch_centres,
    select_stable_window,
)
from evaluation.relative_root_trunk_v2_1 import direct_smpl_parameters  # noqa: E402
from motion_rep.pose_authority import authority_project  # noqa: E402
from motion_rep.smplx_utils import default_smpl_model_path  # noqa: E402
from sampling.pelvis_contact_flow_projection_v0_1 import (  # noqa: E402
    CURRENT_ENV_PAIRED_PROTOCOL,
    write_strict_json,
)


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


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def runtime_fingerprint() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "sdp_flash": bool(torch.backends.cuda.flash_sdp_enabled()),
        "sdp_mem_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "sdp_math": bool(torch.backends.cuda.math_sdp_enabled()),
    }
    if torch.cuda.is_available():
        result["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory": int(torch.cuda.get_device_properties(index).total_memory),
            }
            for index in range(torch.cuda.device_count())
        ]
    result["nvidia_smi"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return result


def load_valid_mask(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        for key in ("valid_mask", "motion_mask"):
            if key in value:
                value = value[key]
                break
    mask = torch.as_tensor(value).bool()
    if mask.ndim != 2:
        raise ValueError("valid mask must have shape [B,T]")
    return mask


def vertices(motion: torch.Tensor, model: SMPLX, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        params = direct_smpl_parameters(motion.unsqueeze(0).to(device))
        params = {key: value[0] for key, value in params.items()}
        return model(**params, return_verts=True).vertices.detach().float().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0-norm", type=Path, required=True, help="official_pre_cast normalized [B,T,276]")
    parser.add_argument("--valid-mask", type=Path, required=True)
    parser.add_argument("--mean", type=Path, required=True)
    parser.add_argument("--std", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--runtime-fingerprint", type=Path, default=None)
    parser.add_argument("--source-run", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-ids", nargs=2, default=["94", "34122"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"frozen protocol already exists: {args.output}")
    m0_norm = torch.load(args.m0_norm, map_location="cpu", weights_only=True).float()
    valid_mask = load_valid_mask(args.valid_mask)
    mean = torch.from_numpy(np.load(args.mean)).float()
    std = torch.from_numpy(np.load(args.std)).float()
    if m0_norm.ndim != 3 or m0_norm.shape[-1] != 276:
        raise ValueError("m0-norm must be [B,T,276]")
    if valid_mask.shape != m0_norm.shape[:2] or mean.shape[-1] != 276 or std.shape[-1] != 276:
        raise ValueError("M0, mask, mean and std dimensions do not match")
    manifest_rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest_rows, list) or len(manifest_rows) != m0_norm.shape[0]:
        raise ValueError("manifest row count must match M0 batch")
    listed_ids = [str(row.get("sample_id", row.get("global_id"))) for row in manifest_rows]
    if listed_ids != [str(value) for value in args.sample_ids]:
        raise ValueError(f"manifest IDs {listed_ids} do not match requested sample IDs {args.sample_ids}")
    m0_physical = authority_project(
        m0_norm * std.view(1, 1, -1) + mean.view(1, 1, -1),
        valid_mask=valid_mask,
        output_dtype=torch.float32,
    ).physical_motion
    model_path = args.model_path or default_smpl_model_path("smplx", ROOT)
    device = torch.device(args.device)
    model = SMPLX(
        model_path=str(model_path),
        gender="neutral",
        num_betas=10,
        batch_size=int(m0_physical.shape[1]),
        use_pca=False,
    ).to(device)
    patches = foot_patches(model)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "foot_patches.json").write_text(
        json.dumps(patches, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch.save(m0_physical.cpu(), output / "m0_physical.pt")
    torch.save(valid_mask.cpu(), output / "valid_mask.pt")
    cases: list[dict[str, Any]] = []
    for row_index, sample_id in enumerate(args.sample_ids):
        verts = vertices(m0_physical[row_index], model, device)
        sides: dict[str, Any] = {}
        for side in ("left", "right"):
            heel, toe = patch_centres(verts, patches[side])
            evidence = contact_evidence(
                heel, toe, valid_mask=valid_mask[row_index].cpu()
            )
            window = select_stable_window(
                torch.tensor(evidence["confidence"], dtype=torch.float32),
                valid_mask=valid_mask[row_index].cpu(),
                stable_mask=torch.as_tensor(
                    evidence["valid_masks"]["flat_contact"], dtype=torch.bool
                ),
                pad=4,
            )
            sides[side] = {"evidence": evidence, "stable_window": window}
        if any(sides[side]["stable_window"].get("status") != "PASS" for side in ("left", "right")):
            raise RuntimeError(f"sample {sample_id} does not have evaluable left/right stable windows")
        cases.append(
            {
                "sample_id": str(sample_id),
                "source_index": row_index,
                "row_index": row_index,
                "seed": 0,
                "target_delta_deg": 10.0,
                "sides": sides,
            }
        )
    fingerprint = runtime_fingerprint()
    if args.runtime_fingerprint is not None:
        supplied = json.loads(args.runtime_fingerprint.read_text(encoding="utf-8"))
        fingerprint = {"captured": supplied, "freeze_process": fingerprint}
    inputs: dict[str, Any] = {
        "m0_official_pre_cast": {"path": str(args.m0_norm), "sha256": sha256_path(args.m0_norm)},
        "valid_mask": {"path": str(args.valid_mask), "sha256": sha256_path(args.valid_mask)},
        "manifest": {"path": str(args.manifest), "sha256": sha256_path(args.manifest)},
        "mean": {"path": str(args.mean), "sha256": sha256_path(args.mean)},
        "std": {"path": str(args.std), "sha256": sha256_path(args.std)},
        "smplx_model": {"path": str(model_path), "sha256": sha256_path(Path(model_path)) if Path(model_path).exists() else None},
    }
    if args.source_run is not None:
        inputs["source_run"] = {"path": str(args.source_run), "sha256": sha256_path(args.source_run)}
    protocol = {
        "protocol": CURRENT_ENV_PAIRED_PROTOCOL,
        "geometry_protocol": GEOMETRY_PROTOCOL,
        "protocol_revision": "v0.3",
        "baseline_origin": "current_environment_refreeze",
        "legacy_v3_relation": "reference_only",
        "m0_reference_protocol": CURRENT_ENV_PAIRED_PROTOCOL,
        "m0_boundary": "official_pre_cast_to_authority_project_to_frozen_physical_m0",
        "dose_sign": "positive dose follows the frozen v1.3 pelvis pitch convention",
        "contact": {
            "contact_height_m": 0.025,
            "contact_speed_m_per_frame": 0.030,
            "flat_gap_m": 0.020,
            "stable_confidence": 0.80,
            "minimum_evidence_frames_or_pairs": 3,
            "first_frame_speed_is_valid": False,
            "evidence_source": "current_environment_refreeze_m0",
        },
        "cases": cases,
        "foot_patches": {
            "source": "neutral_SMPL-X_template_bottom_quartile_and_skinning_weights",
            "sha256": patch_hash(patches),
            "file": "foot_patches.json",
        },
        "inputs": inputs,
        "runtime_fingerprint": fingerprint,
        "git": {
            "revision": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "dirty_state": git_value("status", "--porcelain"),
        },
        "artifacts": {
            "m0_physical": "m0_physical.pt",
            "valid_mask": "valid_mask.pt",
            "foot_patches": "foot_patches.json",
        },
    }
    write_strict_json(output / "protocol.json", protocol)
    write_strict_json(output / "environment_fingerprint.json", fingerprint)
    # Mark the protocol files read-only after successful creation.  Future
    # runs fail on an existing directory, so this is an immutable audit input.
    for path in output.iterdir():
        path.chmod(0o444)
    output.chmod(0o555)
    print(json.dumps({"protocol": CURRENT_ENV_PAIRED_PROTOCOL, "output": str(output), "patch_sha256": patch_hash(patches), "cases": list(args.sample_ids)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
