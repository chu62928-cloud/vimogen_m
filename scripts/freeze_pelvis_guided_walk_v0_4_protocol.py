#!/usr/bin/env python3
"""Freeze the sample94 current-environment M0 for v0.4.

The source v0.3 protocol remains untouched. This creates an immutable
protocol manifest by re-authoritatively projecting the first current-environment
M0 replay on the active device and recomputing sample94 contact evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sampling.pelvis_contact_flow_projection_v0_1 import (  # noqa: E402
    DOSE_FIRST_CONTACT_ABLATION_PROTOCOL,
    write_strict_json,
)
from evaluation.pelvis_contact_compensation_v3 import (  # noqa: E402
    contact_evidence,
    foot_patches,
    patch_centres,
    patch_hash,
    select_stable_window,
)
from evaluation.relative_root_trunk_v2_1 import direct_smpl_parameters  # noqa: E402
from motion_rep.pose_authority import authority_project  # noqa: E402
from motion_rep.smplx_utils import default_smpl_model_path  # noqa: E402
from smplx import SMPLX  # noqa: E402


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(child.relative_to(path).as_posix().encode())
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


def _vertices(motion: torch.Tensor, model: SMPLX, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        params = direct_smpl_parameters(motion.unsqueeze(0).to(device))
        params = {key: value[0] for key, value in params.items()}
        return model(**params, return_verts=True).vertices.detach().float().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-protocol-root", type=Path, required=True)
    parser.add_argument(
        "--m0-run-root",
        type=Path,
        required=True,
        help="current-environment M0 replay containing m0_official_norm_batch.pt",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    source = args.source_protocol_root
    output = args.output_root
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {output}")
    source_protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    cases = [
        case for case in source_protocol.get("cases", [])
        if str(case.get("sample_id")) == "94"
    ]
    if len(cases) != 1:
        raise ValueError("source protocol must contain exactly one sample94 case")
    output.mkdir(parents=True)
    source_inputs = dict(source_protocol.get("inputs", {}))
    mean_path = Path(source_inputs["mean"]["path"])
    std_path = Path(source_inputs["std"]["path"])
    valid_path = Path(source_inputs["valid_mask"]["path"])
    m0_dir = args.m0_run_root / "m0_artifacts" / "batch_000"
    m0_norm_path = m0_dir / "m0_official_norm_batch.pt"
    if not m0_norm_path.is_file():
        raise FileNotFoundError(f"M0 replay has no normalized official tensor: {m0_norm_path}")
    m0_norm = torch.load(m0_norm_path, map_location="cpu", weights_only=True).float()
    valid_all = torch.load(valid_path, map_location="cpu", weights_only=True).bool()
    mean = torch.from_numpy(np.load(mean_path)).float()
    std = torch.from_numpy(np.load(std_path)).float()
    if m0_norm.ndim != 3 or m0_norm.shape[-1] != 276:
        raise ValueError("M0 replay tensor must have shape [B,T,276]")
    if valid_all.shape != m0_norm.shape[:2]:
        raise ValueError("M0 replay and frozen valid mask dimensions differ")
    device = torch.device(args.device)
    # Freeze the exact authority boundary used by the sampler.  Doing this on
    # the active device avoids silently comparing a GPU authority result with a
    # CPU-rebuilt endpoint whose rotation kernels can differ by milliradians.
    physical = m0_norm.to(device) * std.to(device).view(1, 1, -1) + mean.to(device).view(1, 1, -1)
    physical = authority_project(
        physical, valid_mask=valid_all.to(device), output_dtype=torch.float32
    ).physical_motion.detach().cpu()
    model_path = Path(source_inputs.get("smplx_model", {}).get("path") or default_smpl_model_path("smplx", ROOT))
    model = SMPLX(
        model_path=str(model_path),
        gender="neutral",
        num_betas=10,
        batch_size=int(m0_norm.shape[1]),
        use_pca=False,
    ).to(device)
    patches = foot_patches(model)
    (output / "foot_patches.json").write_text(
        json.dumps(patches, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch.save(physical[:1], output / "m0_physical.pt")
    torch.save(valid_all[:1], output / "valid_mask.pt")
    protocol = dict(source_protocol)
    cases[0] = dict(cases[0])
    cases[0]["source_index"] = 0
    cases[0]["row_index"] = 0
    # Recompute sample94 contact evidence against this current-environment M0.
    sides: dict[str, Any] = {}
    vertices = _vertices(physical[0], model, device)
    for side in ("left", "right"):
        heel, toe = patch_centres(vertices, patches[side])
        evidence = contact_evidence(heel, toe, valid_mask=valid_all[0])
        confidence = torch.tensor(evidence["confidence"], dtype=torch.float32)
        window = select_stable_window(
            confidence,
            valid_mask=valid_all[0],
            stable_mask=torch.as_tensor(evidence["valid_masks"]["flat_contact"], dtype=torch.bool),
            pad=4,
        )
        sides[side] = {"evidence": evidence, "stable_window": window}
    cases[0]["sides"] = sides
    protocol.update(
        {
            "protocol": DOSE_FIRST_CONTACT_ABLATION_PROTOCOL,
            "protocol_revision": "v0.4",
            "baseline_origin": "current_environment_refreeze",
            "legacy_v3_relation": "reference_only",
            "m0_reference_protocol": str(source_protocol.get("protocol", "")),
            "source_protocol": {
                "path": str(source / "protocol.json"),
                "sha256": sha256_path(source / "protocol.json"),
            },
            "m0_boundary": "current_run_official_pre_cast_to_active_device_authority_project_to_frozen_physical_m0",
            "projection_scope": "full_sequence",
            "sample_id": "94",
            "contact_ablation_modes": {
                "dose_only": {"contact_position_weight": 0.0, "contact_velocity_weight": 0.0},
                "position_only_medium": {"contact_position_weight": 1.0e5, "contact_velocity_weight": 0.0},
                "temporal_weak": {"contact_position_weight": 1.0e4, "contact_velocity_weight": 1.0e4},
                "temporal_medium": {"contact_position_weight": 1.0e5, "contact_velocity_weight": 1.0e5},
                "temporal_strong": {"contact_position_weight": 1.0e6, "contact_velocity_weight": 1.0e6},
            },
            "cases": cases,
            "artifacts": {
                **dict(source_protocol.get("artifacts", {})),
                "source_protocol": "source_protocol.json",
                "m0_physical": "m0_physical.pt",
                "valid_mask": "valid_mask.pt",
                "foot_patches": "foot_patches.json",
            },
            "git": {
                "revision": git_value("rev-parse", "HEAD"),
                "branch": git_value("branch", "--show-current"),
                "dirty_state": git_value("status", "--porcelain"),
            },
        }
    )
    protocol["inputs"] = dict(protocol.get("inputs", {}))
    protocol["inputs"]["m0_official_pre_cast"] = {
        "path": str(m0_norm_path),
        "sha256": sha256_path(m0_norm_path),
    }
    protocol["inputs"]["m0_run"] = {
        "path": str(args.m0_run_root),
        "sha256": sha256_path(args.m0_run_root),
    }
    protocol["inputs"]["source_protocol"] = {
        "path": str(source / "protocol.json"),
        "sha256": sha256_path(source / "protocol.json"),
    }
    protocol["inputs"]["mean"] = {"path": str(mean_path), "sha256": sha256_path(mean_path)}
    protocol["inputs"]["std"] = {"path": str(std_path), "sha256": sha256_path(std_path)}
    protocol["inputs"]["valid_mask"] = {"path": str(valid_path), "sha256": sha256_path(valid_path)}
    protocol["inputs"]["smplx_model"] = {"path": str(model_path), "sha256": sha256_path(model_path) if model_path.exists() else None}
    protocol["foot_patches"] = {
        "source": "neutral_SMPL-X_template_bottom_quartile_and_skinning_weights",
        "sha256": patch_hash(patches),
        "file": "foot_patches.json",
    }
    write_strict_json(output / "protocol.json", protocol)
    environment = source / "environment_fingerprint.json"
    if environment.is_file():
        shutil.copy2(environment, output / "environment_fingerprint.json")
    else:
        write_strict_json(output / "environment_fingerprint.json", protocol.get("runtime_fingerprint", {}))
    for path in output.iterdir():
        path.chmod(0o444)
    output.chmod(0o555)
    print(json.dumps({"protocol": DOSE_FIRST_CONTACT_ABLATION_PROTOCOL, "output": str(output), "source": str(source)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
