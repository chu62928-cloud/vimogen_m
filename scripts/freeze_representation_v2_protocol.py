"""Freeze the v2 FK/angle amendment without touching v1 artifacts.

This script only reads the neutral SMPL-X asset and writes a new protocol
directory.  It deliberately refuses to overwrite an existing protocol file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_rep.consistency_v2 import (  # noqa: E402
    PROTOCOL_NAME,
    SMPLX_22_PARENTS,
    freeze_smplx_neutral_22_skeleton,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    skeleton_path = output.with_name("smplx_neutral_22_skeleton.pt")
    if skeleton_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen skeleton: {skeleton_path}")
    skeleton_report = freeze_smplx_neutral_22_skeleton(source, skeleton_path)
    report = {
        "protocol": PROTOCOL_NAME,
        "amendment": "v2_full_fk_and_local_sagittal_pelvis_angle",
        "status": "FROZEN",
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "layout": {
            "body_pose": [0, 126],
            "joints": [126, 192],
            "joints_velocity": [192, 258],
            "root_rotation": [258, 264],
            "root_rotation_velocity": [264, 270],
            "root_translation": [270, 273],
            "root_translation_velocity": [273, 276],
        },
        "skeleton": skeleton_report,
        "parents": list(SMPLX_22_PARENTS),
        "authority": "body_local_rotation + fused_root_rotation + fused_root_translation -> differentiable FK",
        "repack": "FK 22 joints then recompute dJ, dR and dT into T x 276",
        "pelvis_angle": {
            "definition": "signed pelvis_to_spine angle in per-frame hip-defined local sagittal plane",
            "yaw_invariant": True,
            "degenerate_fallback": "hip -> shoulder -> deterministic horizontal axis",
            "differentiable": True,
        },
        "v1_compatibility": "legacy reconciliation/finalizers and existing result directories are unchanged",
        "skeleton_artifact": str(skeleton_path),
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data/body_models/smplx/SMPLX_NEUTRAL.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase7/representation_v2/protocol.json")
    args = parser.parse_args()
    report = freeze(args.source, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
