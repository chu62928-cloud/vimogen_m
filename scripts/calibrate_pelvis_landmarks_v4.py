#!/usr/bin/env python3
"""Freeze reviewed LASI/RASI/LPSI/RPSI groups from a neutral SMPL-X template.

The script deliberately requires the vertex groups to be supplied by a human
mesh review.  It never invents an ASIS/PSIS mapping.  The resulting JSON is a
small, portable calibration artifact containing the group means and the
template hash; model weights and review images remain outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_rep.anatomical_pelvis import MARKER_NAMES, PelvisCalibration


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True, help="neutral SMPL-X .npz")
    parser.add_argument("--vertex-groups", type=Path, required=True, help="JSON mapping LASI/RASI/LPSI/RPSI to vertex indices")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-rest-joint", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"), help="neutral FK pelvis joint in the model coordinate frame")
    parser.add_argument("--reviewed-image", type=str, default=None)
    args = parser.parse_args()
    groups_payload = json.loads(args.vertex_groups.read_text(encoding="utf-8"))
    groups = groups_payload.get("marker_vertex_groups", groups_payload)
    if set(groups) != set(MARKER_NAMES):
        raise ValueError(f"vertex-groups must contain exactly {MARKER_NAMES}")
    with np.load(args.model_path, allow_pickle=False) as model:
        if "v_template" not in model:
            raise ValueError("SMPL-X model does not contain v_template")
        vertices = np.asarray(model["v_template"], dtype=np.float64)
    points = {}
    for name in MARKER_NAMES:
        indices = np.asarray(groups[name], dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0 or np.any(indices < 0) or np.any(indices >= len(vertices)):
            raise ValueError(f"invalid vertex group for {name}")
        points[name] = (vertices[indices].mean(axis=0) - np.asarray(args.root_rest_joint, dtype=np.float64)).tolist()
    # Catch accidental left/right swaps without imposing a model-axis claim.
    left_width = np.linalg.norm(np.asarray(points["LASI"]) - np.asarray(points["LPSI"]))
    right_width = np.linalg.norm(np.asarray(points["RASI"]) - np.asarray(points["RPSI"]))
    if min(left_width, right_width) <= 1e-8:
        raise ValueError("ASIS/PSIS groups collapse on one side")
    calibration = PelvisCalibration(
        template_sha256=sha256_file(args.model_path),
        model_path=str(args.model_path),
        marker_vertex_groups={name: tuple(int(v) for v in groups[name]) for name in MARKER_NAMES},
        marker_local_points={name: tuple(float(v) for v in points[name]) for name in MARKER_NAMES},
        root_rest_joint=tuple(float(v) for v in args.root_rest_joint),
        reviewed_image=args.reviewed_image,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibration.to_mapping(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "template_sha256": calibration.template_sha256, "marker_local_points": points}, indent=2))


if __name__ == "__main__":
    main()
