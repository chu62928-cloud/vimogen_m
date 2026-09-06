#!/usr/bin/env python3
"""Diagnose B1 lower-body spatial feasibility on saved endpoints only.

This entry point never calls the ViMoGen sampler.  It reuses the frozen
sample94 protocol and one saved pre-cast endpoint, then evaluates marker
subsets and bounded root-translation slack on the frozen flat-contact frames.
All runs are diagnostic and are written to a new output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sampling.lower_body_terminal_compensation import (  # noqa: E402
    LOWER_BODY_DOF_PROTOCOL,
    LowerBodyDofMap,
    LowerBodySolverConfig,
    solve_lower_body_position,
)
from sampling.pelvis_contact_flow_projection_v0_1 import write_strict_json  # noqa: E402
from scripts.evaluate_pelvis_guided_walk_v0_4_1_terminal_dead_zone import (  # noqa: E402
    _as_batch,
    _load_frozen,
    _load_motion,
    _resolve_run_root,
)


MARKER_MODES: dict[str, dict[str, tuple[str, ...]]] = {
    "frozen": {"left": ("heel", "toe"), "right": ("heel", "toe")},
    "left_only": {"left": ("heel", "toe"), "right": ()},
    "right_only": {"left": (), "right": ("heel", "toe")},
    "left_heel_only": {"left": ("heel",), "right": ()},
    "left_toe_only": {"left": ("toe",), "right": ()},
    "right_heel_only": {"left": (), "right": ("heel",)},
    "right_toe_only": {"left": (), "right": ("toe",)},
}
ROOT_LIMITS_MM = (None, 1.0, 5.0, 10.0)


def _frozen_flat_frames(sides: dict[str, Any], frame_count: int) -> list[int]:
    frames: set[int] = set()
    for side in ("left", "right"):
        mask = sides[side]["evidence"]["valid_masks"]["flat_contact"]
        if len(mask) != frame_count:
            raise ValueError(f"{side} flat_contact mask length does not match endpoint")
        frames.update(index for index, active in enumerate(mask) if bool(active))
    return sorted(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v04-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dead-zone", type=float, default=0.0)
    parser.add_argument("--dose", type=float, default=2.0)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic directory: {output}")

    protocol, mean, std, m0_batch, valid_batch, patches, sides = _load_frozen(args.protocol_root)
    summary_path = args.summary or (args.v04_root / "v0_4_ablation_summary.json")
    rows = json.loads(summary_path.read_text(encoding="utf-8"))["rows"]
    source = next(
        row for row in rows
        if float(row["dose"]) == args.dose and str(row["mode"]) == "position_only_medium"
    )
    endpoint_path = (
        _resolve_run_root(args.v04_root, source["run_root"])
        / "projection_artifacts"
        / "terminal_projection_endpoints.pt"
    )
    endpoint = torch.load(endpoint_path, map_location="cpu", weights_only=True)
    pre_norm = _as_batch(endpoint["official_pre_cast_norm"], "official_pre_cast_norm")
    pre = _load_motion(pre_norm, mean, std, valid_batch, "official_pre_cast")[0]
    m0 = m0_batch[0]
    valid = valid_batch[0]
    frame_indices = _frozen_flat_frames(sides, pre.shape[0])
    device = torch.device(args.device)

    from smplx import SMPLX

    model = SMPLX(
        model_path=protocol["inputs"]["smplx_model"]["path"],
        gender="neutral",
        num_betas=10,
        batch_size=int(valid.sum()),
        use_pca=False,
    ).to(device)
    evidence = {side: sides[side]["evidence"] for side in ("left", "right")}
    floor_heights = {
        side: float(sides[side]["evidence"]["floor_height_m"])
        for side in ("left", "right")
    }
    ground_axis_index = int(protocol.get("ground_axis_index", 2))
    ground_axis_source = (
        "frozen_protocol"
        if protocol.get("ground_axis_index") is not None
        else "frozen_v0_4_default_axis_2"
    )

    records: list[dict[str, Any]] = []
    for marker_mode, marker_filter in MARKER_MODES.items():
        for root_limit_mm in ROOT_LIMITS_MM:
            root_limit_m = None if root_limit_mm is None else root_limit_mm / 1000.0
            solved = solve_lower_body_position(
                pre.to(device),
                m0.to(device),
                valid.to(device),
                args.dose,
                model=model,
                patches=patches,
                dead_zone_deg=args.dead_zone,
                dof_map=LowerBodyDofMap.default(),
                config=LowerBodySolverConfig(),
                floor_heights=floor_heights,
                contact_evidence=evidence,
                ground_axis_index=ground_axis_index,
                ground_axis_source=ground_axis_source,
                marker_filter=marker_filter,
                root_translation_limit_m=root_limit_m,
                frame_indices=frame_indices,
            )
            records.append(
                {
                    "marker_mode": marker_mode,
                    "marker_filter": marker_filter,
                    "root_translation_limit_mm": root_limit_mm,
                    "frame_indices": frame_indices,
                    "solver": solved.diagnostics(),
                }
            )

    output.mkdir(parents=True)
    write_strict_json(
        output / "lower_body_feasibility_diagnostic.json",
        {
            "protocol": "vimogen_pelvis_guided_walk_v0_4_2_lower_body_feasibility_diagnostic_v1",
            "solver_protocol": LOWER_BODY_DOF_PROTOCOL,
            "sampler_invoked": False,
            "dose_deg": args.dose,
            "dead_zone_deg": args.dead_zone,
            "source_endpoint": str(endpoint_path),
            "source_endpoint_sha256": __import__("hashlib").sha256(endpoint_path.read_bytes()).hexdigest(),
            "flat_contact_frame_indices": frame_indices,
            "ground_axis_index": ground_axis_index,
            "ground_axis_source": ground_axis_source,
            "root_translation_limits_mm": [item for item in ROOT_LIMITS_MM if item is not None],
            "records": records,
        },
    )
    print(json.dumps({"output": str(output), "records": len(records), "flat_frames": frame_indices}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
