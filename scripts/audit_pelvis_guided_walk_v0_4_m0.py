#!/usr/bin/env python3
"""Audit current-environment M0 replay consistency for v0.4 sample94."""

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

from motion_rep.pose_authority import authority_project  # noqa: E402
from sampling.pelvis_contact_flow_projection_v0_1 import write_strict_json  # noqa: E402


def _load(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True).float()


def _max_direct(left: torch.Tensor, right: torch.Tensor) -> float:
    spans = (slice(0, 126), slice(258, 264), slice(270, 273))
    return max(float((left[..., span] - right[..., span]).abs().max()) for span in spans)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--dual-runs", type=Path, nargs=2, required=True)
    parser.add_argument("--singleton-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.protocol_root / "protocol.json").read_text(encoding="utf-8"))
    mean = torch.from_numpy(np.load(protocol["inputs"]["mean"]["path"])).float()
    std = torch.from_numpy(np.load(protocol["inputs"]["std"]["path"])).float()
    # The v0.4 frozen protocol contains only sample94, while the replay
    # archives used for the batch-size audit still contain the original
    # sample94/sample34122 pair.  Use the immutable source mask for replay
    # conversion and the one-row frozen mask for the reference comparison.
    source_valid_path = protocol.get("inputs", {}).get("valid_mask", {}).get("path")
    valid_all = _load(Path(source_valid_path)).bool() if source_valid_path else _load(args.protocol_root / "valid_mask.pt").bool()
    frozen = _load(args.protocol_root / "m0_physical.pt")
    runs = []
    for label, root in [("dual_01", args.dual_runs[0]), ("dual_02", args.dual_runs[1]), ("singleton_01", args.singleton_run)]:
        tensor = _load(root / "m0_artifacts" / "batch_000" / "m0_official_norm_batch.pt")
        valid = valid_all[: tensor.shape[0]]
        physical = authority_project(tensor * std.view(1, 1, -1) + mean.view(1, 1, -1), valid_mask=valid).physical_motion
        runs.append({"label": label, "root": str(root), "shape": list(tensor.shape), "sample94_norm_sha256": __import__("hashlib").sha256(tensor[0].numpy().tobytes()).hexdigest(), "physical": physical})
    ref = frozen[0:1]
    first = runs[0]["physical"][0:1]
    comparisons = []
    for item in runs:
        current = item["physical"][0:1]
        comparisons.append({"label": item["label"], "full_max_abs": float((current - ref).abs().max()), "direct_max_abs": _max_direct(current, ref), "repeat_full_max_abs": float((current - first).abs().max()), "repeat_direct_max_abs": _max_direct(current, first)})
    passed = all(item["repeat_direct_max_abs"] <= 2.0e-3 and item["repeat_full_max_abs"] <= 2.0e-3 for item in comparisons)
    output = {
        "protocol": protocol.get("protocol"),
        "sample_id": "94",
        "baseline_origin": "current_environment_refreeze",
        "m0_boundary": "official_pre_cast_to_authority_project_to_frozen_physical_m0",
        "runs": [{key: value for key, value in item.items() if key != "physical"} for item in runs],
        "comparisons_to_frozen_m0": comparisons,
        "status": "M0_PAIRING_PASS" if passed else "M0_PAIRING_FAIL",
        "eligible": passed,
        "criteria": {"repeat_direct_max_abs": 2.0e-3, "repeat_full_max_abs": 2.0e-3, "noise_and_mask_reused": True},
    }
    write_strict_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
