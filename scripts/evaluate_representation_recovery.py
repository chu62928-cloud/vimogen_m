#!/usr/bin/env python3
"""Run paired direct/velocity/reconciled recovery metrics on a frozen split."""

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

from evaluation.representation_recovery import (  # noqa: E402
    CorruptionConfig,
    calibrate_corruption,
    evaluate_one,
    summarize,
)
from motion_rep.reconciliation import ReconciliationConfig  # noqa: E402


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN":
        raise ValueError(f"manifest is not frozen: {path}")
    return payload


def _load_motion(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        value = value.get("motion")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"missing motion tensor at {path}")
    return value.float()


def run(
    *,
    manifest: Path,
    motion_root: Path,
    output: Path,
    calibration_json: Path | None,
    max_items: int | None,
    seed: int,
    correction_window: int,
    anchor_weight: float,
) -> dict[str, Any]:
    payload = _load_manifest(manifest)
    rows = payload["items"][:max_items] if max_items is not None else payload["items"]
    if calibration_json is None:
        if payload["role"] != "development":
            raise ValueError("validation/test evaluation requires a calibration JSON made on development data")
        motions = [_load_motion(motion_root / str(row["motion_path"])) for row in rows]
        corruption = calibrate_corruption(motions)
        calibration_source = "representation_dev_v1"
    else:
        calibration_payload = json.loads(calibration_json.read_text(encoding="utf-8"))
        corruption = CorruptionConfig(**calibration_payload["corruption"])
        calibration_source = str(calibration_json)
    reconciliation = ReconciliationConfig(
        correction_window=correction_window,
        anchor_weight=anchor_weight,
        root_rotation_anchor_weight=anchor_weight,
    )
    records = []
    for index, row in enumerate(rows, start=1):
        motion = _load_motion(motion_root / str(row["motion_path"]))
        records.append(
            evaluate_one(
                motion,
                sample_key=str(row["id"]),
                corruption=corruption,
                reconciliation=reconciliation,
                seed=seed,
            )
        )
        if index % 250 == 0:
            print(f"evaluated {index}/{len(rows)}", flush=True)
    report = {
        "status": "VERIFIED_REPRESENTATION_RECOVERY_EVALUATION",
        "protocol": "representation_source_holdout_v1",
        "manifest": str(manifest),
        "role": payload["role"],
        "calibration_source": calibration_source,
        "sample_count": len(records),
        "reconciliation": {
            "correction_window": correction_window,
            "anchor_weight": anchor_weight,
        },
        "corruption": corruption.as_dict(),
        "summary": summarize(records),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--correction-window", type=int, default=9)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    args = parser.parse_args()
    report = run(
        manifest=args.manifest,
        motion_root=args.motion_root,
        output=args.output,
        calibration_json=args.calibration_json,
        max_items=args.max_items,
        seed=args.seed,
        correction_window=args.correction_window,
        anchor_weight=args.anchor_weight,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
