#!/usr/bin/env python3
"""Audit exact zero-dose identity for real M1--M6 generation attempts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ACTION_TOLERANCE = 1.0e-7
ANGLE_TOLERANCE_DEG = 1.0e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _one(root: Path, pattern: str) -> Path:
    paths = list(root.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {pattern!r} below {root}, found {len(paths)}")
    return paths[0]


def _audit_one(root: Path) -> list[dict[str, Any]]:
    record = json.loads((root / "run_record.json").read_text(encoding="utf-8"))
    if float(record.get("target_dose_deg", 1.0)) != 0.0:
        raise ValueError(f"not a zero-dose run: {root}")
    artifact = root / "guided_artifacts/batch_000"
    m0 = torch.load(
        artifact / "m0_authority_norm_batch.pt", map_location="cpu", weights_only=True
    ).float()
    candidate = torch.load(
        artifact / "g0_norm_batch.pt", map_location="cpu", weights_only=True
    ).float()
    archive_path = _one(root, "trainer/**/batch_000/mbench_raw_norm_batch.pt")
    archive = torch.load(archive_path, map_location="cpu", weights_only=True)
    sample_ids = [str(value) for value in archive["sample_ids"]]
    if tuple(m0.shape) != tuple(candidate.shape):
        raise ValueError(f"M0/candidate shape mismatch below {root}")
    action_difference = (candidate - m0).abs()
    evaluation_paths = sorted((root / "evaluation").glob("*/run_record.json"))
    if len(evaluation_paths) != int(record.get("sample_count", len(evaluation_paths))):
        # ``sample_ids`` in historical runner records was hard-coded to two;
        # the evaluation records are the authoritative per-sample count.
        if not evaluation_paths:
            raise ValueError(f"missing zero-dose evaluation records below {root}")
    rows = []
    for evaluation_path in evaluation_paths:
        evaluated = json.loads(evaluation_path.read_text(encoding="utf-8"))
        control = evaluated["all_metrics"]["control"]["per_sequence"][0]
        content = evaluated["all_metrics"]["content"]["per_sequence"][0]
        sample_id = str(evaluated["prompt_id"])
        if sample_id not in sample_ids:
            raise ValueError(f"evaluation sample {sample_id} missing from archive below {root}")
        index = sample_ids.index(sample_id)
        action_max = float(action_difference[index].max())
        angle_drift = float(control.get("zero_dose_drift_mae_deg", 0.0))
        row = {
            "method": record["method"],
            "method_version": record.get("method_version", "v1"),
            "seed": int(record["seed"]),
            "sample_id": sample_id,
            "run_root": str(root),
            "evaluation_record": str(evaluation_path),
            "action_max_abs_norm": action_max,
            "angle_drift_mae_deg": angle_drift,
            "mpjpe_vs_m0_mm": float(content["mpjpe_vs_m0_mm"]),
            "root_translation_deviation_p95_mm": float(
                content["root_translation_deviation_p95_mm"]
            ),
            "status": evaluated["status"],
            "passed": bool(
                action_max <= ACTION_TOLERANCE
                and angle_drift <= ANGLE_TOLERANCE_DEG
                and float(content["mpjpe_vs_m0_mm"]) <= ACTION_TOLERANCE * 1000.0
                and float(content["root_translation_deviation_p95_mm"])
                <= ACTION_TOLERANCE * 1000.0
            ),
        }
        rows.append(row)
    return rows


def audit(run_roots: list[Path]) -> dict[str, Any]:
    if not run_roots:
        raise ValueError("at least one zero-dose run is required")
    rows = [row for root in run_roots for row in _audit_one(root)]
    return {
        "protocol": "vimogen_m1_m7_zero_dose_identity_v1",
        "status": "PASS" if rows and all(row["passed"] for row in rows) else "FAIL",
        "action_tolerance_max_abs_norm": ACTION_TOLERANCE,
        "angle_tolerance_mae_deg": ANGLE_TOLERANCE_DEG,
        "run_count": len(run_roots),
        "sequence_count": len(rows),
        "run_roots": [str(root) for root in run_roots],
        "rows": rows,
        "s1_tuning_allowed": bool(rows and all(row["passed"] for row in rows)),
        "source_hashes": {
            str(root): _sha256(root / "run_record.json") for root in run_roots
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite zero-dose report: {args.output}")
    result = audit(args.run)
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "sequence_count": result["sequence_count"]}))


if __name__ == "__main__":
    main()
