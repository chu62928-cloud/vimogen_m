#!/usr/bin/env python3
"""Repair the two MBench dimensions that were unavailable in v1.

The generation and the other five dimensions are reused from v1. Each worker
evaluates Body_Penetration and Pose_Quality into a new versioned directory,
then merges those dimensions with the immutable v1 per-motion file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPAIR_DIMENSIONS = ("Body_Penetration", "Pose_Quality")
ALL_DIMENSIONS = (
    "Jitter_Degree",
    "Ground_Penetration",
    "Foot_Floating",
    "Foot_Sliding",
    "Dynamic_Degree",
    *REPAIR_DIMENSIONS,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def latest_per_motion(directory: Path) -> Path | None:
    paths = sorted(directory.glob("*_per_motion_results.json"), key=lambda p: p.stat().st_mtime)
    return paths[-1] if paths else None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_repaired_motions(path: Path, expected_count: int) -> None:
    """Reject evaluators that silently skipped official pose-quality samples."""

    payload = load_json(path)
    motions = payload.get("motions", [])
    if len(motions) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} repaired motions, found {len(motions)} in {path}"
        )
    observed_ids = {str(item.get("id")) for item in motions}
    if len(observed_ids) != expected_count:
        raise RuntimeError("repaired motions contain duplicate identifiers")


def merge_per_motion(old_path: Path, repaired_path: Path, output_path: Path) -> int:
    old = load_json(old_path)
    repaired = load_json(repaired_path)
    old_motions = old.get("motions", [])
    repaired_motions = repaired.get("motions", [])
    if not repaired_motions:
        raise RuntimeError("repaired evaluation contains no motions")

    merged_by_id: dict[str, dict] = {}
    for item in old_motions:
        motion_id = str(item.get("id"))
        if motion_id in merged_by_id:
            raise RuntimeError(f"duplicate source motion id {motion_id}")
        merged_by_id[motion_id] = dict(item)

    repaired_ids: set[str] = set()
    for repaired_item in repaired_motions:
        motion_id = str(repaired_item.get("id"))
        if motion_id in repaired_ids:
            raise RuntimeError(f"duplicate repaired motion id {motion_id}")
        repaired_ids.add(motion_id)

        merged_item = dict(merged_by_id.get(motion_id, repaired_item))
        dimensions = dict(merged_item.get("dimensions", {}))
        for dimension in REPAIR_DIMENSIONS:
            value = repaired_item.get("dimensions", {}).get(dimension)
            numeric = value.get("value") if isinstance(value, dict) else None
            if (
                isinstance(numeric, bool)
                or not isinstance(numeric, (int, float))
                or not math.isfinite(numeric)
            ):
                raise RuntimeError(f"missing numeric {dimension} for motion {motion_id}")
            dimensions[dimension] = value
        merged_item["dimensions"] = dimensions
        merged_by_id[motion_id] = merged_item

    merged = list(merged_by_id.values())
    output = dict(old)
    output["motions"] = merged
    output["repair"] = {
        "protocol": "vimogen_publication_mbench_motion_quality_repair_v1",
        "source_per_motion": str(old_path),
        "repair_per_motion": str(repaired_path),
        "dimensions": list(REPAIR_DIMENSIONS),
        "source_motion_count": len(old_motions),
        "repaired_motion_count": len(repaired_motions),
        "motion_count": len(merged),
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(merged)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--organized-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", required=True, choices=("absolute_position", "velocity_integral", "reconciled"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-count", type=int, default=450)
    parser.add_argument("--expected-repair-count", type=int, default=100)
    args = parser.parse_args()

    relative = Path(args.condition) / f"seed_{args.seed:03d}" / args.method
    eval_dir = args.organized_root / relative
    old_dir = args.original_root / relative
    output_dir = args.output_root / relative
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".worker.lock"
    record_path = output_dir / "run_record.json"

    if record_path.exists():
        existing = load_json(record_path)
        if existing.get("status") == "COMPLETED":
            print(f"SKIP completed {relative}", flush=True)
            return 0
        if existing.get("status") == "RUNNING":
            print(f"SKIP already running {relative}", flush=True)
            return 0
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        print(f"SKIP locked {relative}", flush=True)
        return 0

    command = [
        sys.executable,
        "evaluate_mbench.py",
        "--evaluation_path", str(eval_dir),
        "--output_path", str(output_dir),
        "--dimension", *REPAIR_DIMENSIONS,
        "--device", args.device,
    ]
    env = os.environ.copy()
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    # The host's auto-detected OpenBLAS kernel produces invalid GMM inverses.
    # Pin a verified-compatible kernel before the evaluator imports NumPy.
    env["OPENBLAS_CORETYPE"] = "HASWELL"
    # Keep 8-12 simultaneous evaluators below the container's 25-vCPU quota.
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["MALLOC_ARENA_MAX"] = "2"
    record = {
        "status": "RUNNING",
        "protocol": "vimogen_publication_mbench_motion_quality_repair_v1",
        "condition": args.condition,
        "seed": args.seed,
        "method": args.method,
        "dimensions": list(ALL_DIMENSIONS),
        "repair_dimensions": list(REPAIR_DIMENSIONS),
        "evaluation_path": str(eval_dir),
        "output_path": str(output_dir),
        "source_run_record": str(old_dir / "run_record.json"),
        "expected_count": args.expected_count,
        "expected_repair_count": args.expected_repair_count,
        "command": command,
        "numerical_environment": {
            "OPENBLAS_CORETYPE": env["OPENBLAS_CORETYPE"],
            "OPENBLAS_NUM_THREADS": env["OPENBLAS_NUM_THREADS"],
            "OMP_NUM_THREADS": env["OMP_NUM_THREADS"],
            "MKL_NUM_THREADS": env["MKL_NUM_THREADS"],
            "PYOPENGL_PLATFORM": env["PYOPENGL_PLATFORM"],
        },
        "started_at": now(),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    try:
        files = sorted(eval_dir.glob("*.npy"))
        if len(files) != args.expected_count:
            raise RuntimeError(f"expected {args.expected_count} files, found {len(files)} in {eval_dir}")
        old_record = load_json(old_dir / "run_record.json")
        old_per_motion = Path(old_record["per_motion_path"])
        if not old_per_motion.exists():
            raise RuntimeError(f"source per-motion file does not exist: {old_per_motion}")
        log_path = output_dir / "evaluate.log"
        print(f"RUN {relative}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env,
                                       stdout=log, stderr=subprocess.STDOUT, check=False)
        repaired_per_motion = latest_per_motion(output_dir)
        if completed.returncode != 0 or repaired_per_motion is None:
            raise RuntimeError(f"evaluator failed with return code {completed.returncode}")
        validate_repaired_motions(repaired_per_motion, args.expected_repair_count)
        merged_path = output_dir / "merged_per_motion_results.json"
        motion_count = merge_per_motion(old_per_motion, repaired_per_motion, merged_path)
        record.update({
            "returncode": completed.returncode,
            "repair_per_motion_path": str(repaired_per_motion),
            "per_motion_path": str(merged_path),
            "motion_count": motion_count,
            "finished_at": now(),
            "status": "COMPLETED",
        })
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        record.update({"status": "FAILED", "error": repr(exc), "finished_at": now()})
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"FAILED {relative}: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
