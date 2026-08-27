#!/usr/bin/env python3
"""Run one official MBench motion-quality combination safely.

This worker is intentionally one-condition/one-seed/one-method so several
independent jobs can run concurrently without sharing a result directory.
Existing COMPLETED or RUNNING records are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DIMENSIONS = (
    "Jitter_Degree",
    "Ground_Penetration",
    "Foot_Floating",
    "Foot_Sliding",
    "Dynamic_Degree",
    "Body_Penetration",
    "Pose_Quality",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_per_motion(output_dir: Path) -> Path | None:
    paths = sorted(output_dir.glob("*_per_motion_results.json"), key=lambda p: p.stat().st_mtime)
    return paths[-1] if paths else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--organized-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", required=True, choices=("absolute_position", "velocity_integral", "reconciled"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-count", type=int, default=450)
    args = parser.parse_args()

    eval_dir = args.organized_root / args.condition / f"seed_{args.seed:03d}" / args.method
    output_dir = args.output_root / args.condition / f"seed_{args.seed:03d}" / args.method
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / "run_record.json"
    lock_path = output_dir / ".worker.lock"

    if record_path.exists():
        try:
            existing = json.loads(record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if existing.get("status") == "COMPLETED":
            print(f"SKIP completed {args.condition} seed {args.seed} {args.method}", flush=True)
            return 0
        if existing.get("status") == "RUNNING":
            print(f"SKIP already running {args.condition} seed {args.seed} {args.method}", flush=True)
            return 0

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        print(f"SKIP locked {args.condition} seed {args.seed} {args.method}", flush=True)
        return 0

    try:
        files = sorted(eval_dir.glob("*.npy")) if eval_dir.is_dir() else []
        if len(files) != args.expected_count:
            raise RuntimeError(f"{eval_dir}: expected {args.expected_count} .npy files, found {len(files)}")

        command = [
            sys.executable,
            "evaluate_mbench.py",
            "--evaluation_path", str(eval_dir),
            "--output_path", str(output_dir),
            "--dimension", *DIMENSIONS,
            "--device", args.device,
        ]
        record = {
            "status": "RUNNING",
            "protocol": "vimogen_publication_mbench_motion_quality_v1",
            "condition": args.condition,
            "seed": args.seed,
            "method": args.method,
            "dimensions": list(DIMENSIONS),
            "evaluation_path": str(eval_dir),
            "output_path": str(output_dir),
            "expected_count": args.expected_count,
            "command": command,
            "worker": True,
            "started_at": _now(),
        }
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        log_path = output_dir / "evaluate.log"
        print(f"RUN {args.condition} seed {args.seed} {args.method}", flush=True)
        env = os.environ.copy()
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env,
                                       stdout=log, stderr=subprocess.STDOUT, check=False)
        per_motion = _find_per_motion(output_dir)
        record["returncode"] = completed.returncode
        record["per_motion_path"] = str(per_motion) if per_motion else None
        record["finished_at"] = _now()
        record["status"] = "COMPLETED" if completed.returncode == 0 and per_motion else "FAILED"
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if record["status"] != "COMPLETED":
            return 1
        return 0
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
