#!/usr/bin/env python3
"""Resume official MBench v3 without touching the previous stopped run.

The stopped run is treated as immutable evidence.  A new output root receives
copies of only its completed job directories; interrupted and absent jobs are
left out so the normal resumable scheduler reruns them from scratch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


JOBS = tuple(
    (condition, seed, method)
    for condition in ("m0", "m1_plus5", "m1_plus10")
    for seed in (0, 1, 2)
    for method in ("absolute_position", "velocity_integral", "reconciled")
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_job(job: tuple[str, int, str]) -> Path:
    condition, seed, method = job
    return Path(condition) / f"seed_{seed:03d}" / method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--resume-root", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--initial-workers", type=int, default=4)
    parser.add_argument("--launch-step", type=int, default=2)
    parser.add_argument("--launch-stagger-seconds", type=float, default=3.0)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source = args.source_root.resolve()
    resume = args.resume_root.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if resume.exists():
        raise FileExistsError(f"refusing to overwrite existing resume root: {resume}")
    state_path = source / "scheduler_state.json"
    state = load_json(state_path)
    if state.get("status") != "USER_STOPPED_ARCHIVED":
        raise RuntimeError(f"source scheduler state is not USER_STOPPED_ARCHIVED: {state.get('status')!r}")
    if state.get("active_workers") != 0:
        raise RuntimeError(f"source still reports active workers: {state.get('active_workers')}")
    resume.mkdir(parents=True)

    completed: list[dict] = []
    interrupted: list[dict] = []
    pending: list[dict] = []
    for job in JOBS:
        relative = relative_job(job)
        source_dir = source / relative
        record_path = source_dir / "run_record.json"
        if not record_path.is_file():
            pending.append({"job": list(job), "reason": "no_source_record"})
            continue
        record = load_json(record_path)
        status = record.get("status")
        if status == "COMPLETED":
            destination_dir = resume / relative
            shutil.copytree(source_dir, destination_dir)
            completed.append(
                {
                    "job": list(job),
                    "source_record": str(record_path),
                    "copied_record": str(destination_dir / "run_record.json"),
                    "source_record_sha256": sha256_file(record_path),
                }
            )
        elif status in {"RUNNING", "FAILED"}:
            interrupted.append({"job": list(job), "source_status": status, "reason": "rerun_in_new_root"})
        else:
            pending.append({"job": list(job), "reason": f"source_status_{status}"})

    manifest = {
        "protocol": "vimogen_publication_mbench_motion_quality_repair_v1",
        "resume_protocol": "official_mbench_v3_resume_no_overwrite",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source),
        "resume_root": str(resume),
        "source_scheduler_state": str(state_path),
        "source_scheduler_state_sha256": sha256_file(state_path),
        "source_completed_count": len(completed),
        "copied_completed_jobs": completed,
        "rerun_interrupted_jobs": interrupted,
        "rerun_pending_jobs": pending,
        "invariant": "completed source job directories are copied read-only in spirit; source root is never modified and scheduler launches only missing jobs under resume_root",
    }
    manifest_path = resume / "resume_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    log_path = resume / "scheduler.log"
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_official_mbench_repair_parallel.py")),
        "--organized-root", str(source.parent / "organized_v1"),
        "--original-root", str(source.parent / "official_motion_quality_v1"),
        "--output-root", str(resume),
        "--summary-output", str(args.summary_output),
        "--max-workers", str(args.max_workers),
        "--initial-workers", str(args.initial_workers),
        "--launch-step", str(args.launch_step),
        "--launch-stagger-seconds", str(args.launch_stagger_seconds),
        "--poll-seconds", str(args.poll_seconds),
        "--max-cpu-percent", "85",
        "--max-gpu-utilization", "90",
        "--max-gpu-memory-percent", "70",
        "--max-memory-working-set-percent", "75",
        "--max-gpu-temperature", "78",
        "--min-free-disk-gib", "64",
        "--device", args.device,
    ]
    env = os.environ.copy()
    env.update({"OPENBLAS_CORETYPE": "HASWELL", "PYOPENGL_PLATFORM": "egl", "PYTHONUNBUFFERED": "1"})
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=source.parent.parent.parent.parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
    pid_path = resume / "scheduler.pid"
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    launch = {
        "status": "STARTED",
        "pid": process.pid,
        "resume_manifest": str(manifest_path),
        "scheduler_log": str(log_path),
        "summary_output": str(args.summary_output),
        "command": command,
        "completed_copied": len(completed),
        "jobs_to_rerun": len(interrupted) + len(pending),
    }
    (resume / "resume_launch.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(launch, indent=2))


if __name__ == "__main__":
    main()
