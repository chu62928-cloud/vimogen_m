#!/usr/bin/env python3
"""Resumable bounded-parallel scheduler for the MBench metric repair."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


CONDITIONS = ("m0", "m1_plus5", "m1_plus10")
METHODS = ("absolute_position", "velocity_integral", "reconciled")
SEEDS = (0, 1, 2)
CGROUP_ROOT = Path("/sys/fs/cgroup")


def status(path: Path) -> str | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def cgroup_cpu_quota() -> float:
    quota, period = (CGROUP_ROOT / "cpu.max").read_text(encoding="utf-8").split()
    if quota == "max":
        return float(os.cpu_count() or 1)
    return int(quota) / int(period)


def cgroup_cpu_usage_usec() -> int:
    fields = dict(
        line.split() for line in (CGROUP_ROOT / "cpu.stat").read_text(encoding="utf-8").splitlines()
    )
    return int(fields["usage_usec"])


def inspect_resources(
    output_root: Path,
    previous_cpu: tuple[float, int] | None = None,
) -> tuple[dict[str, float | None], tuple[float, int]]:
    """Measure real container limits rather than the visible host CPU count."""

    wall_time = time.monotonic()
    cpu_usage = cgroup_cpu_usage_usec()
    cpu_quota = cgroup_cpu_quota()
    cpu_percent = None
    if previous_cpu is not None and wall_time > previous_cpu[0]:
        elapsed = wall_time - previous_cpu[0]
        used = (cpu_usage - previous_cpu[1]) / 1_000_000
        cpu_percent = min(100.0, max(0.0, used / elapsed / cpu_quota * 100.0))

    memory_current = int((CGROUP_ROOT / "memory.current").read_text(encoding="utf-8"))
    memory_limit_text = (CGROUP_ROOT / "memory.max").read_text(encoding="utf-8").strip()
    if memory_limit_text == "max":
        raise RuntimeError("container memory limit is not finite")
    memory_limit = int(memory_limit_text)
    memory_fields = dict(
        line.split() for line in (CGROUP_ROOT / "memory.stat").read_text(encoding="utf-8").splitlines()
    )
    inactive_file = int(memory_fields.get("inactive_file", 0))
    memory_working_set = max(0, memory_current - inactive_file)

    gpu_line = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip().splitlines()[0]
    gpu_utilization, gpu_memory_used, gpu_memory_total, gpu_temperature = [
        float(part.strip()) for part in gpu_line.split(",")
    ]
    disk_free_gib = shutil.disk_usage(output_root).free / 1024**3

    snapshot = {
        "cpu_quota_cores": cpu_quota,
        "cpu_percent": cpu_percent,
        "gpu_utilization_percent": gpu_utilization,
        "gpu_memory_used_mib": gpu_memory_used,
        "gpu_memory_total_mib": gpu_memory_total,
        "gpu_memory_percent": gpu_memory_used / gpu_memory_total * 100.0,
        "gpu_temperature_c": gpu_temperature,
        "memory_current_gib": memory_current / 1024**3,
        "memory_working_set_gib": memory_working_set / 1024**3,
        "memory_limit_gib": memory_limit / 1024**3,
        "memory_working_set_percent": memory_working_set / memory_limit * 100.0,
        "disk_free_gib": disk_free_gib,
    }
    return snapshot, (wall_time, cpu_usage)


def resource_blockers(snapshot: dict[str, float | None], args: argparse.Namespace) -> list[str]:
    blockers = []
    cpu_percent = snapshot.get("cpu_percent")
    if cpu_percent is not None and cpu_percent >= args.max_cpu_percent:
        blockers.append(f"cpu={cpu_percent:.1f}%")
    if snapshot["gpu_utilization_percent"] >= args.max_gpu_utilization:
        blockers.append(f"gpu={snapshot['gpu_utilization_percent']:.1f}%")
    if snapshot["gpu_memory_percent"] >= args.max_gpu_memory_percent:
        blockers.append(f"gpu_memory={snapshot['gpu_memory_percent']:.1f}%")
    if snapshot["memory_working_set_percent"] >= args.max_memory_working_set_percent:
        blockers.append(f"memory={snapshot['memory_working_set_percent']:.1f}%")
    if snapshot["disk_free_gib"] < args.min_free_disk_gib:
        blockers.append(f"disk_free={snapshot['disk_free_gib']:.1f}GiB")
    if snapshot["gpu_temperature_c"] >= args.max_gpu_temperature:
        blockers.append(f"gpu_temperature={snapshot['gpu_temperature_c']:.0f}C")
    return blockers


def cache_count(organized_root: Path) -> int:
    return sum(
        len(list((organized_root / condition / f"seed_{seed:03d}" / method).glob("*.pt")))
        for condition in CONDITIONS
        for seed in SEEDS
        for method in METHODS
    )


def write_scheduler_state(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def abort_children(children: dict[tuple[str, int, str], subprocess.Popen]) -> None:
    for child in children.values():
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                continue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--organized-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--initial-workers", type=int, default=4)
    parser.add_argument("--launch-step", type=int, default=2)
    parser.add_argument("--launch-stagger-seconds", type=float, default=3.0)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--max-cpu-percent", type=float, default=85.0)
    parser.add_argument("--max-gpu-utilization", type=float, default=90.0)
    parser.add_argument("--max-gpu-memory-percent", type=float, default=70.0)
    parser.add_argument("--max-memory-working-set-percent", type=float, default=75.0)
    parser.add_argument("--max-gpu-temperature", type=float, default=78.0)
    parser.add_argument("--min-free-disk-gib", type=float, default=64.0)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.initial_workers <= args.max_workers:
        parser.error("initial workers must be between 1 and max workers")
    if args.launch_step < 1 or args.poll_seconds < 1:
        parser.error("launch step and poll seconds must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = [(condition, seed, method) for condition in CONDITIONS for seed in SEEDS for method in METHODS]
    children: dict[tuple[str, int, str], subprocess.Popen] = {}
    state_path = args.output_root / "scheduler_state.json"
    previous_cpu = None
    first_iteration = True
    started_at = datetime.now(timezone.utc).isoformat()

    while True:
        failures = []
        for job, child in list(children.items()):
            if child.poll() is not None:
                print(f"DONE {job} returncode={child.returncode}", flush=True)
                if child.returncode != 0:
                    failures.append({"job": list(job), "returncode": child.returncode})
                del children[job]

        pending = []
        completed_count = 0
        for condition, seed, method in jobs:
            relative = Path(condition) / f"seed_{seed:03d}" / method
            record = args.output_root / relative / "run_record.json"
            current = status(record)
            if current == "COMPLETED":
                completed_count += 1
                continue
            if current == "FAILED":
                failures.append({"job": [condition, seed, method], "record": str(record)})
                continue
            if current == "RUNNING" or (condition, seed, method) in children:
                continue
            pending.append((condition, seed, method))

        snapshot, previous_cpu = inspect_resources(args.output_root, previous_cpu)
        blockers = resource_blockers(snapshot, args)
        current_state = {
            "status": "FAILED" if failures else "RUNNING",
            "started_at": started_at,
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "max_workers": args.max_workers,
            "active_workers": len(children),
            "completed_jobs": completed_count,
            "pending_jobs": len(pending),
            "failed_jobs": failures,
            "smplify_cache_count": cache_count(args.organized_root),
            "expected_smplify_cache_count": len(jobs) * 100,
            "resources": snapshot,
            "resource_blockers": blockers,
        }
        write_scheduler_state(state_path, current_state)
        if failures:
            print(f"ABORT failures={failures}", flush=True)
            abort_children(children)
            return 1

        launches_remaining = args.initial_workers if first_iteration else args.launch_step
        first_iteration = False
        while (
            pending
            and len(children) < args.max_workers
            and launches_remaining > 0
            and not blockers
        ):
            condition, seed, method = pending.pop(0)
            relative = f"{condition}/seed_{seed:03d}/{method}"
            log = args.output_root / relative / "scheduler.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(Path(__file__).with_name("run_official_mbench_repair_worker.py")),
                "--organized-root", str(args.organized_root),
                "--original-root", str(args.original_root),
                "--output-root", str(args.output_root),
                "--condition", condition,
                "--seed", str(seed),
                "--method", method,
                "--device", args.device,
            ]
            handle = log.open("a", encoding="utf-8")
            child = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            handle.close()
            children[(condition, seed, method)] = child
            print(f"LAUNCH {relative} pid={child.pid}", flush=True)
            launches_remaining -= 1
            if launches_remaining > 0:
                time.sleep(args.launch_stagger_seconds)

        current_state.update({
            "active_workers": len(children),
            "pending_jobs": len(pending),
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
        })
        write_scheduler_state(state_path, current_state)

        if not pending and not children:
            if args.summary_output is not None:
                summary_command = [
                    sys.executable,
                    str(Path(__file__).with_name("summarize_official_mbench_threeway.py")),
                    "--run-record-root", str(args.output_root),
                    "--output", str(args.summary_output),
                ]
                completed = subprocess.run(summary_command, check=False)
                if completed.returncode != 0:
                    current_state.update({
                        "status": "FAILED",
                        "summary_returncode": completed.returncode,
                        "heartbeat_at": datetime.now(timezone.utc).isoformat(),
                    })
                    write_scheduler_state(state_path, current_state)
                    print("REPAIR_SUMMARY_FAILED", flush=True)
                    return 1
                current_state["summary_output"] = str(args.summary_output)
            current_state.update({"status": "COMPLETED", "heartbeat_at": datetime.now(timezone.utc).isoformat()})
            write_scheduler_state(state_path, current_state)
            print("REPAIR_SCHEDULER_COMPLETE", flush=True)
            return 0
        print(
            f"SCHEDULER active={len(children)} pending={len(pending)} completed={completed_count} "
            f"cached={current_state['smplify_cache_count']}/{len(jobs) * 100} "
            f"cpu={snapshot['cpu_percent']} gpu={snapshot['gpu_utilization_percent']:.0f}% "
            f"vram={snapshot['gpu_memory_used_mib']:.0f}/{snapshot['gpu_memory_total_mib']:.0f}MiB "
            f"working_memory={snapshot['memory_working_set_gib']:.1f}GiB "
            f"blockers={blockers}",
            flush=True,
        )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
