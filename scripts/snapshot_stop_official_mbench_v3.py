#!/usr/bin/env python3
"""Freeze evidence and safely stop the in-progress official MBench v3 run.

This script is intentionally specific to the v3 scheduler.  It never deletes
results.  A cache file that cannot be loaded after shutdown is moved into the
timestamped stop archive so a future resume cannot mistake it for a valid
SMPLify cache entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np


SCHEDULER_MARKER = "scripts/run_official_mbench_repair_parallel.py"
WORKER_MARKER = "scripts/run_official_mbench_repair_worker.py"
EVALUATOR_MARKER = "evaluate_mbench.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_table() -> dict[int, dict[str, object]]:
    table: dict[int, dict[str, object]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
            status_lines = (entry / "status").read_text(encoding="utf-8").splitlines()
            fields = dict(line.split(":", 1) for line in status_lines if ":" in line)
            table[pid] = {
                "pid": pid,
                "ppid": int(fields["PPid"].strip()),
                "pgid": os.getpgid(pid),
                "command": command,
            }
        except (FileNotFoundError, ProcessLookupError, KeyError, ValueError):
            continue
    return table


def matching_processes(*markers: str) -> list[dict[str, object]]:
    return sorted(
        [row for row in process_table().values() if any(marker in str(row["command"]) for marker in markers)],
        key=lambda row: int(row["pid"]),
    )


def artifact_record(path: Path, base: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "path": str(path.relative_to(base)),
        "size_bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
    }
    # Group outputs are small JSON/log evidence.  Avoid hashing any accidental
    # large binary here; SMPLify caches have their own audited inventory below.
    if path.stat().st_size <= 64 * 1024 * 1024:
        record["sha256"] = sha256(path)
    return record


def inventory_run_records(output_root: Path) -> tuple[list[dict], list[dict]]:
    completed: list[dict] = []
    incomplete: list[dict] = []
    for record_path in sorted(output_root.rglob("run_record.json")):
        try:
            payload = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception as exc:  # preserve malformed evidence in the manifest
            incomplete.append({"path": str(record_path.relative_to(output_root)), "read_error": repr(exc)})
            continue
        entry = {
            "path": str(record_path.relative_to(output_root)),
            "record": payload,
            "artifacts": [
                artifact_record(path, output_root)
                for path in sorted(record_path.parent.rglob("*"))
                if path.is_file()
            ],
        }
        if payload.get("status") == "COMPLETED":
            completed.append(entry)
        else:
            incomplete.append(entry)
    return completed, incomplete


def terminate_exact_groups(processes: list[dict[str, object]], timeout: float) -> dict[str, object]:
    groups = sorted({int(row["pgid"]) for row in processes if int(row["pgid"]) > 1})
    actions: list[dict[str, object]] = []
    for pgid in groups:
        try:
            os.killpg(pgid, signal.SIGTERM)
            actions.append({"pgid": pgid, "signal": "SIGTERM", "at": utc_now()})
        except ProcessLookupError:
            actions.append({"pgid": pgid, "signal": "already_exited", "at": utc_now()})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = matching_processes(WORKER_MARKER, EVALUATOR_MARKER)
        if not remaining:
            return {"actions": actions, "remaining": []}
        time.sleep(1.0)
    remaining = matching_processes(WORKER_MARKER, EVALUATOR_MARKER)
    for pgid in sorted({int(row["pgid"]) for row in remaining if int(row["pgid"]) > 1}):
        try:
            os.killpg(pgid, signal.SIGKILL)
            actions.append({"pgid": pgid, "signal": "SIGKILL_after_timeout", "at": utc_now()})
        except ProcessLookupError:
            pass
    time.sleep(1.0)
    return {"actions": actions, "remaining": matching_processes(WORKER_MARKER, EVALUATOR_MARKER)}


def validate_cache(path: Path) -> tuple[bool, str | None]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"expected dict, got {type(payload).__name__}")
        for key in ("pose", "joints", "vertices"):
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                finite = bool(torch.isfinite(value).all())
            elif isinstance(value, np.ndarray):
                finite = bool(np.isfinite(value).all())
            else:
                raise TypeError(f"missing tensor/array key {key}")
            if not finite:
                raise ValueError(f"non-finite tensor {key}")
        return True, None
    except Exception as exc:
        return False, repr(exc)


def audit_caches(organized_root: Path, archive_root: Path) -> dict[str, object]:
    cache_files = sorted(organized_root.glob("*/seed_*/*/*.pt"))
    records: list[dict[str, object]] = []
    quarantined: list[dict[str, str]] = []
    total = len(cache_files)
    for index, path in enumerate(cache_files, start=1):
        valid, error = validate_cache(path)
        record: dict[str, object] = {
            "path": str(path.relative_to(organized_root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "valid": valid,
        }
        if error is not None:
            record["error"] = error
            destination = archive_root / "quarantined_cache" / path.relative_to(organized_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
            quarantined.append({"source": str(path), "preserved_at": str(destination)})
        records.append(record)
        if index % 50 == 0 or index == total:
            print(f"CACHE_AUDIT {index}/{total} valid={sum(bool(item['valid']) for item in records)}", flush=True)
    return {
        "count_before_audit": total,
        "valid_count": sum(bool(item["valid"]) for item in records),
        "quarantined": quarantined,
        "files": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--organized-root", type=Path, required=True)
    parser.add_argument("--archive-parent", type=Path, required=True)
    parser.add_argument("--expected-scheduler-pid", type=int, required=True)
    parser.add_argument("--term-timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()

    scheduler_state_path = args.output_root / "scheduler_state.json"
    before_state = json.loads(scheduler_state_path.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root = args.archive_parent / f"user_stop_{stamp}"
    archive_root.mkdir(parents=True, exist_ok=False)

    processes_before = matching_processes(SCHEDULER_MARKER, WORKER_MARKER, EVALUATOR_MARKER)
    scheduler_rows = [row for row in processes_before if SCHEDULER_MARKER in str(row["command"])]
    if [int(row["pid"]) for row in scheduler_rows] != [args.expected_scheduler_pid]:
        raise RuntimeError(f"scheduler identity mismatch: {scheduler_rows}")

    completed_before, incomplete_before = inventory_run_records(args.output_root)
    pre_manifest = {
        "protocol": "official_mbench_v3_user_stop_archive_v1",
        "captured_at": utc_now(),
        "scheduler_state": before_state,
        "processes": processes_before,
        "completed_count": len(completed_before),
        "incomplete_count": len(incomplete_before),
        "completed_runs": completed_before,
        "incomplete_runs": incomplete_before,
    }
    atomic_json(archive_root / "pre_stop_manifest.json", pre_manifest)
    atomic_json(archive_root / "scheduler_state_before_stop.json", before_state)
    print(f"ARCHIVED_PRE_STOP completed={len(completed_before)} at={archive_root}", flush=True)

    os.kill(args.expected_scheduler_pid, signal.SIGTERM)
    time.sleep(2.0)
    worker_rows = [row for row in processes_before if WORKER_MARKER in str(row["command"])]
    termination = terminate_exact_groups(worker_rows, args.term_timeout_seconds)
    if matching_processes(SCHEDULER_MARKER):
        raise RuntimeError("scheduler process still exists after SIGTERM")
    if termination["remaining"]:
        raise RuntimeError(f"worker/evaluator processes still exist: {termination['remaining']}")
    print("PROCESSES_STOPPED", flush=True)

    cache_audit = audit_caches(args.organized_root, archive_root)
    atomic_json(archive_root / "cache_manifest.json", cache_audit)
    completed_after, incomplete_after = inventory_run_records(args.output_root)
    final_state = dict(before_state)
    final_state.update(
        {
            "status": "USER_STOPPED_ARCHIVED",
            "heartbeat_at": utc_now(),
            "stopped_by_user_request": True,
            "active_workers": 0,
            "completed_jobs": len(completed_after),
            "archive_manifest": str(archive_root / "post_stop_manifest.json"),
            "valid_smplify_cache_count": cache_audit["valid_count"],
            "quarantined_cache_count": len(cache_audit["quarantined"]),
        }
    )
    atomic_json(scheduler_state_path, final_state)
    post_manifest = {
        "protocol": "official_mbench_v3_user_stop_archive_v1",
        "stopped_at": utc_now(),
        "state_before": before_state,
        "state_after": final_state,
        "termination": termination,
        "processes_after": matching_processes(SCHEDULER_MARKER, WORKER_MARKER, EVALUATOR_MARKER),
        "completed_count": len(completed_after),
        "incomplete_count": len(incomplete_after),
        "completed_runs": completed_after,
        "incomplete_runs": incomplete_after,
        "cache_summary": {
            "count_before_audit": cache_audit["count_before_audit"],
            "valid_count": cache_audit["valid_count"],
            "quarantined": cache_audit["quarantined"],
        },
    }
    atomic_json(archive_root / "post_stop_manifest.json", post_manifest)
    print(
        "USER_STOP_COMPLETE "
        f"completed={len(completed_after)} valid_cache={cache_audit['valid_count']} "
        f"quarantined={len(cache_audit['quarantined'])} archive={archive_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
