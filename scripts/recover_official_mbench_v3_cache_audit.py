#!/usr/bin/env python3
"""Correct the first v3 stop audit's tensor/NumPy type misclassification."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


EXPECTED_TRAILING_SHAPES = {
    "pose": (24, 3),
    "joints": (22, 3),
    "vertices": (6890, 3),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate(path: Path) -> tuple[bool, str | None]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"expected dict, got {type(payload).__name__}")
        frame_count = None
        for key, trailing_shape in EXPECTED_TRAILING_SHAPES.items():
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                actual_shape = tuple(value.shape)
                finite = bool(torch.isfinite(value).all())
            elif isinstance(value, np.ndarray):
                actual_shape = tuple(value.shape)
                finite = bool(np.isfinite(value).all())
            else:
                raise TypeError(f"{key}: expected tensor or ndarray, got {type(value).__name__}")
            if len(actual_shape) != 3 or actual_shape[1:] != trailing_shape:
                raise ValueError(f"{key}: expected trailing shape {trailing_shape}, got {actual_shape}")
            if frame_count is None:
                frame_count = actual_shape[0]
            elif actual_shape[0] != frame_count:
                raise ValueError(f"{key}: frame count {actual_shape[0]} differs from {frame_count}")
            if not finite:
                raise ValueError(f"{key}: non-finite values")
        return True, None
    except Exception as exc:
        return False, repr(exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--organized-root", type=Path, required=True)
    parser.add_argument("--scheduler-state", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.archive_root / "cache_manifest.json").read_text(encoding="utf-8"))
    expected = {entry["path"]: entry for entry in original["files"]}
    restored: list[dict[str, object]] = []
    already_restored: list[dict[str, object]] = []
    invalid: list[dict[str, object]] = []
    for index, (relative, record) in enumerate(sorted(expected.items()), start=1):
        source = args.archive_root / "quarantined_cache" / relative
        destination = args.organized_root / relative
        if not source.is_file():
            if destination.is_file() and sha256(destination) == record["sha256"]:
                valid, error = validate(destination)
                if valid:
                    already_restored.append({"path": relative, "sha256": record["sha256"]})
                    continue
                invalid.append({"path": relative, "error": error, "location": "organized_root"})
                continue
            invalid.append({"path": relative, "error": "quarantined source missing"})
            continue
        actual_hash = sha256(source)
        if actual_hash != record["sha256"]:
            invalid.append({"path": relative, "error": "SHA256 changed after quarantine", "sha256": actual_hash})
            continue
        valid, error = validate(source)
        if not valid:
            invalid.append({"path": relative, "error": error, "sha256": actual_hash})
            continue
        if destination.exists():
            invalid.append({"path": relative, "error": "destination unexpectedly exists", "sha256": actual_hash})
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        restored.append({"path": relative, "sha256": actual_hash, "size_bytes": destination.stat().st_size})
        if index % 50 == 0 or index == len(expected):
            print(f"RECOVERY_AUDIT {index}/{len(expected)} restored_now={len(restored)} already={len(already_restored)} invalid={len(invalid)}", flush=True)

    correction = {
        "protocol": "official_mbench_v3_cache_audit_correction_v2",
        "corrected_at": datetime.now(timezone.utc).isoformat(),
        "reason": "The first audit required tensors and the first correction also assumed 100 frames. Official joints are NumPy arrays and valid motions have variable frame counts.",
        "original_audit_preserved_at": str(args.archive_root / "cache_manifest.json"),
        "expected_count": len(expected),
        "restored_count": len(restored) + len(already_restored),
        "restored_now_count": len(restored),
        "already_restored_count": len(already_restored),
        "invalid_count": len(invalid),
        "restored": restored,
        "already_restored": already_restored,
        "invalid": invalid,
    }
    correction_path = args.archive_root / "cache_audit_correction_v2.json"
    atomic_json(correction_path, correction)
    state = json.loads(args.scheduler_state.read_text(encoding="utf-8"))
    state.update(
        {
            "valid_smplify_cache_count": len(restored) + len(already_restored),
            "quarantined_cache_count": len(invalid),
            "cache_audit_correction": str(correction_path),
        }
    )
    atomic_json(args.scheduler_state, state)
    print(
        f"CACHE_AUDIT_CORRECTED valid={len(restored) + len(already_restored)} restored_now={len(restored)} invalid={len(invalid)} correction={correction_path}",
        flush=True,
    )
    return 0 if not invalid else 2


if __name__ == "__main__":
    raise SystemExit(main())
