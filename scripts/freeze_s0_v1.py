#!/usr/bin/env python3
"""Freeze the complete 84-sequence S0-v1 baseline without copying results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


EXPECTED_METHODS = tuple(f"M{index}" for index in range(1, 8))
EXPECTED_SEEDS = (0, 42)
EXPECTED_DOSES = (-2.0, 0.0, 2.0)
EXPECTED_PROMPTS = ("94", "34122")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_key(record: dict[str, Any]) -> tuple[str, int, float, str]:
    return (
        str(record["method_name"]),
        int(record["seed"]),
        float(record["target_dose_deg"]),
        str(record["prompt_id"]),
    )


def _is_sequence_record(value: Any) -> bool:
    return isinstance(value, dict) and {
        "run_id",
        "method_name",
        "prompt_id",
        "seed",
        "target_dose_deg",
        "baseline_motion_id",
        "evaluator_version",
        "all_metrics",
    }.issubset(value)


def _records_from_json(path: Path) -> Iterable[tuple[dict[str, Any], Path]]:
    value = _json(path)
    if _is_sequence_record(value):
        yield value, path
        return
    values: Sequence[Any]
    if isinstance(value, list):
        values = value
    elif isinstance(value, dict) and isinstance(value.get("records"), list):
        values = value["records"]
    else:
        return
    for item in values:
        if _is_sequence_record(item):
            yield item, path
        elif isinstance(item, str):
            target = Path(item)
            if not target.is_absolute():
                target = path.parent / target
            if target.is_file():
                target_value = _json(target)
                if _is_sequence_record(target_value):
                    yield target_value, target


def _resolve_record_path(path_text: str, anchor: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = (anchor / path, Path.cwd() / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _canonical_record_paths(source: Path) -> list[Path] | None:
    progress_path = source / "s0_matrix_progress.json"
    if progress_path.is_file():
        progress = _json(progress_path)
        latest: dict[tuple[str, int, float], dict[str, Any]] = {}
        for item in progress.get("jobs", []):
            key = (str(item["method"]), int(item["seed"]), float(item["dose"]))
            latest[key] = item
        paths: list[Path] = []
        for key in sorted(latest):
            evaluation = latest[key].get("evaluation", {})
            if evaluation.get("status") in {"EVALUATION_FAILED", "EVALUATION_OUTPUT_INVALID"}:
                raise RuntimeError(f"unrepaired canonical evaluation remains for {key}")
            values = evaluation.get("records", [])
            if len(values) != len(EXPECTED_PROMPTS):
                raise RuntimeError(f"canonical S0 batch {key} has {len(values)} records")
            paths.extend(_resolve_record_path(str(value), source) for value in values)
        return paths
    summary_path = source / "summary.json"
    if summary_path.is_file():
        summary = _json(summary_path)
        values = summary.get("records")
        if isinstance(values, list) and values:
            return [_resolve_record_path(str(value), source) for value in values]
    return None


def collect_sequence_records(
    source_paths: Sequence[Path],
) -> dict[tuple[str, int, float, str], tuple[dict[str, Any], Path]]:
    records: dict[tuple[str, int, float, str], tuple[dict[str, Any], Path]] = {}
    duplicates: list[tuple[str, int, float, str]] = []
    for source in source_paths:
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(source)
        canonical = None if source.is_file() else _canonical_record_paths(source)
        candidates = (
            [source]
            if source.is_file()
            else canonical
            if canonical is not None
            else sorted(source.rglob("*.json"))
        )
        for path in candidates:
            if not path.is_file():
                raise FileNotFoundError(path)
            try:
                found = list(_records_from_json(path))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for record, record_path in found:
                key = _record_key(record)
                if key in records and records[key][1].resolve() != record_path.resolve():
                    duplicates.append(key)
                records[key] = (record, record_path)
    if duplicates:
        raise RuntimeError(f"duplicate S0 sequence records: {sorted(set(duplicates))}")
    return records


def _expected_keys() -> set[tuple[str, int, float, str]]:
    return {
        (method, seed, dose, prompt)
        for method in EXPECTED_METHODS
        for seed in EXPECTED_SEEDS
        for dose in EXPECTED_DOSES
        for prompt in EXPECTED_PROMPTS
    }


def _related_batch_record(record_path: Path) -> dict[str, Any]:
    for parent in record_path.parents:
        candidate = parent / "run_record.json"
        if candidate == record_path or not candidate.is_file():
            continue
        try:
            value = _json(candidate)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and "noise_cache" in value:
            return value
    return {}


def _snapshot(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def freeze_s0_manifest(
    *,
    source_paths: Sequence[Path],
    output: Path,
    code_commit: str,
    snapshot_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen S0 baseline: {output}")
    records = collect_sequence_records(source_paths)
    expected = _expected_keys()
    actual = set(records)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            f"S0 matrix is not exactly 84 unique records; missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}, count={len(actual)}"
        )

    rows: list[dict[str, Any]] = []
    for key in sorted(records):
        record, record_path = records[key]
        batch_record = _related_batch_record(record_path)
        motion_path = record_path.parent / "motion_physical.pt"
        row = {
            "method": key[0],
            "seed": key[1],
            "dose_deg": key[2],
            "prompt_id": key[3],
            "run_id": str(record["run_id"]),
            "code_commit": str(record.get("code_commit", "not_recorded")),
            "evaluator_version": str(record["evaluator_version"]),
            "method_config_hash": record.get("method_config_hash"),
            "baseline_motion_id": str(record["baseline_motion_id"]),
            "noise_cache_id": record.get("noise_cache_id", batch_record.get("noise_cache", "not_recorded")),
            "checkpoint_hash": record.get("vimogen_checkpoint_hash", batch_record.get("checkpoint_hash", "not_recorded")),
            "source_manifest_sha256": batch_record.get("manifest_sha256"),
            "source_record": _snapshot(record_path),
            "motion_output": _snapshot(motion_path) if motion_path.is_file() else None,
            "angle_status": str(record.get("status", "not_recorded")),
            "physical_status": str(record["all_metrics"].get("physical", {}).get("status", "NOT_EVALUATED")),
        }
        rows.append(row)

    snapshots: list[dict[str, Any]] = []
    for source in snapshot_paths:
        source = Path(source)
        if source.is_file():
            snapshots.append(_snapshot(source))
        elif source.is_dir():
            snapshots.extend(_snapshot(path) for path in sorted(source.rglob("*")) if path.is_file())
        else:
            raise FileNotFoundError(source)
    manifest = {
        "protocol": "vimogen_m1_m7_s0_v1_freeze",
        "status": "S0_V1_FROZEN",
        "code_commit": str(code_commit),
        "sequence_count": len(rows),
        "expected_matrix": {
            "methods": list(EXPECTED_METHODS),
            "seeds": list(EXPECTED_SEEDS),
            "doses_deg": list(EXPECTED_DOSES),
            "prompt_ids": list(EXPECTED_PROMPTS),
        },
        "records": rows,
        "snapshots": snapshots,
        "invariants": {
            "s0_v1_read_only": True,
            "failed_attempts_preserved": True,
            "candidate_outputs_not_regenerated_for_physical_evaluation": True,
        },
    }
    output.mkdir(parents=True)
    destination = output / "manifest.json"
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--snapshot", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    manifest = freeze_s0_manifest(
        source_paths=args.source,
        snapshot_paths=args.snapshot,
        output=args.output,
        code_commit=args.code_commit,
    )
    print(json.dumps({"status": manifest["status"], "sequence_count": manifest["sequence_count"]}))


if __name__ == "__main__":
    main()
