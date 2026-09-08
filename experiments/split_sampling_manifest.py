#!/usr/bin/env python3
"""Split a frozen multi-sample JSON list into auditable singleton manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_id(entry: dict[str, Any]) -> str:
    values = {
        str(entry[key])
        for key in ("id", "sample_id", "global_id")
        if entry.get(key) is not None
    }
    if not values:
        raise ValueError("manifest entry has no id, sample_id, or global_id")
    if len(values) != 1:
        raise ValueError(f"manifest entry has conflicting IDs: {sorted(values)}")
    return values.pop()


def split_manifest(source: Path, output: Path, requested_ids: list[str]) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite singleton manifests: {output}")
    entries = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise TypeError("source manifest must be a JSON list")
    indexed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("every source manifest entry must be an object")
        identifier = sample_id(entry)
        if identifier in indexed:
            raise ValueError(f"duplicate sample ID {identifier}")
        indexed[identifier] = entry
    requested = [str(value) for value in requested_ids]
    if len(set(requested)) != len(requested):
        raise ValueError("requested sample IDs must be unique")
    missing = sorted(set(requested) - set(indexed))
    if missing:
        raise KeyError(f"requested sample IDs are absent: {missing}")

    output.mkdir(parents=True)
    rows = []
    for identifier in requested:
        path = output / f"sample_{identifier}.json"
        path.write_text(
            json.dumps([indexed[identifier]], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "sample_id": identifier,
                "path": str(path),
                "sha256": sha256(path),
            }
        )
    summary = {
        "protocol": "vimogen_sampling_singleton_manifest_split_v1",
        "status": "FROZEN_SINGLETON_MANIFESTS",
        "source": str(source),
        "source_sha256": sha256(source),
        "source_entry_count": len(entries),
        "requested_sample_ids": requested,
        "manifests": rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            split_manifest(args.source, args.output, args.sample_id),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
