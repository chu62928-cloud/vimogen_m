#!/usr/bin/env python3
"""Build a pure-text MBench input list from a frozen phase-3 manifest.

The ViMoGen evaluation loader uses ``MBenchWiRefMotion``.  Entries without
``use_ref_motion`` are text-conditioned and only need a fixed-shape placeholder
motion for batching; the placeholder is never used as a reference condition.
This script creates a new input file and never edits the frozen manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FROZEN_STATUSES = {"FROZEN", "FROZEN_SINGLE_REVIEW_OVERRIDE"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_input(manifest_path: Path, source_path: Path, output_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in FROZEN_STATUSES:
        raise ValueError("input generation requires a frozen manifest status")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_by_id = {int(item["id"]): item for item in source}
    rows = []
    for item in manifest["items"]:
        motion_id = int(item["id"])
        if motion_id not in source_by_id:
            raise KeyError(f"frozen id={motion_id} is absent from source JSON")
        rows.append(
            {
                "global_id": str(motion_id),
                "sample_id": str(motion_id),
                "prompt": item["motion_text_annot"],
                "motion_text_annot": item["motion_text_annot"],
                "category": item["category"],
                "dev_index": int(item["dev_index"]),
                "motion_path": "./data_samples/dummy_motion.pt",
                "source_motion_path": source_by_id[motion_id].get("motion_path"),
                "use_ref_motion": False,
            }
        )
    if len(rows) != 20 or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("frozen input must contain 20 unique sample IDs")
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": "phase3_devset_text_input_v1",
        "status": "READY_TEXT_ONLY",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "input": str(output_path),
        "input_sha256": _sha256(output_path),
        "target_size": len(rows),
        "condition": "text_only",
        "reference_motion_used": False,
        "placeholder_motion": "./data_samples/dummy_motion.pt (batch shape only; never used as ref condition)",
        "sample_id_rule": "string form of frozen manifest motion id",
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_input(args.manifest, args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
