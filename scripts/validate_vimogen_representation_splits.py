#!/usr/bin/env python3
"""Validate frozen ViMoGen representation manifests and motion fingerprints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.vimogen_representation_protocol import (  # noqa: E402
    materialize_manifest,
    validate_materialized_overlap,
    validate_split_manifests,
)


NAMES = ("representation_dev_v1", "representation_val_v1", "representation_test_v1")


def validate(
    manifest_dir: Path,
    *,
    motion_root: Path | None = None,
    mbench_root: Path | None = None,
    require_frozen: bool = True,
) -> dict:
    manifests = {
        name: json.loads((manifest_dir / f"{name}.json").read_text(encoding="utf-8"))
        for name in NAMES
    }
    summary = validate_split_manifests(manifests, require_frozen=require_frozen)
    if motion_root is not None:
        materialized = {
            name: materialize_manifest(
                manifest,
                motion_root=motion_root,
                mbench_root=mbench_root,
                drop_duplicates=True,
            )
            for name, manifest in manifests.items()
        }
        summary["materialized_counts"] = {
            name: len(manifest["items"]) for name, manifest in materialized.items()
        }
        summary["fingerprinted"] = True
        summary["materialized_overlap"] = validate_materialized_overlap(materialized)
    else:
        summary["fingerprinted"] = False
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--motion-root", type=Path)
    parser.add_argument("--mbench-root", type=Path)
    parser.add_argument("--allow-candidate", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = validate(
        args.manifest_dir,
        motion_root=args.motion_root,
        mbench_root=args.mbench_root,
        require_frozen=not args.allow_candidate,
    )
    payload = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
