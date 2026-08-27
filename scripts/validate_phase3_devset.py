#!/usr/bin/env python3
"""Validate a phase-3 development manifest and build a non-executable matrix.

The matrix is a protocol artifact.  It does not launch ViMoGen and it refuses
to become formal until the manifest status is FROZEN and all review fields are
complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


METHODS = ["M0_official", "B0", "B1", "B2", "M1"]
COMMANDS = [0.0, 5.0, 10.0]
SEEDS = [0, 1, 2]
FROZEN_STATUSES = {"FROZEN", "FROZEN_SINGLE_REVIEW_OVERRIDE"}


def validate_manifest(manifest: dict[str, Any], require_frozen: bool = False) -> list[str]:
    errors: list[str] = []
    status = manifest.get("status")
    if status not in {"CANDIDATE_NOT_FROZEN", *FROZEN_STATUSES}:
        errors.append(f"invalid status: {status!r}")
    if require_frozen and status not in FROZEN_STATUSES:
        errors.append("formal matrix requires a frozen manifest status")
    categories = manifest.get("categories", [])
    per_category = manifest.get("items_per_category")
    items = manifest.get("items", [])
    if not categories or not isinstance(per_category, int):
        errors.append("missing categories/items_per_category")
        return errors
    if len(items) != len(categories) * per_category:
        errors.append("item count does not match category target")
    ids = [item.get("id") for item in items]
    texts = [item.get("normalised_text") for item in items]
    if len(set(ids)) != len(ids):
        errors.append("duplicate motion ids")
    if len(set(texts)) != len(texts):
        errors.append("duplicate normalized texts")
    for category in categories:
        count = sum(item.get("category") == category for item in items)
        if count != per_category:
            errors.append(f"category {category!r} has {count}, expected {per_category}")
    if status in FROZEN_STATUSES:
        for item in items:
            review = item.get("manual_review", {})
            if any(review.get(key) in {None, "", "PENDING"} for key in ("reviewer_a", "reviewer_b", "adjudication")):
                errors.append(f"incomplete review for id={item.get('id')}")
    return errors


def build_matrix(manifest: dict[str, Any]) -> dict[str, Any]:
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid manifest: " + "; ".join(errors))
    frozen = manifest.get("status") in FROZEN_STATUSES
    rows = []
    for item in manifest["items"]:
        for seed in SEEDS:
            for command in COMMANDS:
                rows.append(
                    {
                        "dev_index": item["dev_index"],
                        "motion_id": item["id"],
                        "category": item["category"],
                        "seed": seed,
                        "command_delta_deg": command,
                        "methods": METHODS,
                        "noise_key": {
                            "sample_id": item["id"],
                            "seed": seed,
                            "shape": [100, 276],
                            "dtype": "bfloat16",
                            "protocol_version": "vimogen-sample-noise-v1",
                        },
                        "execution_status": "READY" if frozen else "PREVIEW_ONLY",
                    }
                )
    return {
        "schema_version": "phase3_dev_matrix_v1",
        "status": "FROZEN_MATRIX_READY" if frozen else "PREVIEW_NOT_FORMAL",
        "manifest_status": manifest["status"],
        "target_units": len(rows),
        "methods": METHODS,
        "commands_degrees": COMMANDS,
        "seeds": SEEDS,
        "same_initial_noise_required": True,
        "do_not_run_until_manifest_frozen": not frozen,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = validate_manifest(manifest, require_frozen=args.require_frozen)
    if errors:
        raise SystemExit("manifest validation failed: " + "; ".join(errors))
    matrix = build_matrix(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing matrix: {args.output}")
    args.output.write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: matrix[key] for key in ("status", "target_units", "do_not_run_until_manifest_frozen")}, indent=2))


if __name__ == "__main__":
    main()
