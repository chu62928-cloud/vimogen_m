#!/usr/bin/env python3
"""Build and optionally freeze the source-disjoint ViMoGen 276D protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.vimogen_representation_protocol import (  # noqa: E402
    build_split_manifests,
    build_unique_dev_manifest,
    fingerprint_motion,
    freeze_manifests,
    materialize_manifest,
    validate_materialized_overlap,
    write_manifests,
)


def _load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"expected a list in {path}")
    return payload


def _scan_frame_counts(items: list[dict[str, Any]], motion_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for index, item in enumerate(items, start=1):
        fingerprint = fingerprint_motion(motion_root / str(item["motion_path"]))
        counts[str(item["id"])] = fingerprint.frame_count
        if index % 1000 == 0:
            print(f"scanned {index}/{len(items)} motion files", flush=True)
    return counts


def build_protocol(
    *,
    source_json: Path,
    motion_root: Path,
    output_dir: Path,
    mbench_root: Path | None,
    dev_size: int,
    seed: int,
    scan_all_lengths: bool,
    materialize_selected: bool,
    freeze: bool,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    items = _load_items(source_json)
    frame_counts = _scan_frame_counts(items, motion_root) if scan_all_lengths else None
    manifests = build_split_manifests(items, dev_size=dev_size, seed=seed, frame_counts=frame_counts)
    if materialize_selected:
        manifests["representation_dev_v1"] = build_unique_dev_manifest(
            items,
            dev_size=dev_size,
            seed=seed,
            motion_root=motion_root,
            mbench_root=mbench_root,
        )
        manifests["representation_val_v1"] = materialize_manifest(
            manifests["representation_val_v1"],
            motion_root=motion_root,
            mbench_root=mbench_root,
            drop_duplicates=True,
        )
        manifests["representation_test_v1"] = materialize_manifest(
            manifests["representation_test_v1"],
            motion_root=motion_root,
            mbench_root=mbench_root,
            drop_duplicates=True,
        )
        validate_materialized_overlap(manifests)
    if freeze:
        manifests = freeze_manifests(manifests, source_json=source_json, seed=seed)
    write_manifests(manifests, output_dir)
    summary = {
        "status": "FROZEN" if freeze else "CANDIDATE_NOT_FROZEN",
        "output_dir": str(output_dir),
        "source_json": str(source_json),
        "source_json_sha256": __import__("hashlib").sha256(source_json.read_bytes()).hexdigest(),
        "selection_seed": seed,
        "scan_all_lengths": scan_all_lengths,
        "materialize_selected": materialize_selected,
        "counts": {name: len(manifest["items"]) for name, manifest in manifests.items()},
        "excluded_duplicate_counts": {
            name: len(manifest.get("excluded_duplicate_rows", [])) for name, manifest in manifests.items()
        },
    }
    (output_dir / "protocol_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme = [
        "# ViMoGen 276D representation protocol",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This protocol evaluates representation recovery, not whether the released ViMoGen model saw a motion during training.",
        "The source annotation uses `split` as the original dataset name, so the project creates its own source-disjoint protocol.",
        "",
        "## Effective sample counts",
        "",
    ]
    for name in ("representation_dev_v1", "representation_val_v1", "representation_test_v1"):
        readme.append(
            f"- `{name}`: {summary['counts'][name]} active tensors; "
            f"{summary['excluded_duplicate_counts'][name]} exact duplicates recorded and excluded from effective evaluation."
        )
    readme.extend(
        [
            "",
            "The final test split is the active, unique tensor set from FIT3D, Mixamo, HumanSC3D, ARCTIC, RICH and EMDB.",
            "No final-test result may be used to tune weights, smoothing, thresholds or fallback rules.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mbench-root", type=Path)
    parser.add_argument("--dev-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--scan-all-lengths", action="store_true")
    parser.add_argument("--no-materialize-selected", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    summary = build_protocol(
        source_json=args.source_json,
        motion_root=args.motion_root,
        output_dir=args.output_dir,
        mbench_root=args.mbench_root,
        dev_size=args.dev_size,
        seed=args.seed,
        scan_all_lengths=args.scan_all_lengths,
        materialize_selected=not args.no_materialize_selected,
        freeze=args.freeze,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
