#!/usr/bin/env python3
"""Materialize per-prompt physical 276D files from generation archives.

The no-video generation path writes one normalized batch archive to avoid
per-sample GPU/CPU synchronization.  This offline step applies the frozen
normalization once and writes the directory layout consumed by the three-way
MBench organizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def materialize_run(input_run: Path, output_run: Path, expected_count: int) -> dict:
    archives = sorted(input_run.glob("trainer/test_visualization/**/mbench_raw_norm_batch.pt"))
    if not archives:
        raise RuntimeError(f"{input_run}: no batch archives found")
    if output_run.exists() and any(output_run.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_run}")
    output_run.mkdir(parents=True, exist_ok=True)
    all_sample_ids = []
    all_condition_names = []
    for archive_path in archives:
        archive = torch.load(archive_path, map_location="cpu", weights_only=False)
        motion_norm = archive["motion_norm"].float()
        mask = archive["motion_mask"].bool()
        mean = archive["motion_mean"].float()
        std = archive["motion_std"].float()
        sample_ids = [str(value) for value in archive["sample_ids"]]
        condition_names = [str(value) for value in archive["condition_names"]]
        if len(sample_ids) != len(condition_names) or motion_norm.shape[0] != len(sample_ids):
            raise ValueError(f"{archive_path}: archive batch shape does not match metadata")
        all_sample_ids.extend(sample_ids)
        all_condition_names.extend(condition_names)
        step_name = "step00000001"
        for index, sample_id in enumerate(sample_ids):
            valid = mask[index]
            physical = motion_norm[index:index + 1, valid] * std + mean
            sample_dir = output_run / "trainer" / "test_visualization" / "materialized_mbench" / step_name / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            torch.save(physical[0], sample_dir / "motion_gen_condition_on_text.pt")
    if len(all_sample_ids) != expected_count:
        raise ValueError(f"{input_run}: expected {expected_count} sample ids, found {len(all_sample_ids)}")
    if len(set(all_sample_ids)) != len(all_sample_ids):
        raise ValueError(f"{input_run}: duplicate sample ids across batch archives")
    manifest = {
        "status": "VALID",
        "protocol": "vimogen_publication_mbench_physical_materialization_v1",
        "input_run": str(input_run),
        "source_archives": [str(path) for path in archives],
        "sample_count": len(all_sample_ids),
        "sample_ids": all_sample_ids,
        "condition_names": all_condition_names,
        "normalization": "physical = normalized * motion_std + motion_mean",
    }
    (output_run / "materialization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=450)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for condition_dir in sorted(path for path in args.input_root.iterdir() if path.is_dir()):
        for seed_dir in sorted(path for path in condition_dir.iterdir() if path.is_dir()):
            input_run = seed_dir
            output_run = args.output_root / condition_dir.name / seed_dir.name
            manifests.append(materialize_run(input_run, output_run, args.expected_count))
    if len(manifests) != 9:
        raise RuntimeError(f"expected 9 generation runs, found {len(manifests)}")
    print(json.dumps({"status": "VALID", "run_count": len(manifests)}, indent=2))


if __name__ == "__main__":
    main()
