#!/usr/bin/env python3
"""Materialise paired-M0 heel/toe, contact, ground, and valid-frame evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.physical_metrics import evaluate_physical_metrics_v2
from evaluation.physical_reference import (
    REFERENCE_CACHE_VERSION,
    materialize_reference,
    reference_contact_masks,
    reference_markers,
)
from motion_rep.pose_authority import authority_project


DEFAULT_SAMPLING_ROOT = ROOT / "results/phase9/pelvis_m1_m7/s0_sampling"
DEFAULT_OUTPUT = ROOT / "results/phase9/pelvis_m1_m7/physical_reference_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _one(root: Path, patterns: tuple[str, ...]) -> Path:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    unique = sorted({path.resolve() for path in paths})
    if len(unique) != 1:
        raise RuntimeError(
            f"expected one of {patterns!r} below {root}, found {len(unique)}"
        )
    return unique[0]


def _discover_source(sampling_root: Path, seed: int) -> Path:
    parent = sampling_root / "M1" / f"seed_{seed:03d}" / "dose_+0"
    for attempt in sorted(parent.glob("attempt_*"), reverse=True):
        if list(attempt.glob("guided_artifacts/batch_000/m0_authority_norm_batch.pt")):
            return attempt
    raise FileNotFoundError(f"no materialisable paired M0 source below {parent}")


def _load_source(seed: int, root: Path) -> dict[str, Any]:
    m0_path = _one(
        root,
        (
            "guided_artifacts/batch_000/m0_authority_norm_batch.pt",
            "guided_artifacts/batch_000/m0_consistent_norm_batch.pt",
        ),
    )
    archive_path = _one(root, ("trainer/**/batch_000/mbench_raw_norm_batch.pt",))
    m0_norm = torch.load(m0_path, map_location="cpu", weights_only=True).float()
    archive = torch.load(archive_path, map_location="cpu", weights_only=True)
    sample_ids = [str(value) for value in archive["sample_ids"]]
    valid = archive["motion_mask"].bool()
    mean = archive["motion_mean"].float()
    std = archive["motion_std"].float()
    physical = authority_project(
        m0_norm,
        valid_mask=valid,
        mean=mean,
        std=std,
        input_standardized=True,
        output_standardized=False,
    ).physical_motion
    if physical.shape[:2] != valid.shape or len(sample_ids) != physical.shape[0]:
        raise ValueError(f"inconsistent paired M0 source below {root}")
    return {
        "seed": int(seed),
        "root": root,
        "m0_path": m0_path,
        "archive_path": archive_path,
        "sample_ids": sample_ids,
        "valid": valid,
        "physical": physical,
    }


def _strict_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(
    *,
    sampling_root: Path,
    output: Path,
    code_commit: str,
    sources: list[tuple[int, Path]] | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite physical reference: {output}")
    selected = sources or [
        (seed, _discover_source(sampling_root, seed)) for seed in (0, 42)
    ]
    loaded = [_load_source(seed, root) for seed, root in selected]
    if sorted(item["seed"] for item in loaded) != [0, 42]:
        raise ValueError("S0 physical reference requires seeds 0 and 42 exactly once")
    frame_counts = {item["physical"].shape[1] for item in loaded}
    if len(frame_counts) != 1:
        raise ValueError("paired M0 sources must share one padded frame count")
    motion = torch.cat([item["physical"] for item in loaded], dim=0)
    valid = torch.cat([item["valid"] for item in loaded], dim=0)
    sample_ids = [sample for item in loaded for sample in item["sample_ids"]]
    seeds = [item["seed"] for item in loaded for _ in item["sample_ids"]]
    reference = materialize_reference(
        motion,
        valid,
        sample_ids=sample_ids,
        seeds=seeds,
        code_commit=code_commit,
    )

    self_rows = []
    for index, paired_id in enumerate(reference["paired_m0_ids"]):
        baseline = reference_markers(reference, index)
        contacts, pairs = reference_contact_masks(reference, index)
        result = evaluate_physical_metrics_v2(
            baseline,
            baseline,
            reference["valid_frame_mask"][index : index + 1],
            contacts,
            reference["ground_height_m"][index : index + 1],
            contact_pair_masks=pairs,
        )
        row = dict(result["per_sequence"][0])
        row.update(
            {
                "paired_m0_id": paired_id,
                "seed": reference["seeds"][index],
                "sample_id": reference["sample_ids"][index],
            }
        )
        self_rows.append(row)

    output.mkdir(parents=True)
    cache_path = output / "physical_reference.pt"
    torch.save(reference, cache_path)
    manifest = {
        "protocol": REFERENCE_CACHE_VERSION,
        "status": "PHYSICAL_REFERENCE_MATERIALIZED",
        "code_commit": str(code_commit),
        "sequence_count": len(reference["sample_ids"]),
        "seeds": reference["seeds"],
        "sample_ids": reference["sample_ids"],
        "paired_m0_ids": reference["paired_m0_ids"],
        "unit": reference["unit"],
        "world_up_axis": reference["world_up_axis"],
        "marker_joints": reference["marker_joints"],
        "authority_version": reference["authority_version"],
        "fk_version": reference["fk_version"],
        "skeleton_source": reference["skeleton_source"],
        "contact_version": reference["contact_version"],
        "ground_version": reference["ground_version"],
        "cache_file": {
            "path": cache_path.name,
            "sha256": sha256(cache_path),
            "bytes": cache_path.stat().st_size,
        },
        "sources": [
            {
                "seed": item["seed"],
                "root": str(item["root"]),
                "m0_file": str(item["m0_path"]),
                "m0_sha256": sha256(item["m0_path"]),
                "archive_file": str(item["archive_path"]),
                "archive_sha256": sha256(item["archive_path"]),
            }
            for item in loaded
        ],
        "m0_self_evaluation": {
            "status": "RAW_BASELINE_METRICS_THRESHOLDS_NOT_FROZEN",
            "rows": self_rows,
        },
        "invariants": {
            "reference_source": "PAIRED_M0_ONLY",
            "candidate_contact_reclassification": "FORBIDDEN",
            "candidate_ground_reestimation": "FORBIDDEN",
            "candidate_marker_redefinition": "FORBIDDEN",
        },
    }
    _strict_json(output / "manifest.json", manifest)
    _strict_json(output / "m0_self_evaluation.json", manifest["m0_self_evaluation"])
    return manifest


def _source(value: str) -> tuple[int, Path]:
    seed, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("source must be SEED=RUN_ROOT")
    return int(seed), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-root", type=Path, default=DEFAULT_SAMPLING_ROOT)
    parser.add_argument("--source", action="append", type=_source)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    manifest = run(
        sampling_root=args.sampling_root,
        sources=args.source,
        output=args.output,
        code_commit=args.code_commit,
    )
    print(json.dumps({"status": manifest["status"], "sequence_count": manifest["sequence_count"]}))


if __name__ == "__main__":
    main()
