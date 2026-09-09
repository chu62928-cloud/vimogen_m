#!/usr/bin/env python3
"""Run the M7 slice of S0 from two verified, paired M0 cache roots.

This runner does not invoke the generator: M7 is intentionally the
post-generation geometric-edit baseline.  It consumes each seed's M0,
normalisation statistics, valid mask, and source noise from one cache root,
then writes one strict record per sample and dose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.content_metrics import evaluate_content_metrics
from evaluation.control_metrics import evaluate_control_metrics
from geometry.authoritative_motion import authoritative_motion
from geometry.pelvis_angle import target_angle_curve_deg
from guidance.base import (
    ConstraintPack,
    GuidanceRequest,
    SharedEvidence,
    mapping_sha256,
    tensor_sha256,
    write_run_record,
)
from guidance.m7_paht_edit import M7PAHTGeometricEdit, PROTOCOL_NAME


EVALUATOR_VERSION = "vimogen_m1_m7_evaluator_v1"


def _source_argument(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("source must be SEED=RUN_ROOT")
    return int(seed_text), Path(path_text)


def _one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern!r} below {root}, found {len(matches)}")
    return matches[0]


def _load_source(seed: int, root: Path) -> dict:
    m0_path = _one(root, "guided_artifacts/batch_000/m0_consistent_norm_batch.pt")
    noise_path = _one(root, "m0_artifacts/batch_000/z0_replayed.pt")
    archive_path = _one(root, "trainer/**/batch_000/mbench_raw_norm_batch.pt")
    m0_norm = torch.load(m0_path, map_location="cpu", weights_only=True).float()
    noise = torch.load(noise_path, map_location="cpu", weights_only=True).float()
    archive = torch.load(archive_path, map_location="cpu", weights_only=True)
    sample_ids = [str(value) for value in archive["sample_ids"]]
    valid = archive["motion_mask"].bool()
    mean = archive["motion_mean"].float()
    std = archive["motion_std"].float()
    if mean.ndim == 2:
        mean = mean[:, None, :]
    if std.ndim == 2:
        std = std[:, None, :]
    physical = m0_norm * mean.new_ones(m0_norm.shape) * std + mean
    physical = authoritative_motion(physical, valid_mask=valid).motion
    if physical.shape != noise.shape or physical.shape[:2] != valid.shape:
        raise ValueError(f"cache tensor shapes are inconsistent below {root}")
    if sample_ids != ["94", "34122"]:
        raise ValueError(f"unexpected S0 sample order: {sample_ids}")
    return {
        "seed": seed,
        "root": root,
        "m0": physical,
        "noise": noise,
        "valid": valid,
        "sample_ids": sample_ids,
        "m0_path": m0_path,
        "archive_path": archive_path,
    }


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    sources = [_load_source(seed, root) for seed, root in args.source]
    if sorted(item["seed"] for item in sources) != [0, 42]:
        raise ValueError("M7 S0 requires exactly seeds 0 and 42")
    doses = (-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0)
    method_config: dict[str, float] = {"eps": 1.0e-6}
    rows = []
    method = M7PAHTGeometricEdit()
    for source in sources:
        for index, sample_id in enumerate(source["sample_ids"]):
            baseline = source["m0"][index : index + 1]
            noise = source["noise"][index : index + 1]
            valid = source["valid"][index : index + 1]
            baseline_id = f"seed{source['seed']}_sample{sample_id}_{tensor_sha256(baseline)[:16]}"
            for dose in doses:
                target = target_angle_curve_deg(baseline, dose)
                request = GuidanceRequest(
                    prompt_id=sample_id,
                    seed=source["seed"],
                    target_dose_deg=dose,
                    constraint_pack=ConstraintPack.C0,
                    base_noise=noise,
                    baseline_motion=baseline,
                    shared_evidence=SharedEvidence(
                        valid_mask=valid,
                        target_angle_curve_deg=target,
                        contact_evidence_version=f"m0:{baseline_id}:not_materialized",
                        ground_version=f"m0:{baseline_id}:not_materialized",
                        metadata={"source_cache_root": str(source["root"])},
                    ),
                    run_id=f"M7_s{source['seed']}_p{sample_id}_d{dose:+g}",
                    baseline_motion_id=baseline_id,
                )
                result = method.run(None, request, method_config)
                metrics = {
                    "control": evaluate_control_metrics(
                        result.motion, baseline, target, valid, target_dose_deg=dose
                    ),
                    "content": evaluate_content_metrics(result.motion, baseline, valid),
                    "physical": {
                        "status": "PENDING_SHARED_FK_MARKER_MATERIALIZATION",
                        "reason": "cache contains 276D motion but no frozen heel/toe marker bundle",
                    },
                }
                run_dir = args.output / request.run_id
                run_dir.mkdir()
                torch.save(result.motion.detach().cpu(), run_dir / "motion_physical.pt")
                record = {
                    "run_id": request.run_id,
                    "code_commit": args.code_commit,
                    "vimogen_checkpoint_hash": args.checkpoint_hash,
                    "evaluator_version": EVALUATOR_VERSION,
                    "method_name": "M7",
                    "method_config_hash": mapping_sha256(method_config),
                    "prompt_id": sample_id,
                    "seed": source["seed"],
                    "target_dose_deg": dose,
                    "constraint_pack": ConstraintPack.C0,
                    "baseline_motion_id": baseline_id,
                    "contact_evidence_version": request.shared_evidence.contact_evidence_version,
                    "ground_version": request.shared_evidence.ground_version,
                    "status": result.status,
                    "failure_reason": result.diagnostics_dict()["failure_reason"],
                    "all_metrics": metrics,
                    "all_diagnostics": result.diagnostics_dict(),
                    "source_m0_path": str(source["m0_path"]),
                    "source_archive_path": str(source["archive_path"]),
                    "source_m0_sha256": tensor_sha256(baseline),
                    "protocol": PROTOCOL_NAME,
                }
                write_run_record(run_dir / "run_record.json", record)
                rows.append(record)
    summary = {
        "status": "COMPLETED" if all(row["status"] == "COMPLETED" for row in rows) else "FAILED",
        "protocol": PROTOCOL_NAME,
        "scope": "S1_M7_GENERATED_POST_EDIT_REFERENCE",
        "expected_runs": 28,
        "completed_runs": sum(row["status"] == "COMPLETED" for row in rows),
        "angle_gate_passes": sum(
            row["all_metrics"]["control"]["summary"]["sequence_angle_pass_rate"] == 1.0
            for row in rows
        ),
        "records": [str(args.output / row["run_id"] / "run_record.json") for row in rows],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=_source_argument, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
