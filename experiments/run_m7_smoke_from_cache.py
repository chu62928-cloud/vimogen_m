#!/usr/bin/env python3
"""Run M7 from archived caches or a frozen candidate M0 reference.

This runner does not invoke the generator: M7 is intentionally the
post-generation geometric-edit baseline.  It consumes each seed's M0,
normalisation statistics, valid mask, and source noise from one cache root,
then writes one strict record per sample and dose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.content_metrics import evaluate_content_metrics
from evaluation.control_metrics import evaluate_control_metrics
from evaluation.local_pelvis_metrics import evaluate_local_pelvis_metrics
from geometry.authoritative_motion import authoritative_motion
from geometry.local_pelvis import target_relative_angle_curve_deg
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


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source(seed: int, root: Path) -> dict:
    m0_candidates = (
        "guided_artifacts/batch_000/m0_consistent_norm_batch.pt",
        "guided_artifacts/batch_000/m0_authority_norm_batch.pt",
    )
    found_m0 = [path for pattern in m0_candidates for path in root.glob(pattern)]
    if len(found_m0) != 1:
        raise RuntimeError(f"expected one authoritative M0 below {root}, found {found_m0}")
    m0_path = found_m0[0]
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
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"cache must contain unique sample IDs: {sample_ids}")
    return {
        "seed": seed,
        "root": root,
        "m0": physical,
        "noise": noise,
        "valid": valid,
        "sample_ids": sample_ids,
        "m0_path": m0_path,
        "archive_path": archive_path,
        "reference_path": None,
        "reference_sha256": None,
    }


def _load_reference(path: Path) -> list[dict]:
    reference = torch.load(path, map_location="cpu", weights_only=False)
    required = (
        "sample_ids",
        "seeds",
        "valid_frame_mask",
        "m0_motion_physical",
        "m0_source_noise",
    )
    missing = [name for name in required if name not in reference and name != "m0_source_noise"]
    if missing:
        raise ValueError(f"candidate M0 reference is missing {missing}")
    count = len(reference["sample_ids"])
    if "m0_source_noise" in reference:
        source_noise = reference["m0_source_noise"]
        noise_status = "AVAILABLE_IN_REFERENCE"
    else:
        source_noise = torch.zeros_like(reference["m0_motion_physical"])
        noise_status = "NOT_PRESENT_IN_REFERENCE_M7_UNUSED"
    tensors = (
        reference["valid_frame_mask"],
        reference["m0_motion_physical"],
        source_noise,
    )
    if len(reference["seeds"]) != count or any(value.shape[0] != count for value in tensors):
        raise ValueError("candidate M0 reference arrays have inconsistent lengths")
    source_hash = _sha256(path)
    rows = []
    for index in range(count):
        rows.append(
            {
                "seed": int(reference["seeds"][index]),
                "root": path.parent,
                "m0": reference["m0_motion_physical"][index : index + 1].float(),
                "noise": source_noise[index : index + 1].float(),
                "valid": reference["valid_frame_mask"][index : index + 1].bool(),
                "sample_ids": [str(reference["sample_ids"][index])],
                "paired_m0_ids": [str(reference["paired_m0_ids"][index])],
                "m0_path": path,
                "archive_path": path,
                "reference_path": path,
                "reference_sha256": source_hash,
                "noise_status": noise_status,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    if args.reference is not None:
        sources = _load_reference(args.reference)
    else:
        sources = [_load_source(seed, root) for seed, root in args.source]
        if len({item["seed"] for item in sources}) != len(sources):
            raise ValueError("each M7 cache source must use a unique seed")
    include_keys = set(args.include_key)
    if include_keys:
        available = {
            f"{sample_id}:{source['seed']}"
            for source in sources
            for sample_id in source["sample_ids"]
        }
        unknown = include_keys - available
        if unknown:
            raise KeyError(f"requested M7 keys are absent from the reference: {sorted(unknown)}")
        sources = [
            source
            for source in sources
            if any(
                f"{sample_id}:{source['seed']}" in include_keys
                for sample_id in source["sample_ids"]
            )
        ]
    if args.baseline_sha256:
        sources = [
            source
            for source in sources
            if tensor_sha256(source["m0"]) == args.baseline_sha256
        ]
        if not sources:
            raise KeyError(
                f"baseline SHA256 is absent from the reference: {args.baseline_sha256}"
            )
    keys = [(row["sample_ids"][0], row["seed"]) for row in sources]
    if len(keys) != len(set(keys)):
        raise ValueError(
            "reference contains multiple baselines for a selected sample/seed; "
            "provide --baseline-sha256"
        )
    doses = tuple(float(value) for value in args.dose)
    if not doses:
        raise ValueError("M7 requires at least one dose")
    method_config: dict[str, float | int] = (
        {
            "eps": 1.0e-6,
            "damping": 1.0e-5,
            "iterations": int(args.iterations),
            "max_step_deg": float(args.max_step_deg),
        }
        if args.task == "relative_pelvis_s1"
        else {"eps": 1.0e-6}
    )
    rows = []
    method = M7PAHTGeometricEdit()
    for source in sources:
        for index, sample_id in enumerate(source["sample_ids"]):
            baseline = source["m0"][index : index + 1]
            noise = source["noise"][index : index + 1]
            valid = source["valid"][index : index + 1]
            baseline_id = (
                source["paired_m0_ids"][index]
                if source.get("paired_m0_ids")
                else f"seed{source['seed']}_sample{sample_id}_{tensor_sha256(baseline)[:16]}"
            )
            for dose in doses:
                relative_task = args.task == "relative_pelvis_s1"
                target = (
                    target_relative_angle_curve_deg(baseline, dose)
                    if relative_task
                    else target_angle_curve_deg(baseline, dose)
                )
                request = GuidanceRequest(
                    prompt_id=sample_id,
                    seed=source["seed"],
                    target_dose_deg=dose,
                    constraint_pack=(
                        ConstraintPack.S1_RELATIVE
                        if relative_task
                        else ConstraintPack.C0
                    ),
                    base_noise=noise,
                    baseline_motion=baseline,
                    shared_evidence=SharedEvidence(
                        valid_mask=valid,
                        target_angle_curve_deg=target,
                        contact_evidence_version=f"m0:{baseline_id}:not_materialized",
                        ground_version=f"m0:{baseline_id}:not_materialized",
                        metadata={"source_cache_root": str(source["root"])},
                    ),
                    run_id=f"M7_{args.config_id}_s{source['seed']}_p{sample_id}_d{dose:+g}",
                    baseline_motion_id=baseline_id,
                )
                result = method.run(None, request, method_config)
                metrics = {
                    "control": (
                        evaluate_local_pelvis_metrics(
                            result.motion, baseline, valid, target_dose_deg=dose
                        )
                        if relative_task
                        else evaluate_control_metrics(
                            result.motion, baseline, target, valid, target_dose_deg=dose
                        )
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
                    "method_version": "v1",
                    "config_id": args.config_id,
                    "experiment_stage": args.experiment_stage,
                    "settings": method_config,
                    "method_config_hash": mapping_sha256(method_config),
                    "prompt_id": sample_id,
                    "seed": source["seed"],
                    "target_dose_deg": dose,
                    "constraint_pack": request.constraint_pack,
                    "baseline_motion_id": baseline_id,
                    "contact_evidence_version": request.shared_evidence.contact_evidence_version,
                    "ground_version": request.shared_evidence.ground_version,
                    "status": result.status,
                    "zero_dose_identity": (
                        bool(torch.equal(result.motion, baseline)) if dose == 0.0 else None
                    ),
                    "zero_dose_max_abs": (
                        float((result.motion - baseline).abs().max()) if dose == 0.0 else None
                    ),
                    "failure_reason": result.diagnostics_dict()["failure_reason"],
                    "all_metrics": metrics,
                    "all_diagnostics": result.diagnostics_dict(),
                    "source_m0_path": str(source["m0_path"]),
                    "source_archive_path": str(source["archive_path"]),
                    "source_m0_sha256": tensor_sha256(baseline),
                    "source_noise_sha256": tensor_sha256(noise),
                    "candidate_m0_reference": (
                        None
                        if source["reference_path"] is None
                        else str(source["reference_path"])
                    ),
                    "candidate_m0_reference_sha256": source["reference_sha256"],
                    "source_noise_status": source.get("noise_status", "AVAILABLE_IN_CACHE"),
                    "protocol": result.diagnostics_dict().get("protocol", PROTOCOL_NAME),
                }
                write_run_record(run_dir / "run_record.json", record)
                rows.append(record)
    summary = {
        "status": "COMPLETED" if all(row["status"] == "COMPLETED" for row in rows) else "FAILED",
        "protocol": (
            "vimogen_local_pelvis_s1_m7_v1"
            if args.task == "relative_pelvis_s1"
            else PROTOCOL_NAME
        ),
        "scope": args.experiment_stage,
        "expected_runs": len(rows),
        "completed_runs": sum(row["status"] == "COMPLETED" for row in rows),
        "angle_gate_passes": sum(
            (
                row["all_metrics"]["control"]["summary"].get("target_hit_count") == 1
                if args.task == "relative_pelvis_s1"
                else row["all_metrics"]["control"]["summary"]["sequence_angle_pass_rate"] == 1.0
            )
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
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", action="append", type=_source_argument)
    inputs.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--dose", action="append", type=float, default=[])
    parser.add_argument("--include-key", action="append", default=[])
    parser.add_argument("--baseline-sha256", default="")
    parser.add_argument("--config-id", default="reference")
    parser.add_argument("--experiment-stage", default="candidate_scan")
    parser.add_argument(
        "--task",
        choices=("legacy_c0", "relative_pelvis_s1"),
        default="legacy_c0",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--max-step-deg", type=float, default=1.0)
    args = parser.parse_args()
    if not args.dose:
        args.dose = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0]
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
