#!/usr/bin/env python3
"""Materialise strict per-sample records for a real sampling-hook batch."""

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
from guidance.base import ConstraintPack, mapping_sha256, tensor_sha256, write_run_record
from motion_rep.pose_authority import authority_project
from geometry.pelvis_angle import target_angle_curve_deg


def _one(root: Path, pattern: str) -> Path:
    paths = list(root.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {pattern!r} below {root}, found {len(paths)}")
    return paths[0]


def _diag(summary: dict, index: int, before: torch.Tensor, after: torch.Tensor, valid: torch.Tensor, elapsed: float) -> dict:
    # The batch-invariant sampler stores one outer record per condition and
    # puts the actual per-sample step traces below ``records[0].samples``.
    outer = summary.get("records", [])
    payload = outer[0] if outer and isinstance(outer[0], dict) else summary
    samples = payload.get("samples", [])
    sample = samples[index] if index < len(samples) else summary
    steps = sample.get("step_records", [])
    active = [row for row in steps if row.get("active", False)]
    update = (after - before)[valid]
    def total(key: str) -> int:
        return int(sum(int(row.get(key, 0) or 0) for row in active))
    return {
        "angle_residual_mae_deg": float("nan"),
        "angle_residual_p95_deg": float("nan"),
        "state_update_norm": float(torch.linalg.vector_norm(update).cpu()),
        "max_update_norm": float(update.abs().max().cpu()),
        "model_nfe": 50,
        "full_rollout_count": total("full_rollout_count"),
        "smplx_forward_count": len(active),
        "jvp_count": total("jvp_count"),
        "vjp_count": total("vjp_count"),
        "backward_count": total("backward_count"),
        "solver_iterations": total("solver_iterations"),
        "rejected_steps": total("rejected_steps") + total("rejected_backtracks"),
        "nonfinite_count": total("nonfinite_count"),
        "fallback_used": bool(any(row.get("fallback_used", False) for row in active)),
        "failure_reason": "",
        "wall_time_sec": float(elapsed),
        "peak_gpu_mem_gb": 0.0,
        "step_records": steps,
    }


def run(run_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    meta = json.loads((run_root / "run_record.json").read_text(encoding="utf-8"))
    artifact = run_root / "guided_artifacts/batch_000"
    archive_path = _one(run_root, "trainer/**/batch_000/mbench_raw_norm_batch.pt")
    archive = torch.load(archive_path, map_location="cpu", weights_only=True)
    mean, std = archive["motion_mean"].float(), archive["motion_std"].float()
    valid = archive["motion_mask"].bool()
    before_norm = torch.load(artifact / "m0_authority_norm_batch.pt", map_location="cpu", weights_only=True).float()
    after_norm = torch.load(artifact / "g0_norm_batch.pt", map_location="cpu", weights_only=True).float()
    before = authority_project(
        before_norm, valid_mask=valid, mean=mean, std=std,
        input_standardized=True, output_standardized=False,
    ).motion
    after = authority_project(
        after_norm, valid_mask=valid, mean=mean, std=std,
        input_standardized=True, output_standardized=False,
    ).motion
    summary = json.loads((artifact / "guidance_summary.json").read_text(encoding="utf-8"))
    sample_ids = [str(x) for x in archive["sample_ids"]]
    output.mkdir(parents=True)
    rows = []
    for index, sample_id in enumerate(sample_ids):
        baseline = before[index:index + 1]
        candidate = after[index:index + 1]
        mask = valid[index:index + 1]
        target = target_angle_curve_deg(baseline, float(meta["target_dose_deg"]))
        control = evaluate_control_metrics(
            candidate, baseline, target, mask,
            target_dose_deg=float(meta["target_dose_deg"]),
        )
        content = evaluate_content_metrics(candidate, baseline, mask)
        run_id = f"{meta['method']}_s{meta['seed']}_p{sample_id}_d{float(meta['target_dose_deg']):+g}"
        row_dir = output / run_id
        row_dir.mkdir()
        torch.save(candidate.cpu(), row_dir / "motion_physical.pt")
        diagnostics = _diag(
            summary, index, baseline, candidate, mask,
            float(meta.get("elapsed_seconds", 0.0)),
        )
        # Final metrics are authoritative; replace the provisional step value.
        diagnostics["angle_residual_mae_deg"] = control["per_sequence"][0]["angle_mae_deg"]
        diagnostics["angle_residual_p95_deg"] = control["per_sequence"][0]["angle_p95_deg"]
        record = {
            "run_id": run_id,
            "code_commit": meta["code_commit"],
            "vimogen_checkpoint_hash": meta.get("checkpoint_hash", "not_recorded_in_runner_metadata"),
            "evaluator_version": "m1_m7_evaluator_v1",
            "method_name": meta["method"],
            "method_config_hash": mapping_sha256(meta["settings"]),
            "prompt_id": sample_id,
            "seed": int(meta["seed"]),
            "target_dose_deg": float(meta["target_dose_deg"]),
            "constraint_pack": ConstraintPack.C0,
            "baseline_motion_id": f"current_environment_authority_m0_{tensor_sha256(baseline)[:16]}",
            "contact_evidence_version": "pending_shared_fk_materialization",
            "ground_version": "pending_shared_fk_materialization",
            "status": "COMPLETED" if control["summary"]["sequence_angle_pass_rate"] == 1.0 else "ANGLE_GATE_FAIL",
            "failure_reason": "",
            "all_metrics": {
                "control": control,
                "content": content,
                "physical": {
                    "status": "PENDING_SHARED_FK_MARKER_MATERIALIZATION",
                    "reason": "real generation batch has no frozen heel/toe marker bundle",
                },
            },
            "all_diagnostics": diagnostics,
            "guidance_summary": summary,
        }
        write_run_record(row_dir / "run_record.json", record)
        rows.append(record)
    result = {
        "status": "COMPLETED" if all(row["status"] == "COMPLETED" for row in rows) else "ANGLE_GATE_FAIL",
        "method": meta["method"],
        "seed": meta["seed"],
        "target_dose_deg": meta["target_dose_deg"],
        "sample_count": len(rows),
        "angle_gate_pass_count": sum(row["status"] == "COMPLETED" for row in rows),
        "records": [str(output / f"{meta['method']}_s{meta['seed']}_p{row['prompt_id']}_d{float(meta['target_dose_deg']):+g}" / "run_record.json") for row in rows],
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run_root, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
