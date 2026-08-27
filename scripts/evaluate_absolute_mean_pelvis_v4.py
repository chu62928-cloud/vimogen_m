#!/usr/bin/env python3
"""Evaluate a completed v4 run without another model call."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from evaluation.absolute_mean_pelvis_v4 import evaluate_single, summarize_rows
from motion_rep.anatomical_pelvis import load_pelvis_calibration
from motion_rep.consistency_v3 import reconcile_motion_tensor_v3
from sampling.absolute_mean_pelvis_guidance_v4 import anatomical_angle_curves


def _batch_dirs(root: Path) -> list[Path]:
    rows = sorted(path for path in root.glob("batch_*") if path.is_dir())
    if not rows:
        raise FileNotFoundError(f"no batch directories under {root}")
    return rows


def _physical(norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (norm.float() * std + mean).float()


def _flatten_terminal_records(payload) -> list[dict]:
    if isinstance(payload, dict):
        records = list(payload.get("terminal_records", []))
        for value in payload.get("samples", []) + payload.get("records", []):
            records.extend(_flatten_terminal_records(value))
        return records
    if isinstance(payload, list):
        records: list[dict] = []
        for value in payload:
            records.extend(_flatten_terminal_records(value))
        return records
    return []


def evaluate_run(run_root: Path, calibration_path: Path) -> dict:
    record = json.loads((run_root / "run_record.json").read_text(encoding="utf-8"))
    if record.get("status") not in {"COMPLETED_GENERATION_PENDING_EVALUATION", "COMPLETED"}:
        raise RuntimeError(f"generation status is {record.get('status')!r}")
    manifest = json.loads(Path(record["input_manifest"]).read_text(encoding="utf-8"))
    mean = torch.from_numpy(np.load(ROOT / "data/meta_info/mean.npy")).float()
    std = torch.from_numpy(np.load(ROOT / "data/meta_info/std.npy")).float()
    calibration = load_pelvis_calibration(calibration_path)
    metric_rows: list[dict] = []
    angle_rows: list[dict] = []
    cursor = 0
    m0_dirs = _batch_dirs(run_root / "m0_artifacts")
    guided_dirs = _batch_dirs(run_root / "guided_artifacts")
    for m0_dir, guided_dir in zip(m0_dirs, guided_dirs):
        m0_norm = torch.load(m0_dir / "m0_raw_norm_batch.pt", map_location="cpu", weights_only=True).float()
        g0_norm = torch.load(guided_dir / "g0_norm_batch.pt", map_location="cpu", weights_only=True).float()
        g1_norm = torch.load(guided_dir / "g1_norm_batch.pt", map_location="cpu", weights_only=True).float()
        if m0_norm.shape != g0_norm.shape or g0_norm.shape != g1_norm.shape:
            raise RuntimeError(f"shape mismatch in {guided_dir}")
        mask = torch.ones(m0_norm.shape[:2], dtype=torch.bool)
        m0_result = reconcile_motion_tensor_v3(m0_norm, mean=mean, std=std, input_standardized=True, output_standardized=False, valid_mask=mask)
        m0_phys = m0_result.motion
        g0_phys, g1_phys = _physical(g0_norm, mean, std), _physical(g1_norm, mean, std)
        terminals = _flatten_terminal_records(json.loads((guided_dir / "guidance_summary.json").read_text(encoding="utf-8")))
        batch_items = manifest[cursor : cursor + m0_norm.shape[0]]
        if len(terminals) not in (0, m0_norm.shape[0]):
            raise RuntimeError(f"terminal record count {len(terminals)} != batch size")
        for index, item in enumerate(batch_items):
            sample_id = str(item.get("global_id", item.get("id", cursor + index)))
            row_mask = m0_result.valid_mask[index]
            curves = {name: anatomical_angle_curves(motion[index], calibration)["pelvis_deg"] for name, motion in (("M0", m0_phys), ("G0", g0_phys), ("G1", g1_phys))}
            running = {name: 0.0 for name in curves}
            count = 0
            for frame in range(m0_norm.shape[1]):
                if not bool(row_mask[frame]):
                    continue
                count += 1
                values = {name: float(curve[frame]) for name, curve in curves.items()}
                for name, value in values.items():
                    running[name] += value
                angle_rows.append({"sample_id": sample_id, "seed": int(record["seed"]), "target_mean_deg": float(record["target_mean_deg"]), "frame": frame, "m0_angle_deg": values["M0"], "g0_angle_deg": values["G0"], "g1_angle_deg": values["G1"], "m0_running_mean_deg": running["M0"] / count, "g0_running_mean_deg": running["G0"] / count, "g1_running_mean_deg": running["G1"] / count})
            metric_rows.extend([
                evaluate_single(sample_id=sample_id, method="G0", seed=int(record["seed"]), target_mean_deg=float(record["target_mean_deg"]), baseline_phys=m0_phys[index], candidate_phys=g0_phys[index], valid_mask=row_mask, calibration=calibration),
                evaluate_single(sample_id=sample_id, method="G1", seed=int(record["seed"]), target_mean_deg=float(record["target_mean_deg"]), baseline_phys=m0_phys[index], candidate_phys=g1_phys[index], valid_mask=row_mask, calibration=calibration, terminal_record=(terminals[index] if terminals else None)),
            ])
        cursor += m0_norm.shape[0]
    if cursor != len(manifest):
        raise RuntimeError(f"evaluated {cursor} items but manifest has {len(manifest)}")
    output_dir = run_root / "summaries"
    output_dir.mkdir(exist_ok=True)
    (run_root / "angles").mkdir(exist_ok=True)
    (run_root / "metrics").mkdir(exist_ok=True)
    with (run_root / "angles/per_frame_angles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(angle_rows[0])); writer.writeheader(); writer.writerows(angle_rows)
    fields = sorted({key for row in metric_rows for key in row if key != "consistency"})
    with (run_root / "metrics/per_action_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows([{key: value for key, value in row.items() if key != "consistency"} for row in metric_rows])
    summaries = {method: summarize_rows([row for row in metric_rows if row["method"] == method]) for method in ("G0", "G1")}
    result = {"protocol": "vimogen_absolute_mean_pelvis_v4_anatomical_local", "run_root": str(run_root), "summaries": summaries, "calibration": str(calibration_path), "per_frame_csv": str(run_root / "angles/per_frame_angles.csv"), "per_action_csv": str(run_root / "metrics/per_action_metrics.csv")}
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["status"] = "COMPLETED"; record["evaluation_summary"] = str(output_dir / "summary.json")
    (run_root / "run_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", type=Path, required=True); parser.add_argument("--calibration", type=Path, required=True); args = parser.parse_args(); print(json.dumps(evaluate_run(args.run_root, args.calibration), indent=2))


if __name__ == "__main__":
    main()
