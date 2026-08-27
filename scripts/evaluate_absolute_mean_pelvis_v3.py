#!/usr/bin/env python3
"""Evaluate one completed v3 generation run without another model call."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.absolute_mean_pelvis_v3 import evaluate_single, summarize_rows
from motion_rep.consistency_v3 import reconcile_motion_tensor_v3
from motion_rep.phase1 import MOTION_LAYOUT
from sampling.absolute_mean_pelvis_guidance_v3 import pelvis_angle_curve


def _flatten_terminal_records(payload) -> list[dict]:
    records: list[dict] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("terminal_records"), list):
            records.extend(payload["terminal_records"])
        for value in payload.get("samples", []):
            records.extend(_flatten_terminal_records(value))
        for value in payload.get("records", []):
            records.extend(_flatten_terminal_records(value))
    elif isinstance(payload, list):
        for value in payload:
            records.extend(_flatten_terminal_records(value))
    return records


def _batch_dirs(root: Path) -> list[Path]:
    rows = sorted(path for path in root.glob("batch_*") if path.is_dir())
    if not rows:
        raise FileNotFoundError(f"no batch directories under {root}")
    return rows


def _physical(norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (norm.float() * std + mean).float()


def evaluate_run(run_root: Path) -> dict:
    record_path = run_root / "run_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record["status"] not in {"COMPLETED_GENERATION_PENDING_EVALUATION", "COMPLETED"}:
        raise RuntimeError(f"generation status is {record['status']!r}")
    manifest = json.loads(Path(record["input_manifest"]).read_text(encoding="utf-8"))
    mean = torch.from_numpy(np.load(ROOT / "data/meta_info/mean.npy")).float()
    std = torch.from_numpy(np.load(ROOT / "data/meta_info/std.npy")).float()
    angle_rows: list[dict] = []
    metric_rows: list[dict] = []
    motion_dir = run_root / "motions"
    motion_dir.mkdir(exist_ok=True)
    cursor = 0
    target = float(record["target_mean_deg"])
    seed = int(record["seed"])
    m0_dirs = _batch_dirs(run_root / "m0_artifacts")
    guided_dirs = _batch_dirs(run_root / "guided_artifacts")
    if [path.name for path in m0_dirs] != [path.name for path in guided_dirs]:
        raise RuntimeError("M0 and guided batch directories do not match")
    for m0_dir, guided_dir in zip(m0_dirs, guided_dirs):
        m0_norm = torch.load(m0_dir / "m0_raw_norm_batch.pt", map_location="cpu", weights_only=True).float()
        g0_norm = torch.load(guided_dir / "g0_norm_batch.pt", map_location="cpu", weights_only=True).float()
        g1_norm = torch.load(guided_dir / "g1_norm_batch.pt", map_location="cpu", weights_only=True).float()
        if m0_norm.shape != g0_norm.shape or g0_norm.shape != g1_norm.shape:
            raise RuntimeError(f"shape mismatch in {guided_dir}")
        mask = torch.ones(m0_norm.shape[:2], dtype=torch.bool)
        # M0 is the direct model endpoint and may contain inconsistent J/dJ;
        # make its authority comparable to G0/G1 through the same v3 boundary.
        m0_result = reconcile_motion_tensor_v3(
            m0_norm,
            mean=mean,
            std=std,
            input_standardized=True,
            output_standardized=False,
            valid_mask=mask,
        )
        m0_phys = m0_result.motion
        g0_phys = _physical(g0_norm, mean, std)
        g1_phys = _physical(g1_norm, mean, std)
        summary_payload = json.loads((guided_dir / "guidance_summary.json").read_text(encoding="utf-8"))
        terminals = _flatten_terminal_records(summary_payload)
        if len(terminals) != m0_norm.shape[0]:
            raise RuntimeError(f"terminal record count {len(terminals)} != batch size {m0_norm.shape[0]}")
        batch_items = manifest[cursor : cursor + m0_norm.shape[0]]
        if len(batch_items) != m0_norm.shape[0]:
            raise RuntimeError("artifact count exceeds input manifest")
        for index, item in enumerate(batch_items):
            sample_id = str(item.get("global_id", item.get("id", cursor + index)))
            row_mask = m0_result.valid_mask[index]
            for method, motion in (("m0", m0_phys[index]), ("g0", g0_phys[index]), ("g1", g1_phys[index])):
                torch.save(motion.detach().cpu(), motion_dir / f"sample_{sample_id}_{method}_physical.pt")
            curves = {"M0": pelvis_angle_curve(m0_phys[index]), "G0": pelvis_angle_curve(g0_phys[index]), "G1": pelvis_angle_curve(g1_phys[index])}
            running = {name: 0.0 for name in curves}
            count = 0
            for frame in range(m0_norm.shape[1]):
                if not bool(row_mask[frame]):
                    continue
                count += 1
                values = {name: float(curve[frame]) for name, curve in curves.items()}
                for name, value in values.items():
                    running[name] += value
                angle_rows.append({
                    "sample_id": sample_id, "seed": seed, "target_mean_deg": target,
                    "frame": frame, "m0_angle_deg": values["M0"], "g0_angle_deg": values["G0"], "g1_angle_deg": values["G1"],
                    "m0_running_mean_deg": running["M0"] / count, "g0_running_mean_deg": running["G0"] / count, "g1_running_mean_deg": running["G1"] / count,
                })
            metric_rows.append(evaluate_single(sample_id=sample_id, method="G0", seed=seed, target_mean_deg=target, baseline_phys=m0_phys[index], candidate_phys=g0_phys[index], valid_mask=row_mask))
            metric_rows.append(evaluate_single(sample_id=sample_id, method="G1", seed=seed, target_mean_deg=target, baseline_phys=m0_phys[index], candidate_phys=g1_phys[index], valid_mask=row_mask, terminal_record=terminals[index]))
        cursor += m0_norm.shape[0]
    if cursor != len(manifest):
        raise RuntimeError(f"evaluated {cursor} items but manifest has {len(manifest)}")

    angle_dir, metric_dir, summary_dir = run_root / "angles", run_root / "metrics", run_root / "summaries"
    for path in (angle_dir, metric_dir, summary_dir):
        path.mkdir(exist_ok=True)
    angle_path = angle_dir / "per_frame_angles.csv"
    metric_path = metric_dir / "per_action_metrics.csv"
    with angle_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(angle_rows[0])); writer.writeheader(); writer.writerows(angle_rows)
    with metric_path.open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in metric_rows for key in row if key != "consistency"}); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows([{key: value for key, value in row.items() if key != "consistency"} for row in metric_rows])
    summaries = {method: summarize_rows([row for row in metric_rows if row["method"] == method]) for method in ("G0", "G1")}
    output = {"protocol": "vimogen_absolute_mean_pelvis_v3_tail_safe", "run_root": str(run_root), "split": record["split"], "seed": seed, "target_mean_deg": target, "summaries": summaries, "per_frame_csv": str(angle_path), "per_action_csv": str(metric_path)}
    (summary_dir / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Absolute mean pelvis v3 run summary", "", f"- Split: {record['split']}", f"- Seed: {seed}", f"- Absolute target: +{target:g} deg", "", "| Method | Median error (deg) | <=2 deg rate | Curve corr. | Std ratio | J-FK max (m) | dJ max (m) | dR max (deg) | dT max (m) | G1 trigger rate |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in ("G0", "G1"):
        row = summaries[method]
        lines.append(f"| {method} | {row['absolute_mean_error_median_deg']:.6f} | {row['absolute_mean_error_le_2deg_rate']:.4f} | {row['centered_curve_correlation_median']:.6f} | {row['fluctuation_std_ratio_median']:.6f} | {row['joint_fk_residual_max_m']:.8g} | {row['joint_velocity_residual_max_m']:.8g} | {row['root_rotation_velocity_residual_max_deg']:.8g} | {row['root_translation_velocity_residual_max_m']:.8g} | {row['terminal_trigger_rate']:.4f} |")
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    record["status"] = "COMPLETED"; record["evaluation_summary"] = str(summary_dir / "summary.json")
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", type=Path, required=True); args = parser.parse_args(); print(json.dumps(evaluate_run(args.run_root), indent=2))


if __name__ == "__main__":
    main()
