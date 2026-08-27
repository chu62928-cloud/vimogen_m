#!/usr/bin/env python3
"""Read-only audit of turning-sample M1 angle failures.

The script consumes only frozen M0/B0/M1 artifacts.  It never reruns a model,
changes a tensor, overwrites an existing report, or filters a sample.  The
``recanonicalized`` values are derived diagnostics: they are not replacements
for the frozen M1 main metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_rep.baselines import build_b0  # noqa: E402
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe  # noqa: E402
from scripts.evaluate_m1_pilot import angle_curve, load_norm, tensor_sha256  # noqa: E402


SPEED_THRESHOLD_M_PER_FRAME = 0.05 / 20.0
TARGET_ERROR_THRESHOLD_DEG = 2.0


def _collect(root: Path, subdir: str, name: str) -> torch.Tensor:
    paths = sorted((root / subdir).glob(f"batch_*/{name}"))
    if not paths:
        raise FileNotFoundError(f"missing {name} under {root / subdir}")
    return torch.cat(
        [torch.load(path, weights_only=True, map_location="cpu") for path in paths]
    )


def _physical_batch(root: Path, subdir: str, name: str, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    value = _collect(root, subdir, name).float()
    return value * std[None, None, :] + mean[None, None, :]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _position_velocity(motion: torch.Tensor) -> torch.Tensor:
    """Return the same T-frame position-difference convention as angle_curve."""

    translation = motion[:, MOTION_LAYOUT.root_translation]
    velocity = torch.zeros_like(translation)
    if translation.shape[0] > 1:
        velocity[:-1] = translation[1:] - translation[:-1]
        velocity[-1] = velocity[-2]
    return velocity


def _speed_summary(motion: torch.Tensor) -> dict[str, Any]:
    position_velocity = _position_velocity(motion)
    horizontal_speed = torch.linalg.vector_norm(position_velocity[:, :2], dim=-1)
    stored_velocity = motion[:, MOTION_LAYOUT.root_translation_velocity]
    stored_speed = torch.linalg.vector_norm(stored_velocity[:, :2], dim=-1)
    # The final stored velocity refers to the hidden T+1 boundary and cannot
    # be checked against a difference of the visible T positions.
    visible_consistency = (
        float((stored_velocity[:-1] - position_velocity[:-1]).abs().max().item())
        if motion.shape[0] > 1
        else 0.0
    )
    valid = horizontal_speed >= SPEED_THRESHOLD_M_PER_FRAME
    return {
        "threshold_m_per_frame": SPEED_THRESHOLD_M_PER_FRAME,
        "position_difference_median_m_per_frame": float(horizontal_speed.median().item()),
        "position_difference_p05_m_per_frame": float(torch.quantile(horizontal_speed, 0.05).item()),
        "position_difference_p95_m_per_frame": float(torch.quantile(horizontal_speed, 0.95).item()),
        "position_difference_max_m_per_frame": float(horizontal_speed.max().item()),
        "stored_velocity_median_m_per_frame": float(stored_speed.median().item()),
        "stored_velocity_p95_m_per_frame": float(torch.quantile(stored_speed, 0.95).item()),
        "travel_valid_frame_count": int(valid.sum().item()),
        "frame_count": int(valid.numel()),
        "stored_vs_visible_position_difference_max_abs_m": visible_consistency,
        "last_stored_velocity_is_hidden_boundary": True,
    }


def _heading_summary(motion: torch.Tensor) -> dict[str, Any]:
    velocity = _position_velocity(motion)
    horizontal = velocity[:, :2]
    speed = torch.linalg.vector_norm(horizontal, dim=-1)
    valid = speed >= SPEED_THRESHOLD_M_PER_FRAME
    heading_deg = torch.rad2deg(torch.atan2(horizontal[:, 0], horizontal[:, 1]))
    selected = heading_deg[valid].detach().cpu().numpy()
    if selected.size == 0:
        return {"valid_frame_count": 0, "heading_unwrapped_range_degrees": None}
    unwrapped = np.unwrap(np.deg2rad(selected))
    return {
        "valid_frame_count": int(selected.size),
        "heading_unwrapped_range_degrees": float(np.rad2deg(unwrapped.max() - unwrapped.min())),
        "heading_first_last_unwrapped_degrees": [
            float(np.rad2deg(unwrapped[0])),
            float(np.rad2deg(unwrapped[-1])),
        ],
    }


def _facing_yaw_degrees(motion: torch.Tensor) -> torch.Tensor:
    root = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation])
    local_forward = torch.zeros(3, dtype=root.dtype)
    local_forward[2] = 1.0
    forward = root @ local_forward
    return torch.rad2deg(torch.atan2(forward[:, 0], forward[:, 1]))


def _angle_summary(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    target: float,
) -> dict[str, Any]:
    base_angle, base_valid_canonical = angle_curve(baseline, heading_mode="canonical_y")
    cand_angle, cand_valid_canonical = angle_curve(candidate, heading_mode="canonical_y")
    _, base_valid_travel = angle_curve(baseline, heading_mode="travel")
    _, cand_valid_travel = angle_curve(candidate, heading_mode="travel")
    canonical_valid = base_valid_canonical & cand_valid_canonical
    travel_valid = base_valid_travel & cand_valid_travel
    shift = cand_angle - base_angle
    error = shift - float(target)
    selected = error[canonical_valid].float()
    selected_shift = shift[canonical_valid].float()
    cand_travel_angle, _ = angle_curve(candidate, heading_mode="travel")
    travel_angle_difference = (cand_angle[travel_valid] - cand_travel_angle[travel_valid]).abs()
    return {
        "canonical_y_valid_frame_count": int(canonical_valid.sum().item()),
        "travel_valid_frame_count_intersection": int(travel_valid.sum().item()),
        "target_delta_degrees": float(target),
        "median_shift_degrees": float(selected_shift.median().item()),
        "median_absolute_target_error_degrees": float(selected.abs().median().item()),
        "p95_absolute_target_error_degrees": float(torch.quantile(selected.abs(), 0.95).item()),
        "max_absolute_target_error_degrees": float(selected.abs().max().item()),
        "canonical_y_vs_travel_angle_max_abs_degrees": float(travel_angle_difference.max().item()) if travel_angle_difference.numel() else 0.0,
        "candidate_angle_median_degrees": float(cand_angle.median().item()),
        "baseline_angle_median_degrees": float(base_angle.median().item()),
    }


def _waveform(
    baseline: torch.Tensor,
    m0_raw: torch.Tensor,
    m0_official: torch.Tensor,
    m1_raw: torch.Tensor,
    m1_official: torch.Tensor,
    target: float,
) -> dict[str, Any]:
    recanonical_raw = build_b0(m1_raw).motion
    recanonical_official = build_b0(m1_official).motion
    motions = {
        "m0_raw": m0_raw,
        "m0_official": m0_official,
        "b0": baseline,
        "m1_raw": m1_raw,
        "m1_official": m1_official,
        "m1_raw_recanonicalized_diagnostic": recanonical_raw,
        "m1_official_recanonicalized_diagnostic": recanonical_official,
    }
    angles = {
        name: angle_curve(value, heading_mode="canonical_y")[0].detach().cpu().tolist()
        for name, value in motions.items()
    }
    base = angle_curve(baseline, heading_mode="canonical_y")[0]
    for name in ("m1_raw", "m1_official", "m1_raw_recanonicalized_diagnostic", "m1_official_recanonicalized_diagnostic"):
        angles[f"{name}_shift_from_b0"] = (torch.as_tensor(angles[name]) - base.cpu()).tolist()
        angles[f"{name}_error_from_target"] = (torch.as_tensor(angles[name]) - base.cpu() - float(target)).tolist()
    return {
        "frame": list(range(baseline.shape[0])),
        "canonical_y_angle_degrees": angles,
        "travel_valid_masks": {
            name: angle_curve(value, heading_mode="travel")[1].tolist()
            for name, value in motions.items()
        },
        "root_horizontal_speed_m_per_frame": {
            name: torch.linalg.vector_norm(_position_velocity(value)[:, :2], dim=-1).tolist()
            for name, value in motions.items()
        },
        "root_facing_yaw_degrees": {
            name: _facing_yaw_degrees(value).tolist() for name, value in motions.items()
        },
    }


def audit(
    *,
    metrics_path: Path,
    input_path: Path,
    m0_root: Path,
    m1_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    row_by_id = {str(row["sample_id"]): row for row in rows}
    all_ids = [str(row["sample_id"]) for row in rows]
    turning_ids = [sid for sid in all_ids if row_by_id[sid].get("category") == "turning_walk"]
    if not turning_ids:
        raise ValueError("input has no turning_walk rows")
    mean = torch.from_numpy(np.load(ROOT / "data/meta_info/mean.npy")).float()
    std = torch.from_numpy(np.load(ROOT / "data/meta_info/std.npy")).float()
    records: list[dict[str, Any]] = []
    waveforms: dict[str, Any] = {}
    selected = {("1", "34122", 10.0), ("2", "14796", 10.0), ("0", "34122", 10.0), ("0", "14796", 10.0)}

    for label, run_metrics in metrics["runs"].items():
        seed = int(run_metrics["seed"])
        target = float(run_metrics["target_delta_degrees"])
        m0_dir = m0_root / f"seed_{seed:03d}"
        m0_raw_batch = _physical_batch(m0_dir, "artifacts", "m0_raw_norm_batch.pt", mean, std)
        m0_official_batch = _physical_batch(m0_dir, "artifacts", "m0_official_norm_batch.pt", mean, std)
        run = m1_root / f"seed_{seed:03d}" / f"delta_{int(target):02d}deg"
        m1_raw_batch = _physical_batch(run, "m1_artifacts", "m1_raw_norm_batch.pt", mean, std)
        m1_official_batch = _physical_batch(run, "m1_artifacts", "m1_official_norm_batch.pt", mean, std)
        for sample_id in turning_ids:
            index = list(run_metrics["sample_ids"]).index(sample_id)
            m0_raw = m0_raw_batch[index]
            m0_official = m0_official_batch[index]
            baseline = build_b0(m0_raw).motion
            m1_raw = m1_raw_batch[index]
            m1_official = m1_official_batch[index]
            m1_raw_recanonicalized = build_b0(m1_raw).motion
            m1_official_recanonicalized = build_b0(m1_official).motion
            row = row_by_id[sample_id]
            raw_angle = _angle_summary(baseline, m1_raw, target)
            official_angle = _angle_summary(baseline, m1_official, target)
            raw_recanonicalized_angle = _angle_summary(baseline, m1_raw_recanonicalized, target)
            official_recanonicalized_angle = _angle_summary(baseline, m1_official_recanonicalized, target)
            video_paths = sorted(
                str(path)
                for path in run.rglob(f"{sample_id}/*.mp4")
            )
            records.append(
                {
                    "run_label": label,
                    "seed": seed,
                    "target_delta_degrees": target,
                    "sample_id": sample_id,
                    "category": row.get("category"),
                    "prompt": row.get("prompt", row.get("motion_text_annot")),
                    "source_motion_path": row.get("source_motion_path"),
                    "input_reference_motion_used": bool(row.get("use_ref_motion", False)),
                    "video_paths": video_paths,
                    "noise_and_m0_flags": {
                        "z0_bitwise_equal_m0_run": bool(run_metrics.get("z0_bitwise_equal_m0_run", False)),
                        "m0_raw_bitwise_equal_m0_run": bool(run_metrics.get("m0_raw_bitwise_equal_m0_run", False)),
                        "m0_official_bitwise_equal_m0_run": bool(run_metrics.get("m0_official_bitwise_equal_m0_run", False)),
                    },
                    "m0_raw_sha256": tensor_sha256(m0_raw),
                    "m0_raw_vs_official_angle_median_abs_degrees": float(
                        (angle_curve(m0_raw, heading_mode="canonical_y")[0] - angle_curve(m0_official, heading_mode="canonical_y")[0]).abs().median().item()
                    ),
                    "m0_raw_vs_b0_angle_median_abs_degrees": float(
                        (angle_curve(m0_raw, heading_mode="canonical_y")[0] - angle_curve(baseline, heading_mode="canonical_y")[0]).abs().median().item()
                    ),
                    "m0_official_vs_b0_angle_median_abs_degrees": float(
                        (angle_curve(m0_official, heading_mode="canonical_y")[0] - angle_curve(baseline, heading_mode="canonical_y")[0]).abs().median().item()
                    ),
                    "b0_physical_rms_delta_from_m0_raw": float(torch.sqrt((baseline - m0_raw).square().mean()).item()),
                    "root_speed": {
                        "b0": _speed_summary(baseline),
                        "m1_raw": _speed_summary(m1_raw),
                        "m1_official": _speed_summary(m1_official),
                    },
                    "travel_heading": {
                        "b0": _heading_summary(baseline),
                        "m1_raw": _heading_summary(m1_raw),
                        "m1_official": _heading_summary(m1_official),
                    },
                    "facing_yaw_unwrapped_range_degrees": {
                        name: float(np.rad2deg(np.unwrap(np.deg2rad(_facing_yaw_degrees(value).numpy())).ptp()))
                        for name, value in (("b0", baseline), ("m1_raw", m1_raw), ("m1_official", m1_official))
                    },
                    "m1_raw": raw_angle,
                    "m1_official": official_angle,
                    "m1_raw_recanonicalized_diagnostic": raw_recanonicalized_angle,
                    "m1_official_recanonicalized_diagnostic": official_recanonicalized_angle,
                }
            )
            if (str(seed), sample_id, target) in selected:
                waveforms[f"seed_{seed:03d}_sample_{sample_id}_delta_{int(target):02d}deg"] = _waveform(
                    baseline, m0_raw, m0_official, m1_raw, m1_official, target
                )

    main_records = [record for record in records]
    recanonical_gain = [
        record["m1_official"]["median_absolute_target_error_degrees"]
        - record["m1_official_recanonicalized_diagnostic"]["median_absolute_target_error_degrees"]
        for record in main_records
    ]
    b0_baseline_shift = [record["m0_raw_vs_b0_angle_median_abs_degrees"] for record in main_records]
    report = {
        "status": "VERIFIED_TURNING_ANGLE_AUDIT_READ_ONLY",
        "main_metric_unchanged": True,
        "reviewer_scope": "reviewer1_only; reviewer2_ignored_by_user_instruction",
        "rerun_performed": False,
        "parameter_change": False,
        "sample_exclusion": False,
        "protocol": {
            "main_angle_heading_mode": metrics.get("protocol", {}).get("heading_mode", "canonical_y"),
            "candidate_local_forward_axis": "+z",
            "canonical_axes": "x=left/right, y=forward, z=up",
            "travel_speed_threshold_m_per_frame": SPEED_THRESHOLD_M_PER_FRAME,
            "angle_formula": "atan2(forward_z, sqrt((forward·heading)^2 + (forward·(heading×up))^2))",
            "formal_gate_threshold_degrees": TARGET_ERROR_THRESHOLD_DEG,
            "recanonicalized_values_are_derived_only": True,
        },
        "input_texts": [
            {
                "sample_id": sid,
                "category": row_by_id[sid].get("category"),
                "prompt": row_by_id[sid].get("prompt", row_by_id[sid].get("motion_text_annot")),
                "source_motion_path": row_by_id[sid].get("source_motion_path"),
                "use_ref_motion": bool(row_by_id[sid].get("use_ref_motion", False)),
            }
            for sid in turning_ids
        ],
        "source_hashes": {
            "metrics": _sha256_file(metrics_path),
            "input": _sha256_file(input_path),
            "pelvis_angle_py": _sha256_file(ROOT / "motion_rep/pelvis_angle.py"),
            "phase1_py": _sha256_file(ROOT / "motion_rep/phase1.py"),
            "baselines_py": _sha256_file(ROOT / "motion_rep/baselines.py"),
        },
        "aggregate": {
            "turning_record_count": len(main_records),
            "main_official_gate_failures": sum(
                record["m1_official"]["median_absolute_target_error_degrees"] > TARGET_ERROR_THRESHOLD_DEG
                for record in main_records
            ),
            "max_canonical_y_vs_travel_angle_difference_degrees": max(
                record["m1_official"]["canonical_y_vs_travel_angle_max_abs_degrees"] for record in main_records
            ),
            "max_b0_minus_m0_raw_angle_median_abs_degrees": max(b0_baseline_shift),
            "median_b0_minus_m0_raw_angle_median_abs_degrees": float(np.median(b0_baseline_shift)),
            "m1_official_raw_vs_official_angle_error_difference_median_degrees": float(
                np.median([
                    abs(record["m1_raw"]["median_absolute_target_error_degrees"] - record["m1_official"]["median_absolute_target_error_degrees"])
                    for record in main_records
                ])
            ),
            "m1_official_recanonicalized_error_gain_median_degrees": float(np.median(recanonical_gain)),
            "m1_official_recanonicalized_error_gain_min_degrees": float(min(recanonical_gain)),
            "m1_official_recanonicalized_error_gain_max_degrees": float(max(recanonical_gain)),
            "all_noise_and_m0_flags_true": all(
                all(record["noise_and_m0_flags"].values()) for record in main_records
            ),
            "all_video_paths_exist_in_record": all(bool(record["video_paths"]) for record in main_records),
        },
        "records": main_records,
        "selected_waveforms": waveforms,
        "interpretation": {
            "canonical_y_vs_travel": "由于公式同时使用 heading 方向和 heading×up 的两个水平分量，水平偏航只改变坐标分解，不改变角度值；travel 主要影响低速帧有效性。",
            "b0_baseline": "B0 从 M0 的速度通道恢复 T+1 再最终化；其与 M0 raw 的角度差是正式 B0 参照的已知混杂，不能事后改写主指标。",
            "postprocessing": "M1 raw/official 直接输出与把同一张量重新经过 B0 最终化的派生角度不同；这说明冗余速度/姿态通道不一致可能造成材料性影响，但派生值不是正式主结果。",
            "noise_and_input": "既有 run metrics 的 z0、M0 raw、M0 official 位标志均通过；输入为纯文本且未启用参考动作。",
            "uncertainty": "本审计只能证明表示、基线和后处理的数值关系，不能仅凭这些张量确定 M1 欠校正的唯一因果机制。",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--m1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        metrics_path=args.metrics,
        input_path=args.input,
        m0_root=args.m0_root,
        m1_root=args.m1_root,
        output_path=args.output,
    )
    print(json.dumps({"status": result["status"], "record_count": len(result["records"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
