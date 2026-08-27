#!/usr/bin/env python3
"""Audit the original ViMoGen 276-D representation.

This command is read-only with respect to model results.  It loads frozen M0
Raw/Official outputs, twenty same-source reference motions, and all MBench
reference motions, then writes a new diagnostic directory with scalar tables,
curves, figures, and a Chinese Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from smplx import SMPLX  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.representation_consistency import (  # noqa: E402
    JOINT_NAMES,
    bootstrap_cluster_curve,
    bootstrap_cluster_stat,
    compute_sequence_metrics,
    interpolate_curve,
    summarize_records,
)


SEED = 20260823
BOOTSTRAP_REPETITIONS = 2000
SCALAR_KEYS = (
    "speed_residual_mean_m_per_frame",
    "speed_residual_relative_to_direct_step",
    "trajectory_drift_final_m",
    "trajectory_drift_auc_m",
    "trajectory_drift_slope_m_per_frame",
    "trajectory_drift_final_over_body_scale",
    "trajectory_drift_auc_over_body_scale",
    "fk_absolute_mean_m",
    "fk_absolute_over_body_scale",
    "fk_relative_pelvis_mean_m",
    "fk_relative_pelvis_over_body_scale",
    "root_translation_speed_residual_mean",
    "root_rotation_speed_residual_degrees_mean",
    "root_rotation_integrated_drift_degrees_mean",
)
CURVE_KEYS = (
    "speed_residual_m_per_frame",
    "trajectory_drift_m",
    "fk_absolute_m",
    "fk_relative_pelvis_m",
)


def _load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor at {path}, got {type(value)!r}")
    if value.ndim != 3 or value.shape[-1] != 276:
        raise ValueError(f"expected [B,T,276] at {path}, got {tuple(value.shape)}")
    return value.float()


def _load_m0_seed(seed_root: Path, filename: str, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    paths = sorted((seed_root / "artifacts").glob(f"batch_*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"no {filename} below {seed_root}")
    batches = [_load_tensor(path) for path in paths]
    value = torch.cat(batches, dim=0)
    if value.shape[0] != 20:
        raise ValueError(f"expected 20 samples for {seed_root}, got {value.shape[0]}")
    return value * std[None, None, :] + mean[None, None, :]


def _manifest_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != 20:
        raise ValueError(f"expected 20 manifest items in {path}")
    return items


def _load_source_reference(root: Path, items: list[dict[str, Any]]) -> list[tuple[str, torch.Tensor]]:
    result = []
    for item in items:
        sample_id = str(item["id"])
        path = root / "data/ViMoGen-228K" / item["motion_path"]
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[-1] != 276:
            raise ValueError(f"source motion {path} is not [T,276]: {type(value)!r}, {getattr(value, 'shape', None)}")
        result.append((sample_id, value.float()))
    return result


def _load_mbench(root: Path) -> list[tuple[str, torch.Tensor]]:
    result: list[tuple[str, torch.Tensor]] = []
    # ``data/mbench`` is a symlink in the clean server checkout; ``Path.rglob``
    # does not descend through a symlinked directory on all Python versions.
    paths = []
    for directory, _, filenames in os.walk(root, followlinks=True):
        paths.extend(Path(directory) / name for name in filenames if name.endswith(".pt"))
    for path in sorted(paths):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        value = payload.get("motion") if isinstance(payload, dict) else None
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim != 2 or value.shape[-1] != 276:
            raise ValueError(f"MBench motion {path} has shape {tuple(value.shape)}")
        result.append((str(path.relative_to(root)), value.float()))
    if len(result) != 450:
        raise ValueError(f"expected 450 MBench motion tensors, found {len(result)}")
    return result


def _run_one(
    motion: torch.Tensor,
    *,
    model: SMPLX,
    device: torch.device,
    sample_id: str,
    method: str,
    stage: str,
    source_kind: str,
) -> dict[str, Any]:
    result = compute_sequence_metrics(
        motion.to(device),
        model=model,
        sample_id=sample_id,
        method=method,
        output_stage=stage,
        source_kind=source_kind,
    )
    # Keep scalar metadata and curves on the host for deterministic JSON/NPZ
    # serialization; the input tensor and body model remain untouched.
    return result


def _record_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: record.get(key)
        for key in ("sample_id", "seed_index", "method", "output_stage", "source_kind", "frame_count", "body_scale_mean_bone_m")
    }
    for key in SCALAR_KEYS:
        row[key] = record.get(key)
    return row


def _save_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    rows = [_record_row(record) for record in records]
    fields = list(rows[0]) if rows else ["sample_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"record_count": len(records)}
    for key in SCALAR_KEYS:
        if all(key in record and record[key] is not None for record in records):
            summary[key] = {
                "descriptive": summarize_records(records, key),
                "bootstrap_cluster_median": bootstrap_cluster_stat(
                    records, key, repetitions=BOOTSTRAP_REPETITIONS, seed=SEED
                ),
            }
    for key in CURVE_KEYS:
        if all(key in record["curves"] for record in records):
            summary[f"{key}_curve"] = bootstrap_cluster_curve(
                records, key, repetitions=BOOTSTRAP_REPETITIONS, seed=SEED
            )
    return summary


def _paired_effects(raw: list[dict[str, Any]], official: list[dict[str, Any]]) -> dict[str, Any]:
    raw_by_key = {(str(row["sample_id"]), int(row["seed_index"])): row for row in raw}
    official_by_key = {(str(row["sample_id"]), int(row["seed_index"])): row for row in official}
    keys = sorted(raw_by_key)
    result: dict[str, Any] = {"record_count": len(keys), "metrics": {}}
    for metric in SCALAR_KEYS:
        effects = []
        for key in keys:
            if raw_by_key[key].get(metric) is None or official_by_key[key].get(metric) is None:
                continue
            effects.append({
                "sample_id": key[0],
                "effect": float(official_by_key[key][metric]) - float(raw_by_key[key][metric]),
            })
        if effects:
            result["metrics"][metric] = {
                "official_minus_raw_descriptive": summarize_records(
                    [{"value": row["effect"]} for row in effects], "value"
                ),
                "bootstrap_cluster_median": bootstrap_cluster_stat(
                    [{"sample_id": row["sample_id"], "effect": row["effect"]} for row in effects],
                    "effect", repetitions=BOOTSTRAP_REPETITIONS, seed=SEED,
                ),
            }
    return result


def _boxplot(ax: Any, groups: list[tuple[str, list[float]]], ylabel: str, *, log_scale: bool = False) -> None:
    labels = [item[0] for item in groups]
    values = [item[1] for item in groups]
    ax.boxplot(values, labels=labels, showfliers=False)
    for index, value in enumerate(values, start=1):
        jitter = np.linspace(-0.12, 0.12, max(1, len(value)))
        ax.scatter(np.full(len(value), index) + jitter, value, s=7, alpha=0.25, color="black")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    if log_scale and all(np.all(np.asarray(value) > 0) for value in values):
        ax.set_yscale("log")


def _plot_outputs(output: Path, groups: dict[str, list[dict[str, Any]]], records: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    order = ["reference_source", "reference_mbench", "m0_raw", "m0_official"]
    labels = ["Source-20", "MBench-450", "M0-Raw", "M0-Official"]

    metric_groups = lambda key: [(label, [float(row[key]) for row in groups[name]]) for name, label in zip(order, labels)]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    _boxplot(axes[0], metric_groups("speed_residual_mean_m_per_frame"), "E_speed (m/frame)")
    _boxplot(axes[1], metric_groups("speed_residual_relative_to_direct_step"), "E_speed / direct step")
    figure.savefig(output / "speed_residual.png", dpi=180)
    figure.savefig(output / "speed_residual.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, group_name, title in zip(axes, ("m0_raw", "m0_official"), ("M0 Raw", "M0 Official")):
        rows = groups[group_name]
        for row in rows:
            curve = np.asarray(row["curves"]["trajectory_drift_m"], dtype=float)
            ax.plot(np.arange(curve.size), curve, color="#4472c4", alpha=0.08, linewidth=0.7)
        curve_summary = bootstrap_cluster_curve(rows, "trajectory_drift_m", repetitions=BOOTSTRAP_REPETITIONS, seed=SEED)
        x = np.arange(len(curve_summary["median"]))
        ax.plot(x, curve_summary["median"], color="#c00000", linewidth=2, label="median")
        ax.fill_between(x, curve_summary["ci95_low"], curve_summary["ci95_high"], color="#c00000", alpha=0.18, label="95% CI")
        ax.set_title(title)
        ax.set_xlabel("frame")
        ax.set_ylabel("D_t (m)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    figure.savefig(output / "integrated_drift_curves.png", dpi=180)
    figure.savefig(output / "integrated_drift_curves.pdf")
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    colors = {"reference_source": "#70ad47", "reference_mbench": "#a5a5a5", "m0_raw": "#4472c4", "m0_official": "#c00000"}
    display_names = {
        "reference_source": "Source-20",
        "reference_mbench": "MBench-450",
        "m0_raw": "M0 Raw",
        "m0_official": "M0 Official",
    }
    for group_name in order:
        rows = groups[group_name]
        curve_summary = bootstrap_cluster_curve(rows, "trajectory_drift_m", repetitions=BOOTSTRAP_REPETITIONS, seed=SEED)
        x = np.linspace(0.0, 100.0, len(curve_summary["median"]))
        ax.plot(x, curve_summary["median"], color=colors[group_name], linewidth=2, label=display_names[group_name])
        ax.fill_between(x, curve_summary["ci95_low"], curve_summary["ci95_high"], color=colors[group_name], alpha=0.10)
    ax.set_xlabel("normalized progress (%)")
    ax.set_ylabel("D_t (m)")
    ax.set_title("Direct position vs integrated velocity: normalized time")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    figure.savefig(output / "integrated_drift_normalized_reference.png", dpi=180)
    figure.savefig(output / "integrated_drift_normalized_reference.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    _boxplot(axes[0], metric_groups("fk_absolute_mean_m"), "FK absolute mean joint error (m)")
    _boxplot(axes[1], metric_groups("fk_relative_pelvis_mean_m"), "FK pelvis-relative mean joint error (m)")
    figure.savefig(output / "fk_error.png", dpi=180)
    figure.savefig(output / "fk_error.pdf")
    plt.close(figure)

    official = groups["m0_official"]
    speed_heat = np.stack([
        np.asarray(row["curves"]["speed_joint_residual_m_per_frame"], dtype=float).mean(axis=0)
        for row in official
    ])
    fk_heat = np.stack([
        np.asarray(row["curves"]["fk_joint_error_m"], dtype=float).mean(axis=0)
        for row in official
    ])
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    for ax, matrix, title, colorbar in (
        (axes[0], speed_heat, "M0 Official: joint speed residual", "m/frame"),
        (axes[1], fk_heat, "M0 Official: joint FK error", "m"),
    ):
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
        ax.set_yticks(np.arange(0, len(JOINT_NAMES), 2), [JOINT_NAMES[i] for i in range(0, len(JOINT_NAMES), 2)])
        ax.set_xlabel("sample index")
        ax.set_ylabel("joint")
        ax.set_title(title)
        figure.colorbar(image, ax=ax, label=colorbar)
    figure.savefig(output / "joint_heatmaps.png", dpi=180)
    figure.savefig(output / "joint_heatmaps.pdf")
    plt.close(figure)

    worst = sorted(official, key=lambda row: float(row["trajectory_drift_final_m"]), reverse=True)[:3]
    figure, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for row in worst:
        curve = np.asarray(row["curves"]["trajectory_drift_m"], dtype=float)
        ax.plot(curve, linewidth=1.8, label=f"{row['sample_id']} / seed {row['seed_index']}")
    ax.set_xlabel("frame")
    ax.set_ylabel("D_t (m)")
    ax.set_title("M0 Official: three largest final drifts")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    figure.savefig(output / "worst_case_drift.png", dpi=180)
    figure.savefig(output / "worst_case_drift.pdf")
    plt.close(figure)


def _write_report(path: Path, summary: dict[str, Any], groups: dict[str, list[dict[str, Any]]]) -> None:
    def median(group: str, key: str) -> float:
        return float(summary["groups"][group][key]["descriptive"]["median"])

    raw_speed = median("m0_raw", "speed_residual_mean_m_per_frame")
    off_speed = median("m0_official", "speed_residual_mean_m_per_frame")
    ref_speed = median("reference_mbench", "speed_residual_mean_m_per_frame")
    raw_fk = median("m0_raw", "fk_relative_pelvis_mean_m")
    off_fk = median("m0_official", "fk_relative_pelvis_mean_m")
    ref_fk = median("reference_mbench", "fk_relative_pelvis_mean_m")
    raw_slope = summary["groups"]["m0_raw"]["trajectory_drift_slope_m_per_frame"]["bootstrap_cluster_median"]
    off_slope = summary["groups"]["m0_official"]["trajectory_drift_slope_m_per_frame"]["bootstrap_cluster_median"]
    lines = [
        "# ViMoGen 原始 276 维表示一致性审计",
        "",
        "本报告只审计未加入控制的 M0。Raw 是模型直接输出，Official 是官方后处理输出；同源参考和 MBench 参考仅用于表示校准，不是生成动作的逐帧真值。",
        "",
        "## 数据覆盖",
        "",
        f"- M0 Raw：{len(groups['m0_raw'])} 条；M0 Official：{len(groups['m0_official'])} 条。",
        f"- 同源参考：{len(groups['reference_source'])} 条；MBench 参考：{len(groups['reference_mbench'])} 条。",
        "- 主时间单位为帧；速度通道按米/帧的前向位移处理。前两项不使用最后一行速度。",
        "",
        "## 关键结果（中位数）",
        "",
        "| 数据 | E_speed (m/frame) | E_speed/直接步长 | D_final (m) | D_slope (m/frame) | FK绝对均值 (m) | FK骨盆相对均值 (m) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, label in (("reference_source", "同源参考20"), ("reference_mbench", "MBench参考450"), ("m0_raw", "M0 Raw"), ("m0_official", "M0 Official")):
        lines.append(
            f"| {label} | {median(group, 'speed_residual_mean_m_per_frame'):.6g} | "
            f"{median(group, 'speed_residual_relative_to_direct_step'):.6g} | "
            f"{median(group, 'trajectory_drift_final_m'):.6g} | "
            f"{median(group, 'trajectory_drift_slope_m_per_frame'):.6g} | "
            f"{median(group, 'fk_absolute_mean_m'):.6g} | "
            f"{median(group, 'fk_relative_pelvis_mean_m'):.6g} |"
        )
    lines += [
        "",
        "## 初步解释规则",
        "",
        f"- M0 Raw 的位置—速度中位残差为 `{raw_speed:.6g}`，Official 为 `{off_speed:.6g}`；MBench 参考中位数为 `{ref_speed:.6g}`。Raw/Official 与参考的差异应结合 `paired_raw_vs_official` 中的自举区间判断，不能只看单个绝对数。",
        f"- M0 Raw 的积分轨迹斜率自举区间为 `[{raw_slope['ci95_low']:.6g}, {raw_slope['ci95_high']:.6g}]`，Official 为 `[{off_slope['ci95_low']:.6g}, {off_slope['ci95_high']:.6g}]`。区间整体高于零时，支持“直接位置与速度积分轨迹随时间分离”。",
        f"- M0 Raw 的骨盆相对 FK 中位误差为 `{raw_fk:.6g}`，Official 为 `{off_fk:.6g}`，MBench 参考为 `{ref_fk:.6g}`。绝对 FK 误差还会受到中性 SMPL-X、零体型参数和坐标规范化影响；只有骨盆相对误差也明显高于参考，才支持身体旋转与直接关节位置不属于同一姿态。",
        "- 本轮不设置未经验证的硬性通过阈值；结论依据原始量纲、相对参考倍数、Raw−Official 配对差值和时间曲线增长共同给出。",
        "",
        "## 产物",
        "",
        "- `summary.json`：完整机器可读汇总、自举区间和配对差值。",
        "- `records.csv`：每条轨迹的标量指标。",
        "- `curves.npz`：逐帧和逐关节曲线。",
        "- `figures/`：单步残差、积分漂移、FK 误差、关节热图和最严重样本图。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"

    mean = torch.from_numpy(np.load(args.root / "data/meta_info/mean.npy")).float()
    std = torch.from_numpy(np.load(args.root / "data/meta_info/std.npy")).float()
    if mean.shape != (276,) or std.shape != (276,):
        raise ValueError(f"expected [276] statistics, got {mean.shape}, {std.shape}")
    items = _manifest_items(args.manifest)
    ids = [str(item["id"]) for item in items]
    source = _load_source_reference(args.root, items)
    mbench = _load_mbench(args.mbench)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SMPLX(
        model_path=str(args.root / "data/body_models/smplx"),
        gender="neutral", use_pca=False, num_betas=10, batch_size=1,
    ).eval().to(device)

    records: list[dict[str, Any]] = []
    for seed_index in (0, 1, 2):
        seed_root = args.m0 / f"seed_{seed_index:03d}"
        for stage, filename in (("raw", "m0_raw_norm_batch.pt"), ("official", "m0_official_norm_batch.pt")):
            batch = _load_m0_seed(seed_root, filename, mean, std)
            for index, sample_id in enumerate(ids):
                record = _run_one(
                    batch[index], model=model, device=device,
                    sample_id=sample_id, method="m0", stage=stage, source_kind="generated",
                )
                record["seed_index"] = seed_index
                records.append(record)
    for sample_id, motion in source:
        record = _run_one(
            motion, model=model, device=device,
            sample_id=sample_id, method="reference", stage="source", source_kind="reference_source",
        )
        record["seed_index"] = -1
        records.append(record)
    for sample_id, motion in mbench:
        record = _run_one(
            motion, model=model, device=device,
            sample_id=sample_id, method="reference", stage="mbench", source_kind="reference_mbench",
        )
        record["seed_index"] = -1
        records.append(record)

    groups = {
        "m0_raw": [row for row in records if row["source_kind"] == "generated" and row["output_stage"] == "raw"],
        "m0_official": [row for row in records if row["source_kind"] == "generated" and row["output_stage"] == "official"],
        "reference_source": [row for row in records if row["source_kind"] == "reference_source"],
        "reference_mbench": [row for row in records if row["source_kind"] == "reference_mbench"],
    }
    expected = {"m0_raw": 60, "m0_official": 60, "reference_source": 20, "reference_mbench": 450}
    for name, count in expected.items():
        if len(groups[name]) != count:
            raise AssertionError(f"{name}: expected {count}, found {len(groups[name])}")

    summary: dict[str, Any] = {
        "schema_version": "vimogen-representation-consistency-v2",
        "status": "VERIFIED_READ_ONLY_AUDIT",
        "protocol": {
            "generated_scope": "nonturning_v10_user_approved",
            "generated_counts": {"texts": 20, "seeds": [0, 1, 2], "raw": 60, "official": 60},
            "reference_counts": {"same_source": 20, "mbench_motion": 450},
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": SEED,
            "velocity_units": "meters_per_frame_forward_difference",
            "exclude_last_velocity_row": True,
            "fk_model": "neutral_smplx_zero_betas_first_22_joints",
            "device": str(device),
        },
        "groups": {name: _group_summary(rows) for name, rows in groups.items()},
        "paired_raw_vs_official": _paired_effects(groups["m0_raw"], groups["m0_official"]),
    }
    summary["reference_comparisons"] = {}
    for generated in ("m0_raw", "m0_official"):
        summary["reference_comparisons"][generated] = {}
        for key in SCALAR_KEYS:
            if key not in summary["groups"][generated] or key not in summary["groups"]["reference_mbench"]:
                continue
            g = summary["groups"][generated][key]["descriptive"]["median"]
            r = summary["groups"]["reference_mbench"][key]["descriptive"]["median"]
            summary["reference_comparisons"][generated][key] = {
                "generated_median": g,
                "reference_median": r,
                "generated_over_reference": None if abs(r) < 1e-8 else g / abs(r),
                "reference_is_zero_within_1e-8": bool(abs(r) < 1e-8),
            }

    _save_records_csv(output / "records.csv", records)
    curve_payload: dict[str, Any] = {}
    for name, rows in groups.items():
        for key in CURVE_KEYS:
            if all(key in row["curves"] for row in rows):
                curve_payload[f"{name}__{key}"] = np.stack([interpolate_curve(row["curves"][key]) for row in rows])
        curve_payload[f"{name}__sample_id"] = np.asarray([str(row["sample_id"]) for row in rows])
    np.savez_compressed(output / "curves.npz", **curve_payload)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _plot_outputs(figures, groups, records)
    _write_report(output / "REPORT.md", summary, groups)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--m0", type=Path, default=ROOT / "results/phase3/devset_baselines/nonturning_v10_user_approved/m0_v2")
    parser.add_argument("--manifest", type=Path, default=ROOT / "diagnostics/phase3/devset_frozen/nonturning_v10_user_approved/frozen_manifest.json")
    parser.add_argument("--mbench", type=Path, default=ROOT / "data/mbench")
    parser.add_argument("--output", type=Path, default=ROOT / "diagnostics/phase1/vimogen_representation_consistency_v2")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({
        "status": summary["status"],
        "groups": {name: value["record_count"] for name, value in summary["groups"].items()},
        "output": str(args.output),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
