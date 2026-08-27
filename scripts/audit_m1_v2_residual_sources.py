"""Audit endpoint evidence for M1-v2 residual sources without resampling.

The real pilot stores final raw and official tensors, but no per-step sampler
trace.  This script therefore quantifies the endpoint effect of official
smoothing and explicitly labels Euler drift and later model re-prediction as
unknown rather than inferring them from the final tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_m1_pilot import load_norm, tensor_sha256
from scripts.evaluate_m1_v2_real_pilot import channel_consistency


CHANNEL_KEYS = (
    "joint_position_velocity_median_m",
    "root_translation_velocity_median_m",
    "root_rotation_velocity_median_degrees",
)


def summarize_source_contribution(raw: dict[str, float], official: dict[str, float]) -> dict[str, object]:
    """Summarize what can and cannot be learned from raw/official endpoints."""

    reductions = {key: float(raw[key] - official[key]) for key in CHANNEL_KEYS}
    all_reduced = all(official[key] <= raw[key] for key in CHANNEL_KEYS)
    return {
        "official_smoothing_reductions": reductions,
        "official_smoothing_reduces_all_three_medians": all_reduced,
        "euler_drift_attribution": "UNKNOWN_NO_STEP_TRACE",
        "model_reprediction_attribution": "UNKNOWN_NO_STEP_TRACE",
        "interpretation": (
            "OFFICIAL_SMOOTHING_REDUCES_ENDPOINT_RESIDUALS_BUT_ROOT_CAUSE_UNKNOWN"
            if all_reduced
            else "SMOOTHING_NOT_REDUCING_ALL_CHANNELS"
        ),
    }


def _median(values: list[float]) -> float:
    return float(torch.tensor(values, dtype=torch.float64).median().item())


def _find_step_traces(run: Path) -> list[str]:
    candidates = []
    for path in run.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if any(token in name for token in ("step_trace", "step-trace", "sampler_trace", "per_step", "per-step")):
            candidates.append(str(path))
    return sorted(candidates)


def audit(root: Path, *, output: Path, markdown: Path) -> dict[str, object]:
    mean = torch.from_numpy(np.load(ROOT / "data/meta_info/mean.npy")).float()
    std = torch.from_numpy(np.load(ROOT / "data/meta_info/std.npy")).float()
    report: dict[str, object] = {
        "status": "ENDPOINT_ONLY_SOURCE_AUDIT",
        "source_attribution_status": "PARTIAL_OFFICIAL_SMOOTHING_QUANTIFIED_EULER_REPREDICTION_UNKNOWN",
        "protocol_unchanged": True,
        "no_model_rerun": True,
        "no_tuning": True,
        "no_holdout": True,
        "observation_boundary": "final M1 raw and official tensors only; no per-step x0/velocity/Euler trace was saved",
        "runs": {},
    }
    for label in ("plus5_realpilot01", "minus5_realpilot01"):
        run = root / label
        m1_dir = run / "m1_artifacts" / "batch_000"
        raw_norm = torch.load(m1_dir / "m1_raw_norm_batch.pt", map_location="cpu", weights_only=True)
        official_norm = torch.load(m1_dir / "m1_official_norm_batch.pt", map_location="cpu", weights_only=True)
        raw = load_norm(m1_dir / "m1_raw_norm_batch.pt", mean, std)
        official = load_norm(m1_dir / "m1_official_norm_batch.pt", mean, std)
        raw_consistency = [channel_consistency(item) for item in raw]
        official_consistency = [channel_consistency(item) for item in official]
        raw_summary = {key: _median([item[key] for item in raw_consistency]) for key in CHANNEL_KEYS}
        official_summary = {key: _median([item[key] for item in official_consistency]) for key in CHANNEL_KEYS}
        source_summary = summarize_source_contribution(raw_summary, official_summary)
        delta = official - raw
        traces = _find_step_traces(run)
        report["runs"][label] = {
            "path": str(run),
            "raw_sha256": tensor_sha256(raw_norm),
            "official_sha256": tensor_sha256(official_norm),
            "raw_channel_consistency_median_across_samples": raw_summary,
            "official_channel_consistency_median_across_samples": official_summary,
            "source_summary": source_summary,
            "official_minus_raw_rms_per_sample": [float(torch.sqrt(item.square().mean()).item()) for item in delta],
            "official_minus_raw_max_abs_per_sample": [float(item.abs().max().item()) for item in delta],
            "step_trace_files": traces,
            "per_step_trace_available": bool(traces),
            "raw_consistency_records": raw_consistency,
            "official_consistency_records": official_consistency,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# M1-v2 残差来源只读审计", "",
        "状态：**ENDPOINT_ONLY_SOURCE_AUDIT**。本审计只读取真实 pilot 的最终 M1 raw/official 张量，没有重跑模型、调参或运行留出集。", "",
        "## 能够回答什么", "",
        "`official` 是对 `raw` 的后处理输出，因此可以量化 official 平滑对最终通道残差的影响；但没有保存每个采样步的 x0、速度、Euler 更新前后状态，不能仅凭终点张量区分 Euler 漂移与后续模型重预测。两者均明确记为 UNKNOWN。", "",
        "| 条件 | 输出 | 关节位置—速度中位误差 | 根平移—速度中位误差 | 根旋转—速度中位误差 |", "|---|---|---:|---:|---:|",
    ]
    for label, item in report["runs"].items():
        for output_name, key in (("raw", "raw_channel_consistency_median_across_samples"), ("official", "official_channel_consistency_median_across_samples")):
            values = item[key]
            lines.append(f"| {label} | {output_name} | {values['joint_position_velocity_median_m']:.6f} m | {values['root_translation_velocity_median_m']:.6f} m | {values['root_rotation_velocity_median_degrees']:.6f}° |")
        source = item["source_summary"]
        lines.append(f"| {label} | official 相对 raw 的中位数变化 | {source['official_smoothing_reductions']['joint_position_velocity_median_m']:.6f} m | {source['official_smoothing_reductions']['root_translation_velocity_median_m']:.6f} m | {source['official_smoothing_reductions']['root_rotation_velocity_median_degrees']:.6f}° |")
    lines += [
        "", "## 归因结论", "",
        "- official 平滑对端点残差的降低是可观测的，但这只说明后处理改变了最终表示，不能证明采样内 M1-v2 已在每一步保持一致。",
        "- Euler 漂移：`UNKNOWN_NO_STEP_TRACE`。",
        "- 后续模型重预测造成的残差：`UNKNOWN_NO_STEP_TRACE`。",
        "- 最小补充观测：每个启用采样步保存 sigma/timestep、引导前 x0、引导后 x0、重组后的 x0、返回模型速度、Euler 更新后的 x，以及三类通道残差；仍使用同一固定 z0，另写新诊断目录，不覆盖本 pilot。",
        "", "当前不能据此修改 M1-v2、宣称 M1 通过或开始新留出集；下一步是先决定是否需要按上述最小观测补充一次独立的诊断运行。", "",
    ]
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results/phase3/m1_v2_pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "diagnostics/phase3/m1_v2/residual_source_audit/m1_v2_residual_source_audit.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "diagnostics/phase3/m1_v2/residual_source_audit/M1_V2_RESIDUAL_SOURCE_AUDIT.md")
    args = parser.parse_args()
    print(json.dumps(audit(args.root, output=args.output, markdown=args.markdown), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
