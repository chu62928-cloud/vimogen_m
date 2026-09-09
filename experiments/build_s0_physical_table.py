#!/usr/bin/env python3
"""Build a dual-track, numeric S0 physical-closure table.

The historical v1 physical result remains available as the strict audit. The
v2 result is reported alongside it and uses the independently frozen v2
decision profile. Raw per-sequence values are preserved in both JSON and a
readable Markdown appendix; no binary pass/fail value replaces the metrics.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRELIMINARY = ROOT / "artifacts/table1_s0_preliminary/table1_s0_summary.json"
DEFAULT_PHYSICAL = ROOT / "results/phase9/pelvis_m1_m7/s0_physical_evaluation_v3/summary.json"
DEFAULT_OUTPUT = ROOT / "results/phase9/pelvis_m1_m7/table1_s0_physical_v2"

PHYSICAL_METRICS = (
    "penetration_p95_mm",
    "penetration_max_mm",
    "penetration_frame_rate",
    "contact_tangent_speed_p95_mm_per_frame",
    "contact_tangent_speed_max_mm_per_frame",
    "total_slide_distance_mm",
    "max_segment_slide_distance_mm",
    "contact_height_p95_mm",
    "contact_height_max_mm",
    "floating_frame_rate",
    "support_height_error_p95_mm",
)
METRIC_LABELS = {
    "penetration_p95_mm": "穿地 P95 (mm)",
    "penetration_max_mm": "穿地最大 (mm)",
    "penetration_frame_rate": "穿地帧率",
    "contact_tangent_speed_p95_mm_per_frame": "脚滑速度 P95 (mm/帧)",
    "contact_tangent_speed_max_mm_per_frame": "脚滑速度最大 (mm/帧)",
    "total_slide_distance_mm": "总脚滑距离 (mm)",
    "max_segment_slide_distance_mm": "最长连续脚滑 (mm)",
    "contact_height_p95_mm": "接触高度 P95 (mm)",
    "contact_height_max_mm": "接触高度最大 (mm)",
    "floating_frame_rate": "悬空帧率",
    "support_height_error_p95_mm": "支撑高度误差 P95 (mm)",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_source(row: Mapping[str, Any], profile: str) -> Mapping[str, Any]:
    value = row.get(profile)
    if isinstance(value, Mapping):
        return value
    # Older fixtures and pre-v2 artifacts only have the legacy ``physical``
    # field. Treat it as v1 for compatibility.
    return row.get("physical", {})


def _profile_summary(
    rows: list[Mapping[str, Any]], profile: str
) -> dict[str, Any]:
    sources = [_profile_source(row, profile) for row in rows]
    statuses = Counter(str(source.get("status", "UNKNOWN")) for source in sources)
    thresholds = dict(sources[0].get("thresholds") or {}) if sources else {}
    hard_metrics = sources[0].get("hard_metrics") if sources else None
    if hard_metrics is None:
        hard_metrics = list(PHYSICAL_METRICS)

    metrics: dict[str, Any] = {}
    for name in PHYSICAL_METRICS:
        metric_rows = [
            source.get("per_sequence", [{}])[0]
            for source in sources
            if source.get("per_sequence")
        ]
        values = [
            float(value)
            for value in (row.get(name) for row in metric_rows)
            if value is not None and math.isfinite(float(value))
        ]
        threshold = thresholds.get(name)
        exceeds = (
            sum(value > float(threshold) for value in values)
            if threshold is not None
            else None
        )
        if values:
            q1, median, q3, p95 = np.percentile(np.asarray(values), [25, 50, 75, 95])
            maximum = max(values)
            display = f"{median:.2f} [{q1:.2f}, {q3:.2f}]"
        else:
            q1 = median = q3 = p95 = maximum = None
            display = "NA"
        metrics[name] = {
            "label": METRIC_LABELS[name],
            "n": len(values),
            "q1": q1,
            "median": median,
            "q3": q3,
            "p95": p95,
            "max": maximum,
            "threshold": threshold,
            "exceed_count": exceeds,
            "hard_gate": name in hard_metrics,
            "display": display,
        }
    return {
        "status_counts": dict(sorted(statuses.items())),
        "pass_count": statuses.get("EVALUATED_PASS", 0),
        "fail_count": statuses.get("EVALUATED_FAIL", 0),
        "not_evaluated_count": statuses.get("NOT_EVALUATED", 0),
        "thresholds": thresholds,
        "hard_metrics": sorted(str(value) for value in hard_metrics),
        "metric_stats": metrics,
    }


def _sequence_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in rows:
        entry: dict[str, Any] = {
            key: source.get(key)
            for key in ("run_id", "method", "seed", "sample_id", "dose_deg", "paired_m0_id")
        }
        entry["physical_reference_status"] = source.get("physical_reference_status")
        entry["physical_v1"] = source.get("physical_v1", source.get("physical"))
        entry["physical_v2"] = source.get("physical_v2")
        for profile in ("physical_v1", "physical_v2"):
            value = entry.get(profile) or {}
            sequence = value.get("per_sequence", [{}])
            entry[profile] = {
                "status": value.get("status"),
                "physical_pass": value.get("physical_pass"),
                "physical_fail_reasons": sequence[0].get("physical_fail_reasons", [])
                if sequence
                else [],
                "metrics": {
                    name: sequence[0].get(name)
                    for name in PHYSICAL_METRICS
                }
                if sequence
                else {},
                "marker_metrics": sequence[0].get("marker_metrics", {})
                if sequence
                else {},
            }
        result.append(entry)
    return result


def build_rows(
    preliminary: Mapping[str, Any], physical: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if physical.get("status") != "S0_PHYSICAL_EVALUATED":
        raise ValueError("physical summary must be fully evaluated")
    physical_records = list(physical.get("records", []))
    if len(physical_records) != 84:
        raise ValueError("physical summary must contain exactly 84 S0 records")
    methods = {str(row["method"]): dict(row) for row in preliminary.get("methods", [])}
    if set(methods) != {f"M{index}" for index in range(1, 8)}:
        raise ValueError("preliminary summary must contain M1 through M7")
    result: list[dict[str, Any]] = []
    for method in methods:
        source = [row for row in physical_records if row.get("method") == method]
        if len(source) != 12:
            raise ValueError(f"{method} must have 12 physical records")
        row = dict(methods[method])
        row["physical_v1"] = _profile_summary(source, "physical_v1")
        row["physical_v2"] = _profile_summary(source, "physical_v2")
        # Keep the old field as an explicit v1 alias for downstream readers.
        row["physical"] = row["physical_v1"]
        failures = Counter(
            reason
            for item in source
            for reason in (
                item.get("physical_v1", item.get("physical", {}))
                .get("per_sequence", [{}])[0]
                .get("physical_fail_reasons", [])
            )
        )
        row["physical"]["failure_reason_counts"] = dict(sorted(failures.items()))
        result.append(row)
    return result


def _metric_cell(summary: Mapping[str, Any], name: str) -> str:
    metric = summary["metric_stats"].get(name, {})
    display = metric.get("display", "NA")
    threshold = metric.get("threshold")
    count = metric.get("exceed_count")
    if threshold is None:
        return display
    gate = "硬门" if metric.get("hard_gate", True) else "描述"
    return f"{display}；阈值 {float(threshold):.2f}；超阈 {count}/{metric.get('n', 0)}（{gate}）"


def _markdown(
    rows: list[Mapping[str, Any]],
    threshold_version: str,
    threshold_version_v2: str,
    per_sequence_name: str,
) -> str:
    lines = [
        "# Table 1：S0 骨盆控制与物理闭环（v1/v2 双轨）",
        "",
        "## Table 1A：控制、内容和物理通过数",
        "",
        "| 方法 | 角度 MAE（°） | 角度 P95（°） | 斜率 | 角度通过 | MPJPE（mm） | 物理 v1 | 物理 v2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {mae} | {p95} | {slope:.3f} | {angle}/{n} | {mpjpe} | {v1}/{n} | {v2}/{n} |".format(
                label=row["label"],
                mae=row["angle_mae_deg"]["display"],
                p95=row["angle_p95_deg"]["display"],
                slope=row["dose_response_slope"],
                angle=row["sequence_angle_pass_count"],
                mpjpe=row["mpjpe_vs_m0_mm"]["display"],
                v1=row["physical_v1"]["pass_count"],
                v2=row["physical_v2"]["pass_count"],
                n=row["sequence_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Table 1B：逐方法物理指标",
            "",
            "每项为中位数 `[Q1, Q3]`；同时列出冻结阈值和超阈值条数。接触高度在 v2 中保留为描述性指标，不再重复计入硬门。",
            "",
        ]
    )
    groups = [
        (
            "穿地与脚滑",
            (
                "penetration_p95_mm",
                "penetration_max_mm",
                "penetration_frame_rate",
                "contact_tangent_speed_p95_mm_per_frame",
                "contact_tangent_speed_max_mm_per_frame",
                "total_slide_distance_mm",
                "max_segment_slide_distance_mm",
            ),
        ),
        (
            "接触高度、悬空与支撑",
            (
                "contact_height_p95_mm",
                "contact_height_max_mm",
                "floating_frame_rate",
                "support_height_error_p95_mm",
            ),
        ),
    ]
    for title, metrics in groups:
        lines.extend(
            [
                f"### {title}",
                "",
                "| 方法 | " + " | ".join(METRIC_LABELS[name] for name in metrics) + " |",
                "|---|" + "---:|" * len(metrics),
            ]
        )
        for row in rows:
            values = [_metric_cell(row["physical_v2"], name) for name in metrics]
            lines.append(f"| {row['method']} | " + " | ".join(values) + " |")
        lines.append("")
    lines.extend(
        [
            "## 阈值协议",
            "",
            f"- v1：`{threshold_version}`，所有 11 项均为硬门。",
            f"- v2：`{threshold_version_v2}`，保留全部 11 项数值；接触高度 P95/最大值仅作描述，支撑高度和悬空门采用 v2 校准协议。",
            f"- 逐序列完整数值、左右脚跟/脚尖明细和失败原因见 `{per_sequence_name}`。",
            "",
            "## 物理失败原因（v1）",
            "",
        ]
    )
    for row in rows:
        reasons = row["physical"].get("failure_reason_counts", {})
        text = "；".join(f"{name}={count}" for name, count in reasons.items()) or "无"
        lines.append(f"- {row['method']}：{text}")
    lines.append("")
    return "\n".join(lines)


def _per_sequence_markdown(sequence_rows: list[Mapping[str, Any]]) -> str:
    lines = [
        "# S0 物理逐序列数值附表",
        "",
        "每行对应一条序列；`v1/v2` 同时保留状态、11 项原始物理值和左右脚跟/脚尖明细。",
        "",
        "| 方法 | seed | sample | dose (°) | v1 | v2 | 穿地 P95 | 穿地最大 | 穿地率 | 脚滑速度 P95 | 脚滑速度最大 | 总脚滑 | 最长脚滑 | 接触高度 P95 | 接触高度最大 | 悬空率 | 支撑误差 P95 | v1失败原因 | v2失败原因 |",
        "|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(
        sequence_rows,
        key=lambda value: (
            str(value.get("method")),
            int(value.get("seed", 0)),
            str(value.get("sample_id")),
            float(value.get("dose_deg", 0)),
        ),
    ):
        v1 = row.get("physical_v1") or {}
        v2 = row.get("physical_v2") or {}
        values = v2.get("metrics", {})
        lines.append(
            "| {method} | {seed} | {sample} | {dose:+g} | {v1s} | {v2s} | {metrics} | {r1} | {r2} |".format(
                method=row.get("method"),
                seed=row.get("seed"),
                sample=row.get("sample_id"),
                dose=float(row.get("dose_deg", 0.0)),
                v1s=v1.get("status"),
                v2s=v2.get("status"),
                metrics=" | ".join(str(values.get(name)) for name in PHYSICAL_METRICS),
                r1="；".join(v1.get("physical_fail_reasons", [])) or "无",
                r2="；".join(v2.get("physical_fail_reasons", [])) or "无",
            )
        )
    return "\n".join(lines) + "\n"


def _latex(rows: list[Mapping[str, Any]]) -> str:
    body = []
    for row in rows:
        method = row["method"] + (r"$^{\dagger}$" if row["method"] == "M7" else "")
        body.append(
            f"{method} & {row['angle_mae_deg']['display']} & {row['angle_p95_deg']['display']} & "
            f"{row['dose_response_slope']:.3f} & {row['sequence_angle_pass_count']}/12 & "
            f"{row['physical_v1']['pass_count']}/12 & {row['physical_v2']['pass_count']}/12 \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Pre-tuning S0 control and dual-track physical-closure results.}",
            r"\label{tab:s0_physical_closure}",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Method & MAE & P95 & Slope & Angle pass & Physical v1 & Physical v2 \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}\footnotesize $^{\dagger}$Post-generation geometric-edit reference. S1 tuning is pending.\end{flushleft}",
            r"\end{table}",
            "",
        ]
    )


def _render_png(rows: list[Mapping[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = ["Method", "Angle pass", "Physical v1", "Physical v2", "Slope", "MPJPE (mm)"]
    cells = [
        [
            row["method"] + ("†" if row["method"] == "M7" else ""),
            f"{row['sequence_angle_pass_count']}/12",
            f"{row['physical_v1']['pass_count']}/12",
            f"{row['physical_v2']['pass_count']}/12",
            f"{row['dose_response_slope']:.3f}",
            row["mpjpe_vs_m0_mm"]["display"],
        ]
        for row in rows
    ]
    figure, axis = plt.subplots(figsize=(12, 4.5), dpi=180)
    axis.axis("off")
    table = axis.table(cellText=cells, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    for column in range(len(columns)):
        table[(0, column)].set_facecolor("#24476b")
        table[(0, column)].set_text_props(color="white", weight="bold")
    for row_index, row in enumerate(rows, start=1):
        color = "#e8f3ea" if row["physical_v2"]["pass_count"] == row["sequence_count"] else "#f8dddd"
        if row["method"] == "M7":
            color = "#ececec"
        for column in range(len(columns)):
            table[(row_index, column)].set_facecolor(color)
    figure.suptitle("S0 control and dual-track physical closure (pre-tuning)", fontsize=13)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _render_heatmap(rows: list[Mapping[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix: list[list[float]] = []
    for row in rows:
        values: list[float] = []
        for name in PHYSICAL_METRICS:
            metric = row["physical_v2"]["metric_stats"].get(name, {})
            median = metric.get("median")
            threshold = metric.get("threshold")
            values.append(
                min(float(median) / float(threshold), 4.0)
                if median is not None and threshold not in (None, 0)
                else 0.0
            )
        matrix.append(values)
    figure, axis = plt.subplots(figsize=(15, 4.2), dpi=180)
    image = axis.imshow(np.asarray(matrix), aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=2.0)
    axis.set_yticks(range(len(rows)), [row["method"] for row in rows])
    axis.set_xticks(
        range(len(PHYSICAL_METRICS)),
        [METRIC_LABELS[name] for name in PHYSICAL_METRICS],
        rotation=55,
        ha="right",
    )
    axis.set_title("S0 physical v2 median / threshold (capped at 4)")
    figure.colorbar(image, ax=axis, label="median / threshold")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run(
    preliminary_path: Path,
    physical_path: Path,
    output: Path,
    *,
    render_png: bool = True,
    physical_source_label: str | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite S0 physical table: {output}")
    preliminary = _load(preliminary_path)
    physical = _load(physical_path)
    rows = build_rows(preliminary, physical)
    output.mkdir(parents=True)
    sequence_rows = _sequence_rows(physical.get("records", []))
    per_sequence_name = "TABLE1_S0_PHYSICAL_PER_SEQUENCE.md"
    summary = {
        "status": "S0_PHYSICAL_CLOSURE_COMPLETE_S1_REQUIRED",
        "sequence_count": 84,
        "sequence_count_per_method": 12,
        "threshold_version": physical.get("threshold_version"),
        "threshold_version_v2": physical.get("threshold_version_v2"),
        "decision_profile": physical.get("decision_profile"),
        "decision_profile_v2": physical.get("decision_profile_v2"),
        "physical_status_counts": physical.get("physical_status_counts"),
        "physical_v2_status_counts": physical.get("physical_v2_status_counts"),
        "s2_allowed": False,
        "s2_blockers": physical.get("s2_blockers", []),
        "sources": {
            "preliminary_summary": str(preliminary_path),
            "preliminary_summary_sha256": _sha256(preliminary_path),
            "physical_summary": physical_source_label or str(physical_path),
            "physical_summary_sha256": _sha256(physical_path),
        },
        "methods": rows,
        "per_sequence": sequence_rows,
        "m7_claims_generator_response": False,
    }
    (output / "table1_s0_physical_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "TABLE1_S0_PHYSICAL.md").write_text(
        _markdown(
            rows,
            str(summary["threshold_version"]),
            str(summary["threshold_version_v2"]),
            per_sequence_name,
        ),
        encoding="utf-8",
    )
    (output / per_sequence_name).write_text(
        _per_sequence_markdown(sequence_rows), encoding="utf-8"
    )
    (output / "table1_s0_physical.tex").write_text(_latex(rows), encoding="utf-8")
    if render_png:
        _render_png(rows, output / "table1_s0_physical.png")
        _render_heatmap(rows, output / "physical_v2_median_threshold_heatmap.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preliminary", type=Path, default=DEFAULT_PRELIMINARY)
    parser.add_argument("--physical", type=Path, default=DEFAULT_PHYSICAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--physical-source-label")
    args = parser.parse_args()
    summary = run(
        args.preliminary,
        args.physical,
        args.output,
        render_png=not args.no_png,
        physical_source_label=args.physical_source_label,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output": str(args.output),
                "physical_v1": summary["physical_status_counts"],
                "physical_v2": summary["physical_v2_status_counts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
