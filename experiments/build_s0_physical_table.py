#!/usr/bin/env python3
"""Merge frozen S0 control/content results with supplemental physical gates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRELIMINARY = ROOT / "artifacts/table1_s0_preliminary/table1_s0_summary.json"
DEFAULT_PHYSICAL = ROOT / "results/phase9/pelvis_m1_m7/s0_physical_evaluation_v3/summary.json"
DEFAULT_OUTPUT = ROOT / "results/phase9/pelvis_m1_m7/table1_s0_physical_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_display(rows: list[Mapping[str, Any]], name: str) -> str:
    values = [
        float(row["physical"]["per_sequence"][0][name])
        for row in rows
        if row["physical"].get("per_sequence")
        and row["physical"]["per_sequence"][0].get(name) is not None
    ]
    if not values:
        return "NA"
    median, q1, q3 = np.percentile(np.asarray(values), [50, 25, 75])
    return f"{median:.1f} [{q1:.1f}, {q3:.1f}]"


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
        statuses = Counter(row["physical"]["status"] for row in source)
        failures = Counter(
            reason
            for row in source
            for reason in row["physical"]["per_sequence"][0].get(
                "physical_fail_reasons", []
            )
        )
        row = dict(methods[method])
        row["physical"] = {
            "pass_count": statuses.get("EVALUATED_PASS", 0),
            "fail_count": statuses.get("EVALUATED_FAIL", 0),
            "not_evaluated_count": statuses.get("NOT_EVALUATED", 0),
            "status_counts": dict(sorted(statuses.items())),
            "failure_reason_counts": dict(sorted(failures.items())),
            "penetration_p95_mm": _metric_display(source, "penetration_p95_mm"),
            "contact_tangent_speed_p95_mm_per_frame": _metric_display(
                source, "contact_tangent_speed_p95_mm_per_frame"
            ),
            "support_height_error_p95_mm": _metric_display(
                source, "support_height_error_p95_mm"
            ),
        }
        result.append(row)
    return result


def _markdown(rows: list[Mapping[str, Any]], threshold_version: str) -> str:
    lines = [
        "# Table 1. S0 骨盆控制与物理闭环（调参前）",
        "",
        "| 方法 | 角度 MAE（°）↓ | 角度 P95（°）↓ | 斜率→1 | 角度通过 | MPJPE（mm）↓ | 物理通过 | 穿透 P95（mm）↓ | 接触切速 P95（mm/帧）↓ | 支撑高度误差 P95（mm）↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        physical = row["physical"]
        lines.append(
            "| {label} | {mae} | {p95} | {slope:.3f} | {angle}/{n} | {mpjpe} | {physical_pass}/{n} | {penetration} | {speed} | {support} |".format(
                label=row["label"],
                mae=row["angle_mae_deg"]["display"],
                p95=row["angle_p95_deg"]["display"],
                slope=row["dose_response_slope"],
                angle=row["sequence_angle_pass_count"],
                physical_pass=physical["pass_count"],
                n=row["sequence_count"],
                mpjpe=row["mpjpe_vs_m0_mm"]["display"],
                penetration=physical["penetration_p95_mm"],
                speed=physical["contact_tangent_speed_p95_mm_per_frame"],
                support=physical["support_height_error_p95_mm"],
            )
        )
    lines.extend(
        [
            "",
            f"物理门使用冻结协议 `{threshold_version}`。数值为每方法 12 条序列的中位数 [Q1, Q3]；通过数按逐序列全部物理阈值联合判定。",
            "",
            "本表已完成 S0 物理闭环，但仍是调参前诊断表。M2/M3/M4 的结构修正、M5 有限预算调参及 S1 唯一配置冻结完成前，不得作为最终论文主表，也不得启动 S2。M7 始终仅为生成后几何编辑参考。",
            "",
            "## 物理失败原因计数",
            "",
        ]
    )
    for row in rows:
        reasons = row["physical"]["failure_reason_counts"]
        text = "；".join(f"{name}={count}" for name, count in reasons.items()) or "无"
        lines.append(f"- {row['method']}：{text}")
    lines.append("")
    return "\n".join(lines)


def _latex(rows: list[Mapping[str, Any]]) -> str:
    body = []
    for row in rows:
        method = row["method"] + (r"$^{\dagger}$" if row["method"] == "M7" else "")
        body.append(
            f"{method} & {row['angle_mae_deg']['display']} & {row['angle_p95_deg']['display']} & "
            f"{row['dose_response_slope']:.3f} & {row['sequence_angle_pass_count']}/12 & "
            f"{row['physical']['pass_count']}/12 \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Pre-tuning S0 control and physical-closure results.}",
            r"\label{tab:s0_physical_closure}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Method & MAE & P95 & Slope & Angle pass & Physical pass \\",
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

    columns = ["Method", "Angle pass", "Physical pass", "Slope", "MPJPE (mm)"]
    cells = [
        [
            row["method"] + ("†" if row["method"] == "M7" else ""),
            f"{row['sequence_angle_pass_count']}/12",
            f"{row['physical']['pass_count']}/12",
            f"{row['dose_response_slope']:.3f}",
            row["mpjpe_vs_m0_mm"]["display"],
        ]
        for row in rows
    ]
    figure, axis = plt.subplots(figsize=(10.5, 4.2), dpi=180)
    axis.axis("off")
    table = axis.table(cellText=cells, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    for column in range(len(columns)):
        table[(0, column)].set_facecolor("#24476b")
        table[(0, column)].set_text_props(color="white", weight="bold")
    for row_index, row in enumerate(rows, start=1):
        color = "#e8f3ea" if row["physical"]["pass_count"] == 12 else "#f8dddd"
        if row["method"] == "M7":
            color = "#ececec"
        for column in range(len(columns)):
            table[(row_index, column)].set_facecolor(color)
    figure.suptitle("S0 control and physical closure (pre-tuning)", fontsize=13)
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
    summary = {
        "status": "S0_PHYSICAL_CLOSURE_COMPLETE_S1_REQUIRED",
        "sequence_count": 84,
        "sequence_count_per_method": 12,
        "threshold_version": physical.get("threshold_version"),
        "physical_status_counts": physical.get("physical_status_counts"),
        "s2_allowed": False,
        "s2_blockers": physical.get("s2_blockers", []),
        "sources": {
            "preliminary_summary": str(preliminary_path),
            "preliminary_summary_sha256": _sha256(preliminary_path),
            "physical_summary": physical_source_label or str(physical_path),
            "physical_summary_sha256": _sha256(physical_path),
        },
        "methods": rows,
        "m7_claims_generator_response": False,
    }
    (output / "table1_s0_physical_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "TABLE1_S0_PHYSICAL.md").write_text(
        _markdown(rows, str(summary["threshold_version"])), encoding="utf-8"
    )
    (output / "table1_s0_physical.tex").write_text(_latex(rows), encoding="utf-8")
    if render_png:
        _render_png(rows, output / "table1_s0_physical.png")
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
    print(json.dumps({"status": summary["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
