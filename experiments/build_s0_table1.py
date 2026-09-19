"""Build the preliminary S0 control-accuracy Table 1 from audited records.

The table is intentionally limited to S0.  It does not convert missing
physical marker evidence into a pass, and it labels M7 as a post-generation
geometric-edit reference rather than a generator-response method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLING_ROOT = ROOT / "results/phase9/pelvis_m1_m7/s0_sampling"
DEFAULT_M7_ROOT = ROOT / "results/phase9/pelvis_m1_m7/s0_m7/attempt_01"
DEFAULT_OUTPUT = ROOT / "results/phase9/pelvis_m1_m7/table1_s0_preliminary"

METHOD_LABELS = {
    "M1": "M1 能量引导",
    "M2": "M2 源噪声优化",
    "M3": "M3 局部投影",
    "M4": "M4 前向射击",
    "M5": "M5 原始—对偶流",
    "M6": "M6 李雅普诺夫引导",
    "M7": "M7 生成后几何编辑†",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_record(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _sampling_records(sampling_root: Path) -> list[dict[str, Any]]:
    progress = _load_json(sampling_root / "s0_matrix_progress.json")
    latest: dict[tuple[str, int, float], dict[str, Any]] = {}
    for item in progress.get("jobs", []):
        key = (str(item["method"]), int(item["seed"]), float(item["dose"]))
        latest[key] = item

    records: list[dict[str, Any]] = []
    for key in sorted(latest):
        evaluation = latest[key].get("evaluation", {})
        if evaluation.get("status") in {"EVALUATION_FAILED", "EVALUATION_OUTPUT_INVALID"}:
            raise RuntimeError(f"unrepaired evaluation remains for {key}")
        record_paths = evaluation.get("records", [])
        if len(record_paths) != 2:
            raise RuntimeError(f"expected two records for {key}, found {len(record_paths)}")
        records.extend(_load_json(_resolve_record(path)) for path in record_paths)
    return records


def _m7_records(m7_root: Path) -> list[dict[str, Any]]:
    summary = _load_json(m7_root / "summary.json")
    if summary.get("completed_runs") != summary.get("expected_runs"):
        raise RuntimeError("M7 S0 summary is incomplete")
    return [_load_json(_resolve_record(path)) for path in summary.get("records", [])]


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return tuple(float(value) for value in np.percentile(array, [50, 25, 75]))


def _format_iqr(values: list[float], digits: int = 3) -> str:
    median, q1, q3 = _quartiles(values)
    return f"{median:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def _summarize_method(method: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 12:
        raise RuntimeError(f"{method}: expected 12 S0 records, found {len(records)}")

    targets: list[float] = []
    actuals: list[float] = []
    maes: list[float] = []
    p95s: list[float] = []
    zero_drifts: list[float] = []
    sign_values: list[bool] = []
    passes: list[bool] = []
    mpjpes: list[float] = []
    root_deviations: list[float] = []
    physical_statuses: set[str] = set()

    for record in records:
        metrics = record["all_metrics"]
        control = metrics["control"]["per_sequence"][0]
        content = metrics["content"]["per_sequence"][0]
        target = float(record["target_dose_deg"])
        targets.append(target)
        actuals.append(float(control["actual_mean_dose_deg"]))
        maes.append(float(control["angle_mae_deg"]))
        p95s.append(float(control["angle_p95_deg"]))
        passes.append(bool(control["sequence_angle_pass"]))
        mpjpes.append(float(content["mpjpe_vs_m0_mm"]))
        root_deviations.append(float(content["root_translation_deviation_p95_mm"]))
        physical_statuses.add(str(metrics["physical"]["status"]))
        if target == 0.0:
            zero_drifts.append(float(control["zero_dose_drift_mae_deg"]))
        else:
            sign_values.append(bool(control["sign_correct"]))

    slope, intercept = np.polyfit(np.asarray(targets), np.asarray(actuals), 1)
    return {
        "method": method,
        "label": METHOD_LABELS[method],
        "sequence_count": len(records),
        "angle_mae_deg": {
            "median_q1_q3": _quartiles(maes),
            "display": _format_iqr(maes),
        },
        "angle_p95_deg": {
            "median_q1_q3": _quartiles(p95s),
            "display": _format_iqr(p95s),
        },
        "sign_correct_rate": float(np.mean(sign_values)),
        "dose_response_slope": float(slope),
        "dose_response_intercept_deg": float(intercept),
        "zero_dose_drift_deg": {
            "median_q1_q3": _quartiles(zero_drifts),
            "display": _format_iqr(zero_drifts),
        },
        "sequence_angle_pass_count": int(sum(passes)),
        "sequence_angle_pass_rate": float(np.mean(passes)),
        "mpjpe_vs_m0_mm": {
            "median_q1_q3": _quartiles(mpjpes),
            "display": _format_iqr(mpjpes, digits=1),
            "maximum": float(max(mpjpes)),
        },
        "root_translation_deviation_p95_mm": {
            "median_q1_q3": _quartiles(root_deviations),
            "display": _format_iqr(root_deviations, digits=1),
            "maximum": float(max(root_deviations)),
        },
        "physical_statuses": sorted(physical_statuses),
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Table 1. S0 骨盆角控制精度（预备结果）",
        "",
        "| 方法 | 约束包 | 剂量（°） | 角度 MAE（°）↓ | 角度 P95（°）↓ | 非零剂量符号正确率↑ | 剂量响应斜率→1 | 零剂量漂移（°）↓ | 序列角度通过率↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | C0 | −2/0/+2 | {mae} | {p95} | {sign:.1%} | {slope:.3f} | {zero} | {passes}/{n} ({rate:.1%}) |".format(
                label=row["label"],
                mae=row["angle_mae_deg"]["display"],
                p95=row["angle_p95_deg"]["display"],
                sign=row["sign_correct_rate"],
                slope=row["dose_response_slope"],
                zero=row["zero_dose_drift_deg"]["display"],
                passes=row["sequence_angle_pass_count"],
                n=row["sequence_count"],
                rate=row["sequence_angle_pass_rate"],
            )
        )
    lines.extend(
        [
            "",
            "数值为 12 条 S0 序列（2 个动作 × 2 个 seed × 3 个剂量）的中位数 [Q1, Q3]。角度硬门为 MAE≤1°、P95≤2°，且非零剂量符号正确。剂量响应斜率由实际平均剂量对目标剂量的带截距线性拟合得到。",
            "",
            "† M7 是生成后几何编辑参考，不代表 ViMoGen 生成器对控制信号的自然响应。全部方法的冻结脚部 marker 尚未物化，地面与接触门均为待评估，因此本表不能作为最终论文主表。",
            "",
            "## 内容保持诊断（不参与角度硬门）",
            "",
            "| 方法 | 22-joint MPJPE vs M0（mm）↓ | 根平移 P95 偏差（mm）↓ | 最大 MPJPE（mm） | 物理门 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        physical = "待评估" if any("PENDING" in value for value in row["physical_statuses"]) else "已评估"
        lines.append(
            "| {label} | {mpjpe} | {root} | {maximum:.1f} | {physical} |".format(
                label=row["label"],
                mpjpe=row["mpjpe_vs_m0_mm"]["display"],
                root=row["root_translation_deviation_p95_mm"]["display"],
                maximum=row["mpjpe_vs_m0_mm"]["maximum"],
                physical=physical,
            )
        )
    lines.extend(
        [
            "",
            "当前最重要的诊断：M2 与 M5 的非零剂量未稳定通过角度门；M3、M4 虽通过角度门，但部分序列出现明显内容偏差。",
            "",
        ]
    )
    return "\n".join(lines)


def _latex(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        label = row["method"] + (r"$^{\dagger}$" if row["method"] == "M7" else "")
        body.append(
            " & ".join(
                [
                    label,
                    "C0",
                    r"$-2/0/+2$",
                    row["angle_mae_deg"]["display"],
                    row["angle_p95_deg"]["display"],
                    f"{100 * row['sign_correct_rate']:.1f}\\%",
                    f"{row['dose_response_slope']:.3f}",
                    row["zero_dose_drift_deg"]["display"],
                    f"{row['sequence_angle_pass_count']}/{row['sequence_count']}",
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Preliminary S0 pelvis-angle control accuracy. Values are median [Q1, Q3] over 12 sequences.}",
            r"\label{tab:s0_control_preliminary}",
            r"\begin{tabular}{lllrrrrrr}",
            r"\toprule",
            r"Method & Pack & Dose ($^\circ$) & MAE $\downarrow$ & P95 $\downarrow$ & Sign $\uparrow$ & Slope $\to1$ & Zero drift $\downarrow$ & Pass $\uparrow$ \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}\footnotesize $^{\dagger}$Post-generation geometric-edit reference; not a generator-response method. Physical gates are pending marker materialization.\end{flushleft}",
            r"\end{table}",
            "",
        ]
    )


def _render_png(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = ["Method", "Pack", "Dose", "MAE (deg)", "P95 (deg)", "Sign", "Slope", "Zero drift", "Pass"]
    cells = []
    for row in rows:
        cells.append(
            [
                row["method"] + ("†" if row["method"] == "M7" else ""),
                "C0",
                "-2/0/+2",
                row["angle_mae_deg"]["display"],
                row["angle_p95_deg"]["display"],
                f"{100 * row['sign_correct_rate']:.0f}%",
                f"{row['dose_response_slope']:.3f}",
                row["zero_dose_drift_deg"]["display"],
                f"{row['sequence_angle_pass_count']}/{row['sequence_count']}",
            ]
        )

    fig, axis = plt.subplots(figsize=(17.5, 4.2), dpi=180)
    axis.axis("off")
    table = axis.table(cellText=cells, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    widths = [0.07, 0.055, 0.075, 0.17, 0.17, 0.07, 0.07, 0.17, 0.07]
    for column, width in enumerate(widths):
        for row in range(len(cells) + 1):
            table[(row, column)].set_width(width)
    for column in range(len(columns)):
        table[(0, column)].set_facecolor("#24476b")
        table[(0, column)].set_text_props(color="white", weight="bold")
    for row_index, row in enumerate(rows, start=1):
        color = "#e8f3ea" if row["sequence_angle_pass_count"] == row["sequence_count"] else "#fff1dc"
        if row["method"] == "M7":
            color = "#ececec"
        for column in range(len(columns)):
            table[(row_index, column)].set_facecolor(color)
    fig.suptitle("Table 1. S0 pelvis-angle control accuracy (preliminary)", fontsize=14, y=0.97)
    fig.text(
        0.5,
        0.035,
        "Median [Q1, Q3], n=12 per method. Green: 12/12 angle-gate pass; orange: partial pass. "
        "†M7 is a post-generation edit reference. Physical gates are pending.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(sampling_root: Path, m7_root: Path, output: Path, render_png: bool = True) -> dict[str, Any]:
    records = _sampling_records(sampling_root) + _m7_records(m7_root)
    grouped: dict[str, list[dict[str, Any]]] = {method: [] for method in METHOD_LABELS}
    for record in records:
        grouped[str(record["method_name"])].append(record)
    rows = [_summarize_method(method, grouped[method]) for method in METHOD_LABELS]

    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "PRELIMINARY_S0_ONLY",
        "sequence_count_per_method": 12,
        "methods": rows,
        "physical_gate": "PENDING_SHARED_FK_MARKER_MATERIALIZATION",
        "m7_claims_generator_response": False,
    }
    (output / "table1_s0_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "TABLE1_S0_PRELIMINARY.md").write_text(_markdown(rows), encoding="utf-8")
    (output / "table1_s0_preliminary.tex").write_text(_latex(rows), encoding="utf-8")
    if render_png:
        _render_png(rows, output / "table1_s0_preliminary.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-root", type=Path, default=DEFAULT_SAMPLING_ROOT)
    parser.add_argument("--m7-root", type=Path, default=DEFAULT_M7_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()
    summary = run(args.sampling_root, args.m7_root, args.output, render_png=not args.no_png)
    print(json.dumps({"status": summary["status"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
