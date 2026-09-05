#!/usr/bin/env python3
"""Create compact per-step diagnostics for a v0.3 projection run.

The evaluator intentionally keeps the frozen v3 diagnostics separate from the
v0.3 primary contact gates.  This report makes a failed primary run easy to
inspect without loading large tensors: it exports one CSV, a strict JSON
summary, and a six-panel PNG from the projection trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


COMPONENTS = (
    "pelvis",
    "heel_position",
    "toe_position",
    "heel_velocity",
    "toe_velocity",
    "penetration",
)


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return value is None or isinstance(value, str)


def _num(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _trace_rows(projection: dict[str, Any]) -> list[dict[str, Any]]:
    records = projection.get("records", [])
    # The sampler stores one case record whose ``step_records`` contains the
    # per-diffusion-step trace.  Accept a flat list as well for older logs.
    if len(records) == 1 and isinstance(records[0], dict) and records[0].get("step_records"):
        records = records[0]["step_records"]
    rows: list[dict[str, Any]] = []
    for step, raw_record in enumerate(records):
        # Active diffusion steps contain a compact summary plus a nested
        # relinearization trace.  Use the final nested iteration for the
        # before/after violation values while retaining the outer step fields.
        detail = raw_record
        nested = raw_record.get("records")
        if isinstance(nested, list) and nested and isinstance(nested[-1], dict):
            detail = nested[-1]
        before = detail.get("normalized_violation_before", {})
        after = detail.get("normalized_violation_after", {})
        per_iteration = raw_record.get(
            "per_iteration_root_translation", detail.get("per_iteration_root_translation")
        )
        if isinstance(per_iteration, list):
            per_iteration = max((_num(item) for item in per_iteration), default=float("nan"))
        row: dict[str, Any] = {
            "step": step,
            "sigma": _num(raw_record.get("sigma")),
            "projection_enabled": bool(raw_record.get("projection_enabled", False)),
            "accepted": bool(raw_record.get("accepted", False)),
            "backtracking_alpha": _num(detail.get("backtracking_alpha")),
            "pelvis_residual_deg": _num(raw_record.get("pelvis_residual", detail.get("pelvis_residual_after_deg"))),
            "heel_position_residual_m": _num(raw_record.get("heel_residual", detail.get("heel_residual_after_m"))),
            "toe_position_residual_m": _num(raw_record.get("toe_residual", detail.get("toe_residual_after_m"))),
            "heel_velocity_residual_m_per_frame": _num(
                raw_record.get("heel_velocity_residual", detail.get("heel_velocity_after_m_per_frame"))
            ),
            "toe_velocity_residual_m_per_frame": _num(
                raw_record.get("toe_velocity_residual", detail.get("toe_velocity_after_m_per_frame"))
            ),
            "penetration_residual_m": _num(raw_record.get("penetration_residual", detail.get("penetration_after_m"))),
            "per_iteration_root_translation_m": _num(
                per_iteration if per_iteration is not None else raw_record.get("delta_root_translation_norm")
            ),
            "cumulative_root_translation_m": _num(
                raw_record.get("cumulative_root_translation", raw_record.get("delta_root_translation_norm"))
            ),
            "max_joint_increment_deg": _num(detail.get("max_joint_increment_deg")),
        }
        for component in COMPONENTS:
            row[f"violation_before_{component}"] = _num(before.get(component))
            row[f"violation_after_{component}"] = _num(after.get(component))
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["step"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, rows: list[dict[str, Any]], final: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["step"] for row in rows]
    enabled = [row["projection_enabled"] for row in rows]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(x, [row["pelvis_residual_deg"] for row in rows], label="pelvis residual (deg)")
    ax.axhline(0.25, color="k", linestyle="--", linewidth=0.8, label="threshold")
    ax.set_title("骨盆实际剂量残差")
    ax.set_xlabel("采样步")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, [1000.0 * row["heel_position_residual_m"] for row in rows], label="heel position")
    ax.plot(x, [1000.0 * row["toe_position_residual_m"] for row in rows], label="toe position")
    ax.plot(x, [1000.0 * row["heel_velocity_residual_m_per_frame"] for row in rows], "--", label="heel velocity")
    ax.plot(x, [1000.0 * row["toe_velocity_residual_m_per_frame"] for row in rows], "--", label="toe velocity")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, label="1 mm gate")
    ax.set_title("脚跟/脚尖位置与速度残差")
    ax.set_ylabel("mm 或 mm/帧")
    ax.set_xlabel("采样步")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(x, [1000.0 * row["per_iteration_root_translation_m"] for row in rows], label="per-step")
    ax.plot(x, [1000.0 * row["cumulative_root_translation_m"] for row in rows], label="cumulative")
    ax.axhline(10.0, color="k", linestyle="--", linewidth=0.8, label="10 mm single-step gate")
    ax.set_title("根平移单步与累计值")
    ax.set_ylabel("mm")
    ax.set_xlabel("采样步")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.step(x, [1.0 if value else 0.0 for value in enabled], where="mid", label="projection enabled")
    ax.plot(x, [row["backtracking_alpha"] for row in rows], marker=".", label="backtracking alpha")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("投影启用步与接受步长")
    ax.set_xlabel("采样步")
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    for component in COMPONENTS:
        ax.plot(x, [row[f"violation_after_{component}"] for row in rows], label=component)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, label="gate")
    ax.set_title("六类归一化违反量（回溯后）")
    ax.set_xlabel("采样步")
    ax.set_ylabel("归一化违反量")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[2, 1]
    before = final.get("pre_residuals", {})
    after = final.get("post_residuals", {})
    labels = ["pelvis", "heel", "toe", "heel vel", "toe vel"]
    before_values = [
        _num(before.get("pelvis_geodesic_rms_deg")),
        1000.0 * _num(before.get("heel_rms_m")),
        1000.0 * _num(before.get("toe_rms_m")),
        1000.0 * _num(before.get("heel_velocity_rms_m_per_frame")),
        1000.0 * _num(before.get("toe_velocity_rms_m_per_frame")),
    ]
    after_values = [
        _num(after.get("pelvis_geodesic_rms_deg")),
        1000.0 * _num(after.get("heel_rms_m")),
        1000.0 * _num(after.get("toe_rms_m")),
        1000.0 * _num(after.get("heel_velocity_rms_m_per_frame")),
        1000.0 * _num(after.get("toe_velocity_rms_m_per_frame")),
    ]
    positions = list(range(len(labels)))
    width = 0.38
    ax.bar([p - width / 2 for p in positions], before_values, width, label="before")
    ax.bar([p + width / 2 for p in positions], after_values, width, label="after")
    ax.set_xticks(positions, labels, rotation=20)
    ax.set_title("投影启用步与最终端点误差")
    ax.legend(fontsize=8)

    fig.suptitle("ViMoGen v0.3 投影失败诊断")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_root = args.run_root
    output_dir = args.output_dir or (run_root / "diagnostics_v0_3")
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation = json.loads((run_root / "evaluation.json").read_text(encoding="utf-8"))
    projection = json.loads(
        (run_root / "projection_artifacts" / "batch_000" / "sampling_projection_log.json").read_text(
            encoding="utf-8"
        )
    )
    rows = _trace_rows(projection)
    _write_csv(output_dir / "per_step_projection_diagnostics.csv", rows)
    trace_records = projection.get("records", [])
    if len(trace_records) == 1 and isinstance(trace_records[0], dict) and trace_records[0].get("step_records"):
        trace_records = trace_records[0]["step_records"]
    final_records = [row for row in trace_records if row.get("projection_enabled")]
    final = final_records[-1] if final_records else {}
    summary = {
        "protocol": evaluation.get("protocol"),
        "sample_id": evaluation.get("sample_id"),
        "side": evaluation.get("side"),
        "target_delta_deg": evaluation.get("target_delta_deg"),
        "interpretation": evaluation.get("interpretation"),
        "m0_pairing": evaluation.get("m0_pairing"),
        "primary_control": evaluation.get("primary_control"),
        "contact": evaluation.get("contact"),
        "projection_trace_steps": len(rows),
        "projection_enabled_steps": sum(1 for row in rows if row["projection_enabled"]),
        # Inactive diffusion steps legitimately omit residual fields; their
        # CSV cells are blank/NaN for plotting.  The strict JSON artifacts
        # themselves are checked against the original finite input trees.
        "finite": _finite(projection) and _finite(evaluation),
        "inactive_steps_without_residuals": sum(
            1 for row in rows if not row["projection_enabled"]
        ),
        "final_projection_record": final,
        "artifacts": {
            "csv": "per_step_projection_diagnostics.csv",
            "plot": "per_step_projection_diagnostics.png",
        },
    }
    (output_dir / "diagnostics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_plot(output_dir / "per_step_projection_diagnostics.png", rows, final)
    print(json.dumps({"output_dir": str(output_dir), "steps": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
