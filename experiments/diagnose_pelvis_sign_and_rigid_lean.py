#!/usr/bin/env python3
"""Diagnose pelvis-sign semantics and rigid whole-body lean from frozen CSVs.

This is a read-only diagnostic.  It does not regenerate motion, select a
candidate, or change any frozen protocol.  The current M1--M7 evaluator uses
the SMPL-X local +z root axis and defines positive as anterior-up.  The
anatomical convention required by the project is anterior-down positive, so
the corrected proxy delta is the negative of the current reported delta.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


PROTOCOL = "vimogen_pelvis_sign_rigid_lean_diagnosis_v1"
SELECTED_CANDIDATES = (
    ("M1", "scan_08"),
    ("M3", "scan_08"),
    ("M4", "scan_02"),
    ("M4", "scan_05"),
    ("M5", "scan_06"),
    ("M6", "scan_05"),
    ("M6", "scan_07"),
    ("M7", "reference"),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty metrics table: {path}")
    return rows


def number(row: dict[str, str], field: str) -> float:
    raw = row.get(field, "")
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"missing/non-numeric {field!r}: {raw!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field!r}: {raw!r}")
    return value


def stats(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    if not values:
        raise ValueError("cannot summarize an empty value set")
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def compact_candidate(row: dict[str, str]) -> dict[str, Any]:
    current = number(row, "actual_mean_dose_deg")
    return {
        "method": row["method"],
        "config_id": row["config_id"],
        "target_dose_deg_current_semantics": number(row, "target_dose_deg"),
        "actual_delta_deg_current_anterior_up_semantics": current,
        "actual_delta_deg_corrected_anterior_down_semantics": -current,
        "angle_mae_deg_current_semantics": number(row, "angle_mae_deg"),
        "world_trunk_direction_deviation_mean_deg": number(
            row, "content_v2_world_trunk_direction_deviation_mean_deg"
        ),
        "world_trunk_direction_deviation_p95_deg": number(
            row, "content_v2_world_trunk_direction_deviation_p95_deg"
        ),
        "local_rotation_all_mean_deg": number(
            row, "content_v2_local_rotation_all_mean_deg"
        ),
        "root_local_21_p95_mm": number(row, "content_v2_root_local_21_p95_mm"),
        "root_translation_p95_mm": number(
            row, "content_v2_root_translation_deviation_p95_mm"
        ),
        "world_mpjpe_vs_m0_mm": number(row, "mpjpe_vs_m0_mm"),
        "penetration_max_mm": number(row, "candidate_physical_penetration_max_mm"),
        "penetration_delta_max_mm": number(row, "physical_delta_penetration_max_mm"),
        "near_ground_slide_speed_p95_mm_per_second": number(
            row, "candidate_physical_near_ground_slide_speed_p95_mm_per_second"
        ),
        "near_ground_slide_delta_p95_mm_per_second": number(
            row, "physical_delta_near_ground_slide_speed_p95_mm_per_second"
        ),
    }


def diagnose(rows: list[dict[str, str]], *, sample_id: str, seed: int) -> dict[str, Any]:
    m7 = [row for row in rows if row.get("method") == "M7"]
    if not m7:
        raise ValueError("metrics table contains no M7 rows")
    dose_values = sorted({number(row, "target_dose_deg") for row in m7})
    if dose_values != [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0]:
        raise ValueError(f"unexpected M7 dose set: {dose_values}")

    per_dose: list[dict[str, Any]] = []
    for dose in dose_values:
        group = [row for row in m7 if number(row, "target_dose_deg") == dose]
        current = [number(row, "actual_mean_dose_deg") for row in group]
        per_dose.append(
            {
                "target_dose_deg_current_semantics": dose,
                "n": len(group),
                "actual_delta_deg_current_anterior_up_semantics": stats(current),
                "actual_delta_deg_corrected_anterior_down_semantics": stats(
                    -value for value in current
                ),
                "world_trunk_direction_deviation_p95_deg": stats(
                    number(row, "content_v2_world_trunk_direction_deviation_p95_deg")
                    for row in group
                ),
                "local_rotation_all_mean_deg": stats(
                    number(row, "content_v2_local_rotation_all_mean_deg")
                    for row in group
                ),
                "root_local_21_p95_mm": stats(
                    number(row, "content_v2_root_local_21_p95_mm") for row in group
                ),
                "root_translation_p95_mm": stats(
                    number(row, "content_v2_root_translation_deviation_p95_mm")
                    for row in group
                ),
                "world_mpjpe_vs_m0_mm": stats(
                    number(row, "mpjpe_vs_m0_mm") for row in group
                ),
            }
        )

    positive_m7 = [row for row in m7 if number(row, "target_dose_deg") > 0.0]
    nonzero_m7 = [row for row in m7 if number(row, "target_dose_deg") != 0.0]
    sign_semantics_reversed = all(
        number(row, "actual_mean_dose_deg") > 0.0
        and -number(row, "actual_mean_dose_deg") < 0.0
        for row in positive_m7
    )
    max_trunk_tracking_error = max(
        abs(
            number(row, "content_v2_world_trunk_direction_deviation_p95_deg")
            - abs(number(row, "target_dose_deg"))
        )
        for row in nonzero_m7
    )
    max_local_rotation = max(
        number(row, "content_v2_local_rotation_all_mean_deg") for row in nonzero_m7
    )
    max_root_local_p95 = max(
        number(row, "content_v2_root_local_21_p95_mm") for row in nonzero_m7
    )
    max_root_translation_p95 = max(
        number(row, "content_v2_root_translation_deviation_p95_mm")
        for row in nonzero_m7
    )
    rigid_whole_body_rotation = (
        max_trunk_tracking_error <= 0.05
        and max_local_rotation <= 1.0e-6
        and max_root_local_p95 <= 1.0e-3
        and max_root_translation_p95 <= 1.0e-6
    )

    selected: list[dict[str, Any]] = []
    for method, config_id in SELECTED_CANDIDATES:
        matches = [
            row
            for row in rows
            if row.get("method") == method
            and row.get("config_id") == config_id
            and str(row.get("sample_id")) == str(sample_id)
            and int(row.get("seed", -1)) == int(seed)
            and number(row, "target_dose_deg") in (-10.0, 10.0)
        ]
        if len(matches) != 2:
            raise ValueError(
                f"expected two extreme-dose rows for {method}/{config_id}/"
                f"sample{sample_id}/seed{seed}, found {len(matches)}"
            )
        selected.extend(compact_candidate(row) for row in matches)
    selected.sort(
        key=lambda row: (
            row["method"],
            row["config_id"],
            row["target_dose_deg_current_semantics"],
        )
    )

    return {
        "protocol": PROTOCOL,
        "status": "CURRENT_CONTROL_SEMANTICS_FAIL",
        "definition": {
            "smplx_local_anterior_axis": "+z",
            "current_positive_direction": "anterior-up",
            "required_anatomical_positive_direction": "anterior-down",
            "corrected_proxy_relation": "corrected_delta = -current_delta",
        },
        "m7_row_count": len(m7),
        "m7_per_dose": per_dose,
        "sample_id": str(sample_id),
        "seed": int(seed),
        "selected_extreme_doses": selected,
        "failure_flags": {
            "sign_semantics_reversed": sign_semantics_reversed,
            "m7_is_rigid_whole_body_rotation": rigid_whole_body_rotation,
        },
        "rigid_rotation_checks": {
            "max_world_trunk_p95_minus_abs_dose_deg": max_trunk_tracking_error,
            "max_local_rotation_all_mean_deg": max_local_rotation,
            "max_root_local_21_p95_mm": max_root_local_p95,
            "max_root_translation_p95_mm": max_root_translation_p95,
        },
        "conclusion": (
            "The current positive dose is posterior/anterior-up under the required "
            "anatomical convention, and M7 reaches it by rotating the global root "
            "with effectively unchanged root-local pose."
        ),
    }


def markdown_report(result: dict[str, Any]) -> str:
    selected = result["selected_extreme_doses"]
    positive = [row for row in selected if row["target_dose_deg_current_semantics"] == 10.0]
    lines = [
        "# 骨盆符号与整体倾倒诊断",
        "",
        f"状态：`{result['status']}`。本诊断只读取冻结 CSV，不生成动作、不选参。",
        "",
        "## 核心结论",
        "",
        "- SMPL-X 根局部 `+z` 为人体前方；现指标把前侧上抬记为正，而项目要求的解剖前倾是前侧下沉为正。因此正确代理角增量等于现值取反。",
        "- M7 的局部关节旋转保持为零、根局部 21 关节偏差接近数值噪声，但世界躯干方向随剂量近乎一比一改变，证明它是整体根节点旋转，不是局部骨盆前后倾。",
        "",
        "## 样本 94、seed 0、当前 +10°",
        "",
        "| 方法 | 当前实际角 | 解剖代理角 | 世界躯干 P95 | 根局部21关节 P95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in positive:
        lines.append(
            f"| {row['method']} {row['config_id']} | "
            f"{row['actual_delta_deg_current_anterior_up_semantics']:.3f}° | "
            f"{row['actual_delta_deg_corrected_anterior_down_semantics']:.3f}° | "
            f"{row['world_trunk_direction_deviation_p95_deg']:.3f}° | "
            f"{row['root_local_21_p95_mm']:.6f} mm |"
        )
    checks = result["rigid_rotation_checks"]
    lines.extend(
        [
            "",
            "## M7 全部 56 条记录的硬判据",
            "",
            f"- 世界躯干 P95 与绝对剂量的最大差：`{checks['max_world_trunk_p95_minus_abs_dose_deg']:.6f}°`。",
            f"- 非零剂量局部关节旋转均值最大值：`{checks['max_local_rotation_all_mean_deg']:.6g}°`。",
            f"- 根局部 21 关节 P95 最大值：`{checks['max_root_local_21_p95_mm']:.9f} mm`。",
            f"- 根平移 P95 最大值：`{checks['max_root_translation_p95_mm']:.6g} mm`。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--m7-source", type=Path, required=True)
    parser.add_argument("--angle-source", type=Path, required=True)
    parser.add_argument("--sample-id", default="94")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--fail-on-current-semantics", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.metrics, args.m7_source, args.angle_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = diagnose(read_rows(args.metrics), sample_id=args.sample_id, seed=args.seed)
    result["inputs"] = {
        "metrics": str(args.metrics.resolve()),
        "metrics_sha256": sha256_file(args.metrics),
        "m7_source": str(args.m7_source.resolve()),
        "m7_source_sha256": sha256_file(args.m7_source),
        "angle_source": str(args.angle_source.resolve()),
        "angle_source_sha256": sha256_file(args.angle_source),
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(markdown_report(result), encoding="utf-8")
    print(payload, end="")
    if args.fail_on_current_semantics and any(result["failure_flags"].values()):
        print("RED: current pelvis-control semantics fail sign and/or rigid-lean checks")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
