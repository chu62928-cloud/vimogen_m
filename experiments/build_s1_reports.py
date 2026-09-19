#!/usr/bin/env python3
"""Build immutable human-readable S1 reports from bounded-run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


METHODS = tuple(f"M{index}" for index in range(1, 7))


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(root: Path, method: str, index: int, stage: str) -> dict[str, Any]:
    path = root / "runs" / method / f"config_{index:02d}" / f"stage_{stage}_summary.json"
    return _json(path) if path.is_file() else {}


def _table_row(method: str, frozen: dict[str, Any], progress: dict[str, Any], root: Path) -> str:
    selection = progress.get("selections", {}).get(method, {})
    winner = selection.get("confirm_winner")
    if winner is None:
        return f"| {method} | {frozen.get('status', selection.get('status', 'NOT_RUN'))} | — | — | — | — |"
    confirm = _summary(root, method, int(winner), "confirm")
    stress = _summary(root, method, int(winner), "stress")
    return (
        f"| {method} | {frozen.get('status', 'NOT_FROZEN')} | {int(winner):02d} | "
        f"{confirm.get('angle_pass_count', 0)}/{confirm.get('sequence_count', 0)} | "
        f"{confirm.get('physical_v2_pass_count', 0)}/{confirm.get('sequence_count', 0)} | "
        f"{stress.get('angle_pass_count', 0)}/{stress.get('sequence_count', 0)} |"
    )


def _failure_lines(root: Path, frozen: dict[str, Any]) -> list[str]:
    lines = [
        "# S1 失败案例",
        "",
        "该表只记录 confirm 阶段冻结配置的失败序列；压力测试不用于再次调参。",
        "",
        "| 方法 | 配置 | seed | 剂量 (°) | 样本 | 角度 | 物理 v2 | MPJPE (mm) | 根平移 P95 (mm) |",
        "|---|---:|---:|---:|---|---|---|---:|---:|",
    ]
    for method in METHODS:
        status = next((item for item in frozen.get("methods", []) if item.get("method") == method), {})
        winner = status.get("config_index")
        if winner is None:
            continue
        summary = _summary(root, method, int(winner), "confirm")
        for row in summary.get("rows", []):
            if row.get("angle_pass") and row.get("physical_v2_pass") and row.get("numerical_ok"):
                continue
            lines.append(
                f"| {method} | {int(winner):02d} | {row.get('seed')} | {row.get('dose_deg')} | "
                f"{row.get('sample_id')} | {'通过' if row.get('angle_pass') else '失败'} | "
                f"{row.get('physical_v2_status', '缺失')} | {row.get('mpjpe_mm')} | "
                f"{row.get('root_translation_p95_mm')} |"
            )
    if len(lines) == 6:
        lines.append("| — | — | — | — | — | 无 | 无 | — | — |")
    return lines


def _stress_lines(root: Path, frozen: dict[str, Any]) -> list[str]:
    lines = [
        "# S1 压力测试",
        "",
        "±10° 只用于记录饱和、反号、内容崩塌和物理崩塌，不回写配置选择。",
        "",
        "| 方法 | 配置 | 剂量 (°) | 样本 | 角度 | 物理 v2 | MPJPE (mm) | 根平移 P95 (mm) |",
        "|---|---:|---:|---|---|---|---:|---:|",
    ]
    for method in METHODS:
        status = next((item for item in frozen.get("methods", []) if item.get("method") == method), {})
        winner = status.get("config_index")
        if winner is None:
            continue
        summary = _summary(root, method, int(winner), "stress")
        for row in summary.get("rows", []):
            lines.append(
                f"| {method} | {int(winner):02d} | {row.get('dose_deg')} | {row.get('sample_id')} | "
                f"{'通过' if row.get('angle_pass') else '失败'} | {row.get('physical_v2_status', '缺失')} | "
                f"{row.get('mpjpe_mm')} | {row.get('root_translation_p95_mm')} |"
            )
    if len(lines) == 6:
        lines.append("| — | — | — | — | — | 未生成 | — | — |")
    return lines


def build(root: Path, output: Path, *, invariant_root: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite S1 reports: {output}")
    progress = _json(root / "s1_progress.json")
    frozen = _json(root / "S1_FROZEN.json")
    output.mkdir(parents=True)
    frozen_by_method = {str(item["method"]): item for item in frozen.get("methods", [])}
    main = [
        "# S1 主表",
        "",
        "确认阶段按角度通过数→数值稳定→物理 v2→物理严重度→内容→成本冻结唯一配置；M7 为生成后编辑参考。",
        "",
        "| 方法 | 冻结状态 | 配置 | confirm 角度 | confirm 物理 v2 | stress 角度 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        main.append(_table_row(method, frozen_by_method.get(method, {}), progress, root))
    (output / "S1_MAIN_TABLE.md").write_text("\n".join(main) + "\n", encoding="utf-8")
    (output / "S1_FAILURE_CASES.md").write_text("\n".join(_failure_lines(root, frozen)) + "\n", encoding="utf-8")
    (output / "S1_STRESS_TABLE.md").write_text("\n".join(_stress_lines(root, frozen)) + "\n", encoding="utf-8")

    invariant_files = sorted(invariant_root.rglob("summary.json"))
    checklist = [
        "# S1 可复现性清单",
        "",
        f"代码提交：`{progress.get('code_commit')}`",
        "",
        "| 检查 | 结果 | 报告 |",
        "|---|---|---|",
    ]
    for path in invariant_files:
        try:
            value = _json(path)
        except json.JSONDecodeError:
            continue
        checklist.append(f"| {value.get('protocol', path.parent.name)} | {value.get('status')} | `{path}` |")
    checklist.extend(
        [
            "| 运行目录覆盖保护 | 已由运行器强制 | `run_sampling_guidance_smoke.py` |",
            "| S2 | 未启动 | S1 冻结文件明确 `s2_allowed=false` |",
        ]
    )
    (output / "S1_REPRODUCIBILITY_CHECKLIST.md").write_text("\n".join(checklist) + "\n", encoding="utf-8")

    summary = {
        "protocol": "vimogen_m1_m7_s1_bounded_tuning_v1",
        "status": "S1_REPORTS_COMPLETE",
        "s2_allowed": False,
        "code_commit": progress.get("code_commit"),
        "frozen": frozen,
        "sources": {
            "progress": str(root / "s1_progress.json"),
            "frozen": str(root / "S1_FROZEN.json"),
            "invariant_reports": [
                {"path": str(path), "sha256": _sha256(path)} for path in invariant_files
            ],
        },
        "artifacts": [str(path) for path in sorted(output.iterdir())],
    }
    (output / "s1_reports_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--invariant-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output, invariant_root=args.invariant_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
