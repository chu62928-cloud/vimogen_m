#!/usr/bin/env python3
"""Render a human-review form from a phase-3 candidate manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(manifest_path: Path, output_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "CANDIDATE_NOT_FROZEN":
        raise ValueError("review form expects CANDIDATE_NOT_FROZEN manifest")
    lines = [
        "# Phase 3 development-set review form",
        "",
        "Status: `CANDIDATE_NOT_FROZEN`",
        "",
        "请对每条文本填写 reviewer A、reviewer B 和 adjudication。可填写 `KEEP` 或 `DROP: 原因`；近义重复也必须注明。",
        "",
        "| 序号 | ID | 类别 | 文本 | Reviewer A | Reviewer B | 裁决 | 备注 |",
        "|---:|---:|---|---|---|---|---|---|",
    ]
    for item in manifest["items"]:
        text = str(item["motion_text_annot"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {item['dev_index']} | {item['id']} | {item['category']} | {text} | PENDING | PENDING | PENDING | |")
    lines.extend(
        [
            "",
            "## Freeze conditions",
            "",
            "- 两名审核人均完成20条记录；",
            "- 所有 DROP 项有明确原因并替换为同类别候选；",
            "- 近义重复已处理；",
            "- 分歧已由第三人裁决；",
            "- 另存为新的 `frozen_v1`，不要修改原始 candidate 目录。",
        ]
    )
    content = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.write_text(content, encoding="utf-8")
    return content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.manifest, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
