#!/usr/bin/env python3
"""Add deterministic paired-bootstrap confidence intervals to recovery output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.representation_recovery import summarize  # noqa: E402


def run(input_path: Path, output_path: Path, repetitions: int = 2000) -> dict:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    summary = summarize(payload["records"], bootstrap_repetitions=repetitions)
    result = {
        "status": "VERIFIED_REPRESENTATION_RECOVERY_BOOTSTRAP_SUMMARY",
        "input": str(input_path),
        "record_count": len(payload["records"]),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": 20260824,
        "summary": summary,
    }
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()
    result = run(args.input, args.output, repetitions=args.repetitions)
    print(json.dumps(result["summary"]["paired_bootstrap_effects"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
