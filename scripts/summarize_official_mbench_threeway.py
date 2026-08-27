#!/usr/bin/env python3
"""Aggregate official MBench per-motion outputs with prompt-level pairing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.mbench_threeway_stats import summarize_official_mbench


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-record-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    paths = sorted(args.run_record_root.glob("**/run_record.json"))
    if len(paths) != 27:
        raise RuntimeError(f"expected 27 official MBench run records, found {len(paths)}")
    result = summarize_official_mbench(paths, bootstrap_repetitions=args.bootstrap_repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "run_count": len(paths), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
