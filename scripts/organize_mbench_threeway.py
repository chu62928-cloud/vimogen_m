#!/usr/bin/env python3
"""Create three paired MBench input directories from one raw 276D run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.mbench_threeway import organize_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-count", type=int, default=450)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    payload = organize_directory(
        args.input_dir,
        args.output_root,
        condition=args.condition,
        seed=args.seed,
        expected_count=args.expected_count,
        verify_only=args.verify_only,
    )
    print(json.dumps(
        {key: payload[key] for key in ("status", "condition", "seed", "record_count", "error_count")},
        indent=2,
    ))


if __name__ == "__main__":
    main()
