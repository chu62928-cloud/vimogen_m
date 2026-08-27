#!/usr/bin/env python3
"""Summarize prompt-level paired drift statistics from three-way manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.mbench_threeway_stats import summarize_drift


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    args = parser.parse_args()
    paths = sorted(args.manifest_root.rglob("manifest.json"))
    if not paths:
        raise FileNotFoundError(f"no manifest.json below {args.manifest_root}")
    summary = summarize_drift(
        paths,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "manifest_count": len(paths),
        "conditions": sorted(summary["conditions"]),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
