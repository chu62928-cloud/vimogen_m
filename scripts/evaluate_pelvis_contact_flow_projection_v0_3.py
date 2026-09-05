#!/usr/bin/env python3
"""Evaluate one v0.3 current-environment paired projection run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sampling.pelvis_contact_flow_projection_v0_1 import (  # noqa: E402
    CURRENT_ENV_PAIRED_PROTOCOL,
)
from scripts.evaluate_pelvis_contact_flow_projection_v0_1 import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol = json.loads((args.protocol_root / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("protocol") != CURRENT_ENV_PAIRED_PROTOCOL:
        raise ValueError("protocol-root is not a v0.3 current-environment paired protocol")
    print(json.dumps(evaluate(args.run_root, args.protocol_root, device=args.device), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
