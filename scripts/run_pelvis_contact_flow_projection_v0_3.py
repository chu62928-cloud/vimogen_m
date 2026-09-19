#!/usr/bin/env python3
"""Run the v0.3 current-environment paired projection protocol."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sampling.pelvis_contact_flow_projection_v0_1 import CURRENT_ENV_PAIRED_PROTOCOL
from scripts.run_pelvis_contact_flow_projection_v0_1 import main as _main

DEFAULT_OUTPUT = ROOT / "results/phase8/pelvis_contact_flow_projection_v0_3"


def main() -> None:
    _main(default_protocol=CURRENT_ENV_PAIRED_PROTOCOL, default_output=DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
