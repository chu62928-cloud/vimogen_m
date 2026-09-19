#!/usr/bin/env python3
"""Run one sample94 full-sequence v0.4 ablation case."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sampling.pelvis_contact_flow_projection_v0_1 import DOSE_FIRST_CONTACT_ABLATION_PROTOCOL
from scripts.run_pelvis_contact_flow_projection_v0_1 import main as _main

DEFAULT_OUTPUT = ROOT / "results/phase8/pelvis_guided_walk_v0_4"
DEFAULT_PROTOCOL_ROOT = DEFAULT_OUTPUT / "protocol_current_env_sample94_v2"


def main() -> None:
    _main(
        default_protocol=DOSE_FIRST_CONTACT_ABLATION_PROTOCOL,
        default_output=DEFAULT_OUTPUT,
        default_protocol_root=DEFAULT_PROTOCOL_ROOT,
    )


if __name__ == "__main__":
    main()
