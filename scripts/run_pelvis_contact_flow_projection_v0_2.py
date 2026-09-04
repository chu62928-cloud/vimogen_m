#!/usr/bin/env python3
"""Run the v0.2 temporal-contact projection protocol."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sampling.pelvis_contact_flow_projection_v0_1 import TEMPORAL_CONTACT_PROTOCOL
from scripts.run_pelvis_contact_flow_projection_v0_1 import main


if __name__ == "__main__":
    main(default_protocol=TEMPORAL_CONTACT_PROTOCOL)
