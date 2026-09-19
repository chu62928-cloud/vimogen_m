#!/usr/bin/env python3
"""Summarize completed v0.4 ablation evaluations without changing them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=None,
        help="only include evaluations paired to this immutable protocol root",
    )
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.rglob("evaluation.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if value.get("protocol") != "vimogen_pelvis_guided_walk_v0_4_dose_first_contact_ablation":
            continue
        if args.protocol_root is not None and str(value.get("protocol_root")) != str(args.protocol_root):
            continue
        per_side = value.get("contact_gate", {}).get("per_side", {})
        left = per_side.get("left", {})
        right = per_side.get("right", {})
        rows.append({
            "run_root": str(path.parent),
            "mode": value.get("contact_mode"),
            "dose": value.get("target_delta_deg"),
            "status": value.get("status"),
            "dose_mae_deg": value.get("dose_gate", {}).get("mae_deg"),
            "dose_p95_deg": value.get("dose_gate", {}).get("p95_deg"),
            "contact_gate": value.get("contact_gate", {}).get("status"),
            "left_slide_p95_mm": (left.get("candidate", {}).get("sliding_m_per_frame", {}).get("p95") or 0.0) * 1000.0,
            "left_lift_p95_mm": (left.get("candidate", {}).get("lift_m", {}).get("p95") or 0.0) * 1000.0,
            "left_penetration_p95_mm": (left.get("candidate", {}).get("penetration_m", {}).get("p95") or 0.0) * 1000.0,
            "right_status": right.get("status"),
            "left_position_status": value.get("contact_gate", {}).get("position_velocity", {}).get("left", {}).get("position_status"),
            "left_velocity_status": value.get("contact_gate", {}).get("position_velocity", {}).get("left", {}).get("velocity_status"),
            "right_position_status": value.get("contact_gate", {}).get("position_velocity", {}).get("right", {}).get("position_status"),
            "right_velocity_status": value.get("contact_gate", {}).get("position_velocity", {}).get("right", {}).get("velocity_status"),
            "trust_region_violation": value.get("trust_region_violation"),
            "forward_distance_m": value.get("walk_metrics", {}).get("forward_distance_m"),
            "terminal_rebound_rms": value.get("terminal_rebound", {}).get("rms"),
        })
    rows.sort(key=lambda row: (float(row.get("dose") or 0.0), str(row.get("mode"))))
    print(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False))
    (args.root / "v0_4_ablation_summary.json").write_text(
        json.dumps({"protocol": "vimogen_pelvis_guided_walk_v0_4_dose_first_contact_ablation", "rows": rows}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
