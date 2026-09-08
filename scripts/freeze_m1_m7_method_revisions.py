#!/usr/bin/env python3
"""Freeze non-overwriting M2/M3/M4-v2 method definitions before S1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/m1_m7"
DEFAULT_OUTPUT = ROOT / "results/phase9/pelvis_m1_m7/protocol_v2_revisions"
FILES = ("m2_v2.yaml", "m3_v2.yaml", "m4_v2.yaml")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_revisions(config_root: Path = CONFIG_ROOT) -> dict[str, dict[str, Any]]:
    configs = {
        name: OmegaConf.to_container(
            OmegaConf.load(config_root / name), resolve=True
        )
        for name in FILES
    }
    expected = {"m2_v2.yaml": "M2", "m3_v2.yaml": "M3", "m4_v2.yaml": "M4"}
    for name, method in expected.items():
        value = configs[name]
        if value.get("method") != method or value.get("method_version") != "v2":
            raise ValueError(f"{name} identity/version mismatch")
        if value.get("constraint_pack") != "C0":
            raise ValueError(f"{name} must remain C0 during S1")
        if value.get("zero_dose_policy") != "strict_paired_m0_bypass":
            raise ValueError(f"{name} must freeze strict zero-dose bypass")
    if configs["m2_v2.yaml"].get("selection_scope") != "independent_per_sample":
        raise ValueError("M2-v2 must select best states per sample")
    if int(configs["m3_v2.yaml"].get("max_projections", 0)) < 1:
        raise ValueError("M3-v2 must freeze a finite projection budget")
    if list(configs["m4_v2.yaml"].get("shooting_sigmas", [1])):
        raise ValueError("M4-v2 primary arm must be terminal-only")
    return configs


def freeze_revisions(
    output: Path, *, code_commit: str, config_root: Path = CONFIG_ROOT
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite method revisions: {output}")
    configs = validate_revisions(config_root)
    output.mkdir(parents=True)
    files = []
    for name in FILES:
        destination = output / name
        shutil.copy2(config_root / name, destination)
        files.append(
            {
                "path": name,
                "sha256": _sha256(destination),
                "method": configs[name]["method"],
                "method_version": "v2",
            }
        )
    manifest = {
        "protocol": "vimogen_pelvis_m1_m7_method_revisions_v2",
        "status": "METHOD_REVISIONS_FROZEN_FOR_S1",
        "code_commit": str(code_commit),
        "parent_protocol": "vimogen_pelvis_m1_m7_scale_v1",
        "files": files,
        "invariants": {
            "s0_v1_overwrite_forbidden": True,
            "zero_dose_strict_bypass": ["M2-v2", "M3-v2", "M4-v2"],
            "m2_single_batch_consistency_required": True,
            "m4_primary_arm": "terminal_only",
            "s2_tuning_forbidden": True,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_revisions(args.output, code_commit=args.code_commit)))


if __name__ == "__main__":
    main()
