#!/usr/bin/env python3
"""Freeze the audited M1--M7 definitions before any tuning result exists."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/m1_m7"
DEFAULT_OUTPUT = ROOT / "results/phase9/pelvis_m1_m7/protocol_v1"
EXPECTED_METHODS = tuple(f"M{index}" for index in range(1, 8))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_configs(config_root: Path = CONFIG_ROOT) -> dict[str, dict]:
    common = OmegaConf.to_container(
        OmegaConf.load(config_root / "common.yaml"), resolve=True
    )
    if not common.get("algorithm_definitions_frozen"):
        raise ValueError("algorithm definitions must be frozen")
    methods: dict[str, dict] = {}
    for method in EXPECTED_METHODS:
        path = config_root / f"{method.lower()}.yaml"
        config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if config.get("method") != method:
            raise ValueError(f"{path.name} method identity mismatch")
        if config.get("constraint_pack") != "C0":
            raise ValueError(f"{method} must support C0 before screening")
        methods[method] = config
    return {"common": common, "methods": methods}


def freeze(output: Path, *, code_commit: str) -> dict:
    validate_configs()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite frozen protocol: {output}")
    output.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in sorted(CONFIG_ROOT.glob("*.yaml")):
        destination = output / source.name
        shutil.copy2(source, destination)
        copied.append(
            {"path": source.name, "sha256": sha256(destination), "bytes": destination.stat().st_size}
        )
    record = {
        "protocol": "vimogen_pelvis_m1_m7_scale_v1",
        "status": "ALGORITHM_DEFINITIONS_FROZEN",
        "code_commit": code_commit,
        "source_plan_sha256": validate_configs()["common"]["source_plan_sha256"],
        "files": copied,
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output, code_commit=args.code_commit), indent=2))


if __name__ == "__main__":
    main()
