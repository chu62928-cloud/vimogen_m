#!/usr/bin/env python3
"""Run one frozen-development-set M0 seed with the explicit sample-noise path.

This runner is text-only, uses the frozen 50-step M0 sampler, and writes an
isolated seed directory.  It never changes the legacy M0 outputs or the
candidate/frozen manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_config(input_json: Path, seed: int, run_root: Path, noise_cache: Path, *, batch_size: int = 4):
    config = OmegaConf.load(ROOT / "configs/t2m_infer.yaml")
    config.mode = "eval"
    config.mbench_name = f"phase3_devset_m0_seed{seed}"
    config.experiment.global_seed = int(seed)
    config.experiment.auto_resume = False
    config.experiment.result_dir = str(run_root / "trainer")
    config.dataloader.test_local_batch = int(batch_size)
    config.dataloader.num_workers = 2
    config.dataset.test_json_file_list = [str(input_json)]
    config.dataset.text_key = "prompt"
    config.m0 = {
        "noise_protocol": "sample_v1",
        "sample_noise_cache_dir": str(noise_cache),
        "artifact_dir": str(run_root / "artifacts"),
        "initial_noise_path": None,
        "batch_invariant": True,
    }
    # Keep M1 absent/disabled: this run is only the frozen M0 input for B0/B1/B2.
    config.m1 = {"enabled": False}
    return config


def run(input_json: Path, seed: int, output_root: Path, *, batch_size: int = 4) -> dict:
    run_root = output_root / f"seed_{seed:03d}"
    if run_root.exists():
        raise FileExistsError(run_root)
    run_root.mkdir(parents=True)
    noise_cache = output_root / "noise_cache"
    config = build_config(input_json, seed, run_root, noise_cache, batch_size=batch_size)
    record = {
        "status": "RUNNING",
        "seed": int(seed),
        "input": str(input_json),
        "input_sha256": _sha256(input_json),
        "output_root": str(run_root),
        "protocol": {
            "steps": 50,
            "shift": 5.0,
            "denoising_strength": 0.7,
            "noise_protocol": "vimogen-sample-noise-v1",
            "batch_invariant": True,
            "dtype": "bfloat16",
            "condition": "text_only",
        },
        "batch_size": int(batch_size),
    }
    (run_root / "run_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    try:
        from train_eval_vimogen import main

        main(config)
        record["status"] = "COMPLETED"
    except Exception as exc:
        record["status"] = "FAILED"
        record["error"] = repr(exc)
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        (run_root / "run_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.seed, args.output_root, batch_size=args.batch_size), indent=2))


if __name__ == "__main__":
    main()
