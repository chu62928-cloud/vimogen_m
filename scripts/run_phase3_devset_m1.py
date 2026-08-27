#!/usr/bin/env python3
"""Run the pre-registered ``window_mid`` M1 candidate on one seed/command."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

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


def build_config(input_json: Path, seed: int, target_delta_deg: float, run_root: Path, noise_cache: Path, *, batch_size: int = 4):
    if target_delta_deg not in (5.0, 10.0):
        raise ValueError("the development M1 run accepts +5 or +10 degrees")
    config = OmegaConf.load(ROOT / "configs/t2m_infer.yaml")
    config.mode = "eval"
    config.mbench_name = f"phase3_devset_m1_window_mid_seed{seed}_{int(target_delta_deg):02d}deg"
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
        "artifact_dir": str(run_root / "m0_artifacts"),
        "initial_noise_path": None,
        "batch_invariant": True,
    }
    config.m1 = {
        "enabled": True,
        "target_delta_deg": float(target_delta_deg),
        "lambda_scale": 0.5,
        "sigma_min": 0.25,
        "sigma_max": 0.65,
        "angle_weight": 1.0,
        "hold_weight": 0.1,
        "max_correction_rms": 0.05,
        "heading_mode": "canonical_y",
        "artifact_dir": str(run_root / "m1_artifacts"),
        "diagnosis_variant": "window_mid",
        "diagnosis_rationale": "pre-registered middle-noise window; RMS cap remains 0.05",
    }
    return config


def run(input_json: Path, seed: int, target_delta_deg: float, output_root: Path, noise_cache: Path, *, batch_size: int = 4) -> dict:
    label = f"seed_{seed:03d}/delta_{int(target_delta_deg):02d}deg"
    run_root = output_root / label
    if run_root.exists():
        raise FileExistsError(run_root)
    run_root.mkdir(parents=True)
    config = build_config(input_json, seed, target_delta_deg, run_root, noise_cache, batch_size=batch_size)
    record = {
        "status": "RUNNING",
        "seed": int(seed),
        "target_delta_deg": float(target_delta_deg),
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
        "m1": {
            "variant": "window_mid",
            "lambda_scale": 0.5,
            "sigma_window": [0.25, 0.65],
            "max_correction_rms": 0.05,
            "heading_mode": "canonical_y",
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
    parser.add_argument("--target-delta-deg", type=float, required=True, choices=[5.0, 10.0])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--noise-cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.seed, args.target_delta_deg, args.output_root, args.noise_cache, batch_size=args.batch_size), indent=2))


if __name__ == "__main__":
    main()
