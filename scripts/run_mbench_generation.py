#!/usr/bin/env python3
"""Run one frozen MBench raw-276D generation condition and seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_config(
    *,
    condition: str,
    target_delta_deg: float,
    seed: int,
    run_root: Path,
    noise_cache: Path,
    batch_size: int,
):
    if condition not in {"m0", "m1_plus5", "m1_plus10"}:
        raise ValueError(f"unsupported condition: {condition}")
    if condition == "m0" and target_delta_deg != 0.0:
        raise ValueError("M0 must use target_delta_deg=0")
    if condition != "m0" and target_delta_deg not in {5.0, 10.0}:
        raise ValueError("M1 must use +5 or +10 degrees")
    config = OmegaConf.load(ROOT / "configs/tm2m_infer.yaml")
    config.mode = "eval"
    config.mbench_name = f"publication_mbench_{condition}_seed{seed}"
    config.experiment.global_seed = int(seed)
    config.experiment.auto_resume = False
    config.experiment.eval_steps = 1
    config.experiment.result_dir = str(run_root / "trainer")
    config.dataloader.test_local_batch = int(batch_size)
    config.dataloader.num_workers = 8
    config.save_motion_visualizations = False
    config.m0 = {
        "noise_protocol": "sample_v1",
        "sample_noise_cache_dir": str(noise_cache),
        "artifact_dir": None,
        "initial_noise_path": None,
        "batch_invariant": True,
    }
    if condition == "m0":
        config.m1 = {"enabled": False}
    else:
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
            "artifact_dir": None,
            "diagnosis_variant": "window_mid",
            "diagnosis_rationale": "frozen publication MBench condition",
        }
    return config


def run(
    *,
    condition: str,
    target_delta_deg: float,
    seed: int,
    output_root: Path,
    noise_cache: Path,
    batch_size: int,
) -> dict:
    run_root = output_root / condition / f"seed_{seed:03d}"
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite {run_root}")
    run_root.mkdir(parents=True)
    config = build_config(
        condition=condition,
        target_delta_deg=target_delta_deg,
        seed=seed,
        run_root=run_root,
        noise_cache=noise_cache,
        batch_size=batch_size,
    )
    record = {
        "status": "RUNNING",
        "protocol": "vimogen_publication_mbench_generation_v1",
        "condition": condition,
        "target_delta_deg": float(target_delta_deg),
        "seed": int(seed),
        "input_manifest": str(ROOT / "data/meta_info/MBench_final.json"),
        "input_manifest_sha256": sha256_file(ROOT / "data/meta_info/MBench_final.json"),
        "output_root": str(run_root),
        "batch_size": int(batch_size),
        "noise_protocol": "vimogen-sample-noise-v1",
        "representation_reconciliation_enabled": False,
    }
    record_path = run_root / "run_record.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
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
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["m0", "m1_plus5", "m1_plus10"], required=True)
    parser.add_argument("--target-delta-deg", type=float, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--noise-cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(run(
        condition=args.condition,
        target_delta_deg=args.target_delta_deg,
        seed=args.seed,
        output_root=args.output_root,
        noise_cache=args.noise_cache,
        batch_size=args.batch_size,
    ), indent=2))


if __name__ == "__main__":
    main()
