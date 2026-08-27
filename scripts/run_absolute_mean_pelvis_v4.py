#!/usr/bin/env python3
"""Run one non-overwriting anatomical-local v4 unit."""

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

PROTOCOL_NAME = "vimogen_absolute_mean_pelvis_v4_anatomical_local"
PROTOCOL_ROOT = ROOT / "results/phase6/absolute_mean_pelvis_v4"
INPUTS = {name: PROTOCOL_ROOT / "data" / filename for name, filename in {
    "smoke": "smoke1.json", "development": "development20.json",
    "primary": "mbench_primary_blind40.json", "robustness": "mbench_robustness450.json",
    "video12": "formal_video_cases12.json",
}.items()}
ALLOWED_STRENGTHS = {0.5, 1.0, 2.0}
ALLOWED_SHAPE_WEIGHTS = {0.05, 0.1, 0.2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_config(*, split: str, target_mean_deg: float, seed: int, guidance_strength: float, shape_weight: float, run_root: Path, noise_cache: Path, calibration: Path, batch_size: int):
    if split not in INPUTS or target_mean_deg not in {5.0, 10.0} or seed not in {0, 1, 2}:
        raise ValueError("invalid split, target, or seed")
    if guidance_strength not in ALLOWED_STRENGTHS or shape_weight not in ALLOWED_SHAPE_WEIGHTS:
        raise ValueError("parameter is outside the frozen development grid")
    config = OmegaConf.load(ROOT / "configs/tm2m_infer.yaml")
    config.mode = "eval"
    config.mbench_name = f"absolute_mean_v4_{split}_target{int(target_mean_deg)}_seed{seed}"
    config.experiment.global_seed = int(seed)
    config.experiment.auto_resume = False
    config.experiment.eval_steps = 1
    config.experiment.result_dir = str(run_root / "trainer")
    config.dataloader.test_local_batch = int(batch_size)
    config.dataloader.num_workers = 4
    config.dataset.test_json_file_list = [str(INPUTS[split])]
    if split == "development":
        config.dataset.text_key = "prompt"
    config.save_motion_visualizations = False
    config.m0 = {"noise_protocol": "sample_v1", "sample_noise_cache_dir": str(noise_cache), "artifact_dir": str(run_root / "m0_artifacts"), "initial_noise_path": None, "batch_invariant": True}
    config.m1 = {"enabled": False}
    config.absolute_mean_pelvis = {"enabled": True, "protocol": PROTOCOL_NAME, "target_mean_deg": float(target_mean_deg), "guidance_strength": float(guidance_strength), "sigma_min": 0.25, "sigma_max": 0.65, "mean_weight": 1.0, "shape_weight": float(shape_weight), "motion_weight": 0.1, "anti_cheat_weight": 1.0, "max_correction_rms": 0.05, "fusion_window": 9, "anchor_weight": 1.0, "terminal_enabled": True, "terminal_max_deg": 1.0, "soft_limit_deg": 2.0, "p95_limit_deg": 3.0, "calibration_path": str(calibration), "artifact_dir": str(run_root / "guided_artifacts")}
    config.representation = {"reconciliation": {"enabled": False}}
    return config


def run(args) -> dict:
    protocol = PROTOCOL_ROOT / "protocol.json"
    if not protocol.is_file() or not INPUTS[args.split].is_file():
        raise FileNotFoundError("v4 protocol and split must be frozen before generation")
    if not args.calibration.is_file():
        raise FileNotFoundError(args.calibration)
    run_base = PROTOCOL_ROOT / "runs" / args.split / f"strength_{args.guidance_strength:g}_shape_{args.shape_weight:g}" / f"seed_{args.seed:03d}" / f"target_{int(args.target_mean_deg):02d}deg"
    attempt = 1
    while (run_base / f"attempt_{attempt:02d}").exists():
        attempt += 1
    run_root = run_base / f"attempt_{attempt:02d}"
    run_root.mkdir(parents=True)
    config = build_config(split=args.split, target_mean_deg=args.target_mean_deg, seed=args.seed, guidance_strength=args.guidance_strength, shape_weight=args.shape_weight, run_root=run_root, noise_cache=args.noise_cache, calibration=args.calibration, batch_size=args.batch_size)
    record = {"status": "RUNNING", "protocol": PROTOCOL_NAME, "protocol_path": str(protocol), "protocol_sha256": sha256_file(protocol), "split": args.split, "input_manifest": str(INPUTS[args.split]), "input_manifest_sha256": sha256_file(INPUTS[args.split]), "target_mean_deg": float(args.target_mean_deg), "seed": int(args.seed), "guidance_strength": float(args.guidance_strength), "shape_weight": float(args.shape_weight), "calibration_path": str(args.calibration), "run_root": str(run_root), "consistency_boundary": "authoritative_pose_to_fk_to_J_to_dJ_dR_dT_to_276D"}
    (run_root / "run_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    OmegaConf.save(config, run_root / "resolved_config.yaml")
    started = time.perf_counter()
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29514"); os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("LOCAL_RANK", "0"); os.environ.setdefault("LOCAL_WORLD_SIZE", "1"); os.environ.setdefault("GROUP_RANK", "0"); os.environ.setdefault("GROUP_WORLD_SIZE", "1")
        from train_eval_vimogen import main as train_eval_main
        train_eval_main(config)
        record["status"] = "COMPLETED_GENERATION_PENDING_EVALUATION"
    except Exception as exc:
        record["status"] = "FAILED"; record["error"] = repr(exc); raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        (run_root / "run_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--split", choices=sorted(INPUTS), required=True); parser.add_argument("--target-mean-deg", type=float, choices=[5.0, 10.0], required=True); parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True); parser.add_argument("--guidance-strength", type=float, choices=sorted(ALLOWED_STRENGTHS), required=True); parser.add_argument("--shape-weight", type=float, choices=sorted(ALLOWED_SHAPE_WEIGHTS), required=True); parser.add_argument("--noise-cache", type=Path, required=True); parser.add_argument("--calibration", type=Path, required=True); parser.add_argument("--batch-size", type=int, default=4); args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
